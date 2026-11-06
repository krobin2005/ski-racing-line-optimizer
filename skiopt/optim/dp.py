"""Dynamic-programming global baseline on a lattice of apex choices (docs/ROADMAP_3D.md §5).

The apex-and-arc line is fixed by one apex per turning gate. Here each apex may only take one of a
small menu of positions: `margins` (metres outside the turning pole, clearance included) x `shifts`
(metres along the fall line from the gate line). Between two apexes the line is two circular arcs
with heading along the course at both ends, so a segment depends only on its two apex choices and
the entry speed. The search runs forward gate by gate:

  - every segment is simulated with the real 3D physics (`sim3d.advance`, identical to full runs),
    starting at the left apex with heading 0 and the label's speed;
  - a label carries (cost, exit speed, back-pointer); cost = time + T_DNF x hazard (the first-order
    expected race score), and any missed gate or hard limit makes a segment infeasible;
  - for each (gate, choice) the cheapest label per exit-speed band survives, thinned to `keep`
    bands across the speed range (slow labels matter: they keep the steep turns feasible).

The best path is converted to arc-line parameters and scored with the ordinary full rollout, so its
result is exactly comparable with every other method. Optional: refine with Adam on the remaining
budget (`refine: adam`), the "global then local" hybrid. 3D courses (`course.kind: gran_risa`) only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from skiopt import sim3d
from skiopt.budget import Budget
from skiopt.courses import turning_gates
from skiopt.optim.common import Problem, Tracker
from skiopt.policies import curvature_from_line


def _arc_segment_x(ya, xa, yb, xb, k):
    """x at integer grid points k of the two-arc line between apexes (ya, xa) and (yb, xb)."""
    L, d = yb - ya, xb - xa
    w = jnp.sqrt(d**2 + 1e-8)
    R = (L**2 + w**2) / (4 * w)

    def bulge(u):
        u = jnp.clip(u, 0.0, L / 2)
        return u**2 / (R + jnp.sqrt(jnp.maximum(R**2 - u**2, 1e-9)))

    t = k.astype(jnp.float32) if k.dtype != jnp.float64 else k
    first = t < ya + L / 2
    x = jnp.where(first, xa + (d / w) * bulge(t - ya), xb - (d / w) * bulge(yb - t))
    return jnp.where(t >= yb, xb, x)


def make_segment_sim(course, p, n_max: int, kappa_c_full, p3=None):
    """Batched simulator for segments: (k0, k1, xa, xb, v0) -> (time, hazard, hard, miss, v_end).
    Segments start at an apex (heading 0) at grid row k0 and end at row k1 (the next apex, or the
    finish with xb = xa for the straight run-out)."""
    g = course.gates
    clearance = p.risk.clearance

    def one(k0, k1, xa, xb, v0):
        i = jnp.arange(n_max + 1)
        k = k0 + i
        x = _arc_segment_x(k0.astype(jnp.float32), xa, k1.astype(jnp.float32), xb, k.astype(jnp.float32))
        kc = jnp.take(kappa_c_full, jnp.clip(k, 0, kappa_c_full.shape[0] - 1))
        kappa = curvature_from_line(x, 1.0, kc)
        n_seg = k1 - k0

        def body(state, j):
            new_state, out = sim3d.advance(state, kappa[j], k0 + j, course, p, p3 or sim3d.Params3D())
            live = j < n_seg
            new_state = jax.tree.map(lambda a, b: jnp.where(live, a, b), new_state, state)
            m = live.astype(out.dt.dtype)
            return new_state, (out.dt * m, jnp.sum(out.hazard) * m, out.hard_dnf * m, out.telemetry.x)

        state0 = sim3d.State3D(xa, jnp.zeros_like(xa), v0)
        final, (dt, h, hard, xs) = jax.lax.scan(body, state0, jnp.arange(n_max))
        # Gates whose line falls inside this segment: same pass window as the objectives.
        idx = g.y.astype(jnp.int32) - k0
        inside = (idx >= 0) & (idx < n_seg) & (g.valid > 0)
        x_at = xs[jnp.clip(idx, 0, n_max - 1)]
        near = g.x_pole + g.side * clearance
        far = g.x_pole + g.side * g.width
        lo, hi = jnp.minimum(near, far), jnp.maximum(near, far)
        miss = jnp.sum(jnp.where(inside, jax.nn.relu(lo - x_at) + jax.nn.relu(x_at - hi), 0.0))
        # The lattice assumes every segment ends on its apex (position xb, heading 0). If the arcs
        # needed curvature beyond the 12 m clip the skier drifts off, and a full run would carry that
        # drift into the next segment, so such a segment counts as a miss of its own end apex.
        drift = jnp.abs(final.n - xb) + 10.0 * jnp.abs(final.psi)
        miss = miss + jnp.where(drift > 0.3, drift, 0.0)
        return jnp.sum(dt), jnp.sum(h), jnp.max(hard), miss, final.v

    return jax.jit(jax.vmap(one))


def _pareto_thin(cost, v, keep: int, v_bin: float):
    """Indices of surviving labels: the cheapest label in each exit-speed band, thinned to `keep`
    bands spread across the speed range (always including the cheapest overall).

    Not a (cost, speed) Pareto front: on steep terrain *more* speed is not always better, because a
    slower entry is what makes the next turns feasible under the G and edge limits, so slow labels
    must survive even when a faster, cheaper one exists."""
    ok = np.isfinite(cost)
    if not ok.any():
        return np.zeros(0, int)
    idx = np.flatnonzero(ok)
    bins = np.floor(v[idx] / v_bin)
    best = {}
    for i, b in zip(idx, bins):
        if b not in best or cost[i] < cost[best[b]]:
            best[b] = i
    by_speed = np.array([best[b] for b in sorted(best)])
    if len(by_speed) <= keep:
        return by_speed
    pick = set(by_speed[np.unique(np.round(np.linspace(0, len(by_speed) - 1, keep - 1)).astype(int))])
    pick.add(int(by_speed[np.argmin(cost[by_speed])]))
    return np.array(sorted(pick))


def dp(prob: Problem, method: dict, budget: Budget, key, log, init=None):
    del init
    policy, p = prob.policy, prob.sim_params
    if getattr(policy, "name", None) != "arc" or p is None or not hasattr(prob.courses, "terrain") \
            or not hasattr(prob.courses.terrain, "F"):
        raise ValueError("dp needs `policy: arc` on a 3D course (course.kind: gran_risa)")
    course = jax.tree.map(lambda a: a[0], prob.courses)
    margins = np.asarray(method.get("margins", [0.5, 0.8, 1.2, 1.8, 2.6, 3.6]), float)
    shifts = np.asarray(method.get("shifts", [-5, -3, -1, 0, 1, 2]), float)
    keep, v_bin = int(method.get("keep", 8)), float(method.get("v_bin", 0.5))
    choices = [(m, s) for m in margins for s in shifts]
    n_choice = len(choices)

    side = np.asarray(policy.side)
    pole = np.asarray(policy.pole_x)
    gate_y = np.asarray(policy.apex_y)
    n_apex = policy.n_apex
    ay = np.round(gate_y[:, None] + np.array([s for _, s in choices])[None, :]).astype(int)  # (apex, choice)
    ax = pole[:, None] + side[:, None] * np.array([m for m, _ in choices])[None, :]
    finish = int(float(course.length))
    t_dnf = prob.obj.t_dnf_factor * float(course.length) / prob.obj.ref_speed

    spans = [ay[0].max(), finish - ay[-1].min()] + [ay[j].max() - ay[j - 1].min() for j in range(1, n_apex)]
    seg_max = int(max(spans) + 2)
    kappa_c = policy.kappa_c if policy.kappa_c is not None else jnp.zeros(policy.n_steps + 1)
    sim = make_segment_sim(course, p, seg_max, jnp.asarray(kappa_c), prob.params3d)
    tracker_cost = 0.0

    def run(k0, k1, xa, xb, v0):
        nonlocal tracker_cost
        out = sim(jnp.asarray(k0, jnp.int32), jnp.asarray(k1, jnp.int32), jnp.asarray(xa, jnp.float32),
                  jnp.asarray(xb, jnp.float32), jnp.asarray(v0, jnp.float32))
        tracker_cost += float(np.sum(np.asarray(k1) - np.asarray(k0))) / prob.n_steps
        dt, h, hard, miss, v_end = (np.asarray(a, float) for a in out)
        cost = dt + t_dnf * h
        cost[(hard > 0) | (miss > 0) | ~np.isfinite(cost)] = np.inf
        return cost, v_end

    # First segment: from the start (0, x_start) to every choice of apex 0.
    cost, v = run(np.zeros(n_choice, int), ay[0], np.full(n_choice, policy.x_start), ax[0], np.full(n_choice, p.v0))
    labels = []  # per apex: list over choices of (cost array, v array, parent (choice, label) array)
    layer = []
    for c in range(n_choice):
        layer.append((np.array([cost[c]]), np.array([v[c]]), np.array([[-1, -1]])))
    labels.append(layer)

    for j in range(1, n_apex):
        prev = labels[-1]
        src = [(c, l) for c in range(n_choice) for l in _pareto_thin(prev[c][0], prev[c][1], keep, v_bin)]
        if not src:
            raise RuntimeError(f"dp: no feasible label reached apex {j - 1}")
        pc = np.array([c for c, _ in src])
        pl = np.array([l for _, l in src])
        base = np.array([prev[c][0][l] for c, l in src])
        v_in = np.array([prev[c][1][l] for c, l in src])
        # every (source label, next choice) pair
        P, Q = np.meshgrid(np.arange(len(src)), np.arange(n_choice), indexing="ij")
        P, Q = P.ravel(), Q.ravel()
        k0, k1 = ay[j - 1, pc[P]], ay[j, Q]
        valid = k1 > k0 + 4
        seg_cost, seg_v = run(k0, np.where(valid, k1, k0 + 5), ax[j - 1, pc[P]], ax[j, Q], v_in[P])
        seg_cost[~valid] = np.inf
        total = base[P] + seg_cost
        layer = []
        for c in range(n_choice):
            sel = np.flatnonzero(Q == c)
            layer.append((total[sel], seg_v[sel], np.stack([pc[P[sel]], pl[P[sel]]], 1)))
        labels.append(layer)

    # Run-out from the last apex straight to the finish.
    last = labels[-1]
    src = [(c, l) for c in range(n_choice) for l in _pareto_thin(last[c][0], last[c][1], keep, v_bin)]
    pc = np.array([c for c, _ in src])
    pl = np.array([l for _, l in src])
    base = np.array([last[c][0][l] for c, l in src])
    v_in = np.array([last[c][1][l] for c, l in src])
    end_cost, _ = run(ay[-1, pc], np.full(len(src), finish), ax[-1, pc], ax[-1, pc], v_in)
    b = int(np.argmin(base + end_cost))
    if not np.isfinite(base[b] + end_cost[b]):
        raise RuntimeError("dp: no feasible path to the finish on this lattice")

    # Back-track the choices.
    path = [None] * n_apex
    c, l = pc[b], pl[b]
    for j in range(n_apex - 1, -1, -1):
        path[j] = c
        c, l = labels[j][c][2][l]

    m_sel = np.array([choices[c][0] for c in path])
    s_sel = np.array([choices[c][1] for c in path])
    u = np.log(np.expm1(m_sel - policy.clearance))
    frac = (s_sel - policy.shift_lo) / (policy.shift_hi - policy.shift_lo)
    v_par = np.log(frac / (1 - frac))
    params = jnp.asarray(np.concatenate([u, v_par]) if policy.apex_shift else u, dtype=jnp.float32)

    budget.charge_rollouts(tracker_cost)
    tracker = Tracker(log, budget)
    ev = prob.evaluate(params[None], prob.courses)
    budget.charge_rollouts(prob.n_courses)
    tracker.update(params[None], ev)
    if method.get("refine") == "adam" and budget.remaining > 1000:
        from skiopt.optim.gradient import adam

        best2, score2, hist2 = adam(prob, {**method.get("adam", {}), "restarts": 1}, budget, key, log, init=params)
        if score2 < tracker.best_score:
            tracker.best_params, tracker.best_score = best2, score2
        tracker.history += hist2
    return tracker.result()
