"""The Gran Risa as a 3D course (docs/ROADMAP_3D.md): real terrain, real piste edges, and a
FIS-valid GS set with the gate count of the 21 December 2025 World Cup race, run 1.

FIS result, Alta Badia men's GS, 21 Dec 2025 (codex 0045), run 1: 52 gates, 51 turning gates
(51 direction changes, so one delay gate), course setter Nathan Bryant (USA); start 1,868 m,
finish 1,420 m, vertical drop 448 m, homologation 12080/10/16; fastest run 1:16.00 (M. Schwarz).
Exact gate positions are not published, so the set is generated: same gate count, FIS rules,
the generator's turn-radius rule, centred on the real piste and kept inside its real edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from skiopt import courses as C
from skiopt import terrain3d as T
from skiopt.sim3d import Course3D, grid_from
from skiopt.types import Gates

RACE_2025_RUN1 = {
    "event": "Audi FIS Ski World Cup, Alta Badia (ITA), men's giant slalom, 21 Dec 2025, run 1",
    "gates": 52, "turning_gates": 51, "course_setter": "Nathan Bryant (USA)",
    "start_m": 1868.0, "finish_m": 1420.0, "vertical_drop_m": 448.0, "homologation": "12080/10/16",
    "fastest_run_s": 76.00, "fastest_run_by": "Marco Schwarz (AUT)",
}


def piste_edges(frame: T.CourseFrame, Eo, No, reach: float = 70.0):
    """For every s, where the cross-course line meets the piste outline: (n_right, n_left)."""
    lo, hi = np.full(len(frame.s), np.nan), np.full(len(frame.s), np.nan)
    P = np.stack([Eo, No], 1)
    A, d = P[:-1], P[1:] - P[:-1]
    for i in range(len(frame.s)):
        c = np.array([frame.E[i], frame.N[i]])
        l = np.array([-frame.ty[i], frame.tx[i]])
        den = d[:, 0] * l[1] - d[:, 1] * l[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((c[0] - A[:, 0]) * l[1] - (c[1] - A[:, 1]) * l[0]) / den
            n = ((c[0] - A[:, 0]) * d[:, 1] - (c[1] - A[:, 1]) * d[:, 0]) / den
        ok = (t >= 0) & (t <= 1) & (np.abs(n) < reach)
        if ok.sum() >= 2:
            lo[i], hi[i] = n[ok].min(), n[ok].max()
    s = frame.s
    good = np.isfinite(lo)
    return np.interp(s, s[good], lo[good]), np.interp(s, s[good], hi[good])


@dataclass
class Course3DBuild:
    course: Course3D            # what the simulator runs
    spec: C.CourseSpec          # gates in course coordinates (gate_y = s, gate_x = n), for policies/plots
    validation_spec: C.CourseSpec  # gates along the surface (y = slope distance) for the FIS validator
    frame: T.CourseFrame
    terrain: T.CourseTerrain
    piste_lo: np.ndarray
    piste_hi: np.ndarray
    meta: dict = field(default_factory=dict)


def _surface_distance(frame, Zc):
    return np.r_[0.0, np.cumsum(np.hypot(np.diff(frame.s), np.diff(Zc)))]


def build_gran_risa(seed: int = 0, smooth_m: float = 4.0, race: dict = RACE_2025_RUN1, fis: dict | None = None,
                    max_attempts: int = 400) -> Course3DBuild:
    fis = fis or C.load_fis()
    rules, model, g = fis["rules"], fis["model"], fis["generator"]
    hf = T.HeightField.load(T.DATA / "gran_risa" / "terrain.npz")
    E, N = T.osm_centreline(T.DATA / "gran_risa" / "osm_pistes.json")
    after = 40.0
    frame, report = T.build_frame(E, N, hf, race["start_m"], race["finish_m"], before=0.0, after=after)
    Eo, No = T.osm_outline(T.DATA / "gran_risa" / "osm_pistes.json")
    lo, hi = piste_edges(frame, Eo, No)
    half = float(np.ceil(max(np.abs(lo).max(), np.abs(hi).max()) + 6))
    ct = T.resample(hf, frame, half_width=half)
    Zc = ct.Z[:, int(np.argmin(np.abs(ct.n)))]
    s_finish = float(np.floor(frame.length - after))
    ell = _surface_distance(frame, Zc)
    # The piste middle from a coarse outline polygon jumps by metres between vertices; gates follow a
    # smoothed middle (40 m Gaussian) so the set's own lateral swings stay those of its turn radii.
    from scipy.ndimage import gaussian_filter1d

    mid = gaussian_filter1d(0.5 * (lo + hi), sigma=40.0 / float(frame.s[1] - frame.s[0]), mode="nearest")
    rng = np.random.default_rng(seed)

    n_turns = race["turning_gates"]
    n_delay = race["gates"] - race["turning_gates"]
    margin = model["body_clearance_m"] + g["line_margin_m"]
    edge_room = 1.5  # outside poles stay this far inside the mapped piste edge

    for _attempt in range(max_attempts):
        first = rng.uniform(*g["first_gate_y_m"])
        available = s_finish - model["finish_margin_m"] - first
        spacing = C._section_spacings(rng, g, n_turns - 1)
        delay_after = rng.choice(np.arange(1, n_turns - 2), size=n_delay, replace=False) if n_delay else []
        for j in delay_after:
            spacing[j] = rng.uniform(*g["delay_spacing_m"])
        spacing *= available / spacing.sum() * rng.uniform(0.95, 1.0)
        normal = np.setdiff1d(np.arange(n_turns - 1), delay_after)
        if spacing[normal].min() < 15.0 or (n_delay and spacing[delay_after].min() < 32.0):
            continue
        turn_s = np.round(first + np.r_[0.0, np.cumsum(spacing)])
        radius = np.clip(rng.normal(*g["turn_radius_m"][:2], size=n_turns - 1), *g["turn_radius_m"][2:])
        dx = C.offsets_for_radius(spacing, radius, margin, g["min_offset_m"])
        sides = rng.choice([-1.0, 1.0]) * (-1.0) ** np.arange(n_turns)
        # Each turning pole sits half its incoming offset from the piste's middle, on its side.
        idx = turn_s.astype(int)
        turn_x = mid[idx] + sides * 0.5 * np.r_[dx[0], dx]
        width = rng.uniform(*g["gate_width_m"], size=n_turns).round(1)
        gy, gx, gside, gw = [list(a) for a in (turn_s, turn_x, sides, width)]
        for j in sorted(delay_after, reverse=True):
            y_d = float(np.round(0.5 * (turn_s[j] + turn_s[j + 1])))
            line_x = 0.5 * (turn_x[j] + turn_x[j + 1])
            gy.insert(j + 1, y_d)
            gx.insert(j + 1, line_x - sides[j] * (model["body_clearance_m"] + rng.uniform(*g["delay_pole_inset_m"])))
            gside.insert(j + 1, sides[j])
            gw.insert(j + 1, round(float(rng.uniform(*g["gate_width_m"])), 1))
        gy, gx, gside, gw = (np.array(a, float) for a in (gy, gx, gside, gw))
        outer = gx + gside * gw
        i = gy.astype(int)
        inside = (np.minimum(gx, outer) >= lo[i] + edge_room) & (np.maximum(gx, outer) <= hi[i] - edge_room)
        if not inside.all():
            continue
        first_colour = int(rng.integers(2))
        colours = [rules["colours"][(first_colour + k) % 2] for k in range(len(gy))]
        # Validation copy: gates along the surface from the start, pitch profile from the centreline.
        y_val = np.round(np.interp(gy, frame.s, ell))
        theta = np.arctan(-np.gradient(Zc, frame.s))
        L_val = float(np.interp(s_finish, frame.s, ell))
        val = C.CourseSpec(name="gran_risa", length=L_val, y_knots=ell[ell <= L_val + 1], theta_knots=theta[ell <= L_val + 1],
                           gate_y=y_val, gate_x=gx, gate_side=gside, gate_width=gw, gate_colour=colours,
                           piste_width=float(np.median(hi - lo)))
        # The validator's own piste rule assumes a straight 40 m piste centred on x = 0; the real-edge
        # check above replaces it. Every other rule, including the turn-radius rule, applies as is.
        violations = [v for v in C.validate(val, fis) if v.rule != "piste"]
        if violations:
            continue
        spec = C.CourseSpec(name="gran_risa", length=s_finish, y_knots=frame.s, theta_knots=theta,
                            gate_y=gy, gate_x=gx, gate_side=gside, gate_width=gw, gate_colour=colours,
                            x_start=float(mid[0]), seed=seed)
        grid = grid_from(ct, smooth_m=smooth_m)
        n = len(gy)
        course = Course3D(
            length=jnp.asarray(s_finish), x_start=jnp.asarray(float(mid[0])),
            piste_half_width=jnp.asarray(float(np.median(hi - lo) / 2)), terrain=grid,
            gates=Gates(y=jnp.asarray(gy), x_pole=jnp.asarray(gx), side=jnp.asarray(gside),
                        width=jnp.asarray(gw), valid=jnp.ones(n)),
            piste_lo=jnp.asarray(lo), piste_hi=jnp.asarray(hi),
        )
        meta = dict(race, seed=seed, smooth_m=smooth_m, frame_report=report, attempts=_attempt + 1,
                    surface_length_m=L_val, vertical_drop_model_m=float(Zc[0] - np.interp(s_finish, frame.s, Zc)),
                    piste_width_median_m=float(np.median(hi - lo)), piste_width_min_m=float((hi - lo).min()),
                    validation=[(v.rule, v.message) for v in C.validate(val, fis) if v.rule != "piste"])
        return Course3DBuild(course, spec, val, frame, ct, lo, hi, meta)
    raise RuntimeError(f"no valid Gran Risa set for seed {seed} in {max_attempts} attempts")
