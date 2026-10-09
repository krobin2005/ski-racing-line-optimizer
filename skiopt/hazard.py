"""DNF model (PLAN.md §5): four hazard terms, hard limits and smooth-objective walls.

Hazard h is a crash risk per metre, accumulated along the run: P_finish = exp(-sum h * ds).
Everything is deterministic and differentiable, so gradient methods see risk directly.

    h_load  = c_load  * softplus(G - G_safe)          edge washout / body can't hold the load
    h_tight = c_tight * softplus(|kappa| - 1/r_skid)  radius under ~14 m: edge catch
    h_late  = c_late  * softplus(kappa_req - 1/r_skid) line too round: next gate needs a tight turn
    h_slow  = c_slow  * softplus(v_slow - v)          too slow: no edge pressure (after the start zone)

Hard limits (G > G_max, v < v_stop, leaving the piste) end the run in the hard score and add a
steep softplus wall to the smooth objective. Every term and limit has a switch in RiskParams.
"""

from __future__ import annotations

import jax.numpy as jnp

from skiopt.dynamics import softplus_b
from skiopt.types import Course, RiskParams

N_TERMS = 4
TERM_NAMES = ("load", "tight", "late", "slow")


def kappa_required(x, y, psi, course: Course, clearance, cap):
    """Curvature of the arc tangent to the current heading that reaches the nearest point of the
    next gate's pass window: kappa_req = 2 e_perp / d^2 (PLAN.md §5). 0 if running straight on
    already lands inside the window, or if no gate is ahead."""
    g = course.gates
    if g.y.shape[-1] == 0:  # gate-less test slope
        return jnp.zeros_like(x)
    ahead = (g.valid > 0) & (g.y > y)
    big = 1e9
    y_next = jnp.min(jnp.where(ahead, g.y, big))
    idx = jnp.argmin(jnp.where(ahead, g.y, big))
    has_next = y_next < big / 2

    near = g.x_pole[idx] + g.side[idx] * clearance
    far = g.x_pole[idx] + g.side[idx] * g.width[idx]
    lo, hi = jnp.minimum(near, far), jnp.maximum(near, far)

    dy_ahead = jnp.where(has_next, y_next - y, 1.0)
    x_straight = x + jnp.tan(psi) * dy_ahead
    x_target = jnp.clip(x_straight, lo, hi)
    rx, ry = x_target - x, dy_ahead
    # Lateral offset of the target from the heading line, in the turn-normal direction.
    e_perp = rx * jnp.cos(psi) - ry * jnp.sin(psi)
    d2 = rx**2 + ry**2
    k_req = 2.0 * jnp.abs(e_perp) / d2
    return jnp.where(has_next, jnp.minimum(k_req, cap), 0.0)


def hazard_terms(G, kappa, kappa_req, v, y, r: RiskParams):
    """Per-metre hazard of each term, shape (4,): load, tight, late, slow."""
    k_skid = 1.0 / r.r_skid
    load = r.load_on * r.c_load * softplus_b(G - r.G_safe, r.beta_load)
    tight = r.tight_on * r.c_tight * softplus_b(jnp.abs(kappa) - k_skid, r.beta_tight)
    late = r.late_on * r.c_late * softplus_b(kappa_req - k_skid, r.beta_late)
    in_start_zone = y < r.slow_after
    slow = r.slow_on * r.c_slow * softplus_b(r.v_slow - v, r.beta_slow) * jnp.where(in_start_zone, 0.0, 1.0)
    return jnp.stack([load, tight, late, slow])


def hard_limits(G, v, x, half_width, r: RiskParams):
    """(broken, wall): broken = 1.0 if any enabled hard limit is crossed at this point; wall is the
    smooth-objective penalty per metre (weight * softplus of each enabled excess)."""
    g_excess = G - r.G_max
    v_excess = r.v_stop - v
    x_excess = jnp.abs(x) - half_width
    broken = jnp.maximum(
        jnp.maximum(r.gmax_on * (g_excess > 0), r.vstop_on * (v_excess > 0)), r.piste_on * (x_excess > 0)
    )
    wall = r.wall_weight * (
        r.gmax_on * softplus_b(g_excess, r.wall_beta)
        + r.vstop_on * softplus_b(v_excess, r.wall_beta)
        + r.piste_on * softplus_b(x_excess, r.wall_beta)
    )
    return broken.astype(G.dtype), wall


def turn_risk(radius, speed, theta, r: RiskParams, g=9.81, turn_deg=90.0, n=181):
    """DNF probability of one constant-radius, constant-speed turn sweeping `turn_deg` across the
    fall line (heading +turn/2 to -turn/2) on pitch theta: 1 - exp(-integral h ds).
    Returns (total, per-term) with the trailing axis of size 4 for per-term. Used by the risk map."""
    radius, speed = jnp.asarray(radius)[..., None], jnp.asarray(speed)[..., None]
    half = jnp.radians(turn_deg) / 2
    psi = jnp.linspace(half, -half, n)                        # turning left through the fall line
    kappa = -1.0 / radius
    a_net = kappa * speed**2 + g * jnp.sin(theta) * jnp.sin(psi)
    G = jnp.sqrt(a_net**2 + (g * jnp.cos(theta)) ** 2) / g
    ds = radius * (2 * half) / (n - 1)
    k_skid = 1.0 / r.r_skid
    load = r.load_on * r.c_load * softplus_b(G - r.G_safe, r.beta_load)
    tight = r.tight_on * r.c_tight * softplus_b(jnp.abs(kappa) - k_skid, r.beta_tight) * jnp.ones_like(psi)
    slow = r.slow_on * r.c_slow * softplus_b(r.v_slow - speed, r.beta_slow) * jnp.ones_like(psi)
    per_term_H = jnp.stack([
        jnp.trapezoid(load, dx=1.0, axis=-1) * ds[..., 0],
        jnp.trapezoid(tight, dx=1.0, axis=-1) * ds[..., 0],
        jnp.zeros(jnp.broadcast_shapes(load.shape[:-1])),     # late: depends on gates, not one turn
        jnp.trapezoid(slow, dx=1.0, axis=-1) * ds[..., 0],
    ], axis=-1)
    hard = (r.gmax_on * jnp.any(G > r.G_max, axis=-1)) > 0
    total = jnp.where(hard, 1.0, -jnp.expm1(-per_term_H.sum(-1)))
    return total, -jnp.expm1(-per_term_H)
