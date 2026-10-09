"""Fixed-length rollout with lax.scan (PLAN.md §4).

Every rollout has n_steps steps of dy, so a batch of candidates or courses fits one vmap with no
ragged shapes. Courses shorter than n_steps * dy are padded: steps past the finish line are
masked out (dt = ds = 0) and the state is frozen there.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from skiopt import hazard, terrain
from skiopt.dynamics import PointMass2D
from skiopt.types import Course, SimParams, State, StepOutput, Summary


class Obs(NamedTuple):
    """What a policy sees at step k. Closed-loop policies build gate-relative features from
    `course`; open-loop policies only use k."""

    x: jnp.ndarray
    y: jnp.ndarray
    psi: jnp.ndarray
    v: jnp.ndarray
    theta: jnp.ndarray
    course: Course


PolicyFn = Callable[[jnp.ndarray, Obs, jnp.ndarray], jnp.ndarray]

_DYNAMICS = PointMass2D()


def n_steps_for(length_m: float, dy: float) -> int:
    return int(math.ceil(float(length_m) / dy))


def rollout(policy: PolicyFn, params, course: Course, p: SimParams, n_steps: int):
    """Run one candidate on one course. Returns (Summary, StepOutput with a leading time axis)."""
    state0 = State(
        x=jnp.asarray(course.x_start, dtype=jnp.result_type(float)),
        psi=jnp.zeros((), dtype=jnp.result_type(float)),
        v=jnp.asarray(p.v0, dtype=jnp.result_type(float)),
    )

    def body(state: State, k):
        y = k * p.dy
        theta = terrain.pitch(course.terrain, state.x, y)
        obs = Obs(state.x, y, state.psi, state.v, theta, course)
        kappa_cmd = policy(params, obs, k)
        new_state, out = _DYNAMICS.step(state, kappa_cmd, theta, p, y)
        active = (y < course.length).astype(out.dt.dtype)
        new_state = jax.tree.map(lambda a, b: jnp.where(active > 0, a, b), new_state, state)

        tel, r = out.telemetry, p.risk
        k_req = hazard.kappa_required(state.x, y, state.psi, course, r.clearance, r.kappa_req_cap)
        h = hazard.hazard_terms(tel.G, tel.kappa, k_req, state.v, y, r)
        broken, wall = hazard.hard_limits(tel.G, state.v, state.x, course.piste_half_width, r)
        ds = out.ds * active
        out = StepOutput(dt=out.dt * active, ds=ds, active=active, telemetry=tel,
                         hazard=h * ds, hard_dnf=broken * active, wall=wall * ds, kappa_req=k_req)
        return new_state, out

    final, traj = jax.lax.scan(body, state0, jnp.arange(n_steps))
    T = jnp.sum(traj.dt)
    D = jnp.sum(traj.ds)
    G = jnp.where(traj.active > 0, traj.telemetry.G, 0.0)
    # Telemetry x at step k is the position at y = k * dy, so x at a gate line is a lookup.
    gate_idx = jnp.clip(jnp.round(course.gates.y / p.dy).astype(jnp.int32), 0, n_steps - 1)
    H_by_cause = jnp.sum(traj.hazard, axis=0)
    summary = Summary(T=T, D=D, v_mean=D / T, v_end=final.v, max_G=jnp.max(G),
                      x_gates=traj.telemetry.x[gate_idx], H=jnp.sum(H_by_cause), H_by_cause=H_by_cause,
                      hard_dnf=jnp.max(traj.hard_dnf), wall=jnp.sum(traj.wall))
    return summary, traj
