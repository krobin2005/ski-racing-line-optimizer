"""Objectives and the batched evaluator every optimizer calls (PLAN.md §5, §9).

A gate is passed when x at the gate line lies between the turning pole plus the body clearance
and the outside pole. A crash and a missed gate both count as a DNF:

    E[score]    = P_finish * T + P_DNF * T_DNF,   P_finish = exp(-H),  T_DNF = 1.5 * T_ref
    smooth_loss = E[score] + lam_gate * sum softplus(miss)^2 + walls       (gradient methods)
    hard_score  = DQ_BASE + sum miss + hard excess   if any gate is missed or a hard limit
                                                     (G_max, v_stop, piste) is broken
                = E[score]                           otherwise           (evolution, reporting)

Every kind of DNF scores DQ_BASE plus a graded measure of how badly it failed, so a line that
leaves the piste can never outrank one that only misses a gate, and evolution still sees progress
among disqualified runs. The hard excess is the wall integral divided by its weight (excess x m).

T_ref is a reference winning time, course length / ref_speed (60 km/h by default), so T_DNF is
above any sensible finish time and each extra 1% of DNF risk costs about 0.01 * (T_DNF - T)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from skiopt.dynamics import softplus_b
from skiopt.rollout import rollout
from skiopt.types import Course, SimParams, Summary


class ObjParams(NamedTuple):
    dq_base: float = 1000.0
    lam_gate: float = 1.0
    beta_miss: float = 5.0
    clearance: float = 0.4
    t_dnf_factor: float = 1.5
    ref_speed: float = 60.0 / 3.6
    gate_miss_on: float = 1.0
    wall_weight: float = 100.0

    @classmethod
    def from_config(cls, cfg: dict) -> "ObjParams":
        o = cfg["objective"]
        gate_miss_on = float(cfg.get("risk", {}).get("hard", {}).get("gate_miss", {}).get("enabled", True))
        return cls(dq_base=float(o["dq_base"]), lam_gate=float(o["lam_gate"]),
                   beta_miss=float(o["beta_miss"]),
                   clearance=float(cfg["fis"]["model"]["body_clearance_m"]),
                   t_dnf_factor=float(o["t_dnf_factor"]), ref_speed=float(o["ref_speed_kmh"]) / 3.6,
                   gate_miss_on=gate_miss_on,
                   wall_weight=float(cfg.get("risk", {}).get("wall", {}).get("weight", 100.0)))


def pass_window(course: Course, clearance):
    """(lo, hi) x-interval of each gate's pass window, clearance included."""
    g = course.gates
    near = g.x_pole + g.side * clearance
    far = g.x_pole + g.side * g.width
    return jnp.minimum(near, far), jnp.maximum(near, far)


def gate_miss(summary: Summary, course: Course, obj: ObjParams):
    """Per-gate miss distance (m): 0 inside the pass window, distance to it outside.
    Returns (hard, smooth); padding gates contribute 0."""
    lo, hi = pass_window(course, obj.clearance)
    x = summary.x_gates
    on = course.gates.valid * obj.gate_miss_on
    hard = (jax.nn.relu(lo - x) + jax.nn.relu(x - hi)) * on
    smooth = (softplus_b(lo - x, obj.beta_miss) + softplus_b(x - hi, obj.beta_miss)) * on
    return hard, smooth


def t_dnf(course: Course, obj: ObjParams):
    return obj.t_dnf_factor * course.length / obj.ref_speed


def p_dnf(summary: Summary):
    """Hazard-only DNF probability, 1 - exp(-H), computed with expm1 so small risks survive."""
    return -jnp.expm1(-summary.H)


def expected_score(summary: Summary, course: Course, obj: ObjParams):
    p = p_dnf(summary)
    return (1.0 - p) * summary.T + p * t_dnf(course, obj)


class Score(NamedTuple):
    smooth_loss: jnp.ndarray
    hard_score: jnp.ndarray
    n_missed: jnp.ndarray
    miss_total: jnp.ndarray


def score(summary: Summary, course: Course, obj: ObjParams) -> Score:
    hard_miss, smooth_miss = gate_miss(summary, course, obj)
    e = expected_score(summary, course, obj)
    n_missed = jnp.sum(hard_miss > 0)
    miss_total = jnp.sum(hard_miss)
    hard_excess = summary.wall / obj.wall_weight
    dnf = (n_missed > 0) | (summary.hard_dnf > 0)
    return Score(
        smooth_loss=e + obj.lam_gate * jnp.sum(smooth_miss**2) + summary.wall,
        hard_score=jnp.where(dnf, obj.dq_base + miss_total + hard_excess, e),
        n_missed=n_missed,
        miss_total=miss_total,
    )


class Evaluation(NamedTuple):
    smooth_loss: jnp.ndarray   # (B,) mean over courses
    hard_score: jnp.ndarray    # (B,) mean over courses
    n_missed: jnp.ndarray      # (B, C)
    summary: Summary           # fields (B, C, ...)


def stack_courses(courses: list[Course]) -> Course:
    """Stack courses along a new leading axis (all must have equal array shapes;
    use courses.to_courses to pad CourseSpecs with different gate counts)."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *courses)


def make_evaluator(policy, p: SimParams, n_steps: int, obj: ObjParams | None = None):
    """Return `evaluate(params_batch, courses, obj=None) -> Evaluation` (jitted).

    params_batch has shape (B, n_params); courses is a stacked Course with leading axis C.
    Every candidate runs on every course; scores are averaged over courses. `obj` may be passed
    per call (e.g. to anneal lam_gate) without recompiling."""
    default_obj = obj or ObjParams()

    def one(params, course, o):
        summary, _ = rollout(policy.act, policy.decode(params), course, p, n_steps)
        return score(summary, course, o), summary

    over_courses = jax.vmap(one, in_axes=(None, 0, None))
    over_params = jax.vmap(over_courses, in_axes=(0, None, None))

    @jax.jit
    def _evaluate(params_batch, courses: Course, o: ObjParams) -> Evaluation:
        s, summary = over_params(params_batch, courses, o)
        return Evaluation(s.smooth_loss.mean(axis=1), s.hard_score.mean(axis=1), s.n_missed, summary)

    def evaluate(params_batch, courses: Course, obj: ObjParams | None = None) -> Evaluation:
        return _evaluate(params_batch, courses, obj or default_obj)

    return evaluate


def make_trajectory(policy, p: SimParams, n_steps: int):
    """Jitted single-candidate rollout returning (Summary, StepOutput) for telemetry."""
    return jax.jit(lambda params, course: rollout(policy.act, policy.decode(params), course, p, n_steps))


def make_value_and_grad(policy, p: SimParams, n_steps: int, obj: ObjParams | None = None):
    """Return `vg(params_batch, courses, obj=None) -> (smooth_loss (B,), grads (B, n_params))`
    (jitted): the gradient of each candidate's course-averaged smooth loss, for gradient methods."""
    default_obj = obj or ObjParams()

    def loss(params, courses, o):
        def one(course):
            summary, _ = rollout(policy.act, policy.decode(params), course, p, n_steps)
            return score(summary, course, o).smooth_loss
        return jnp.mean(jax.vmap(one)(courses))

    batched = jax.jit(jax.vmap(jax.value_and_grad(loss), in_axes=(0, None, None)))

    def vg(params_batch, courses: Course, obj: ObjParams | None = None):
        return batched(params_batch, courses, obj or default_obj)

    return vg
