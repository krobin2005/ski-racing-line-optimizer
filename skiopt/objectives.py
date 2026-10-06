"""Objectives and the batched evaluator every optimizer calls (PLAN.md §5, §9).

A gate is passed when x at the gate line lies between the turning pole plus the body clearance
and the outside pole. Missing any gate is a DNF:

    smooth_loss = E[score] + lam_gate * sum softplus(miss)^2     (gradient methods)
    hard_score  = DQ_BASE + sum miss    if any gate is missed   (evolution, all reporting)
                = E[score]              otherwise

Until the hazard model lands (days 5-6), P_finish = 1, so E[score] = T."""

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

    @classmethod
    def from_config(cls, cfg: dict) -> "ObjParams":
        o = cfg["objective"]
        return cls(dq_base=float(o["dq_base"]), lam_gate=float(o["lam_gate"]),
                   beta_miss=float(o["beta_miss"]),
                   clearance=float(cfg["fis"]["model"]["body_clearance_m"]))


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
    hard = (jax.nn.relu(lo - x) + jax.nn.relu(x - hi)) * course.gates.valid
    smooth = (softplus_b(lo - x, obj.beta_miss) + softplus_b(x - hi, obj.beta_miss)) * course.gates.valid
    return hard, smooth


def expected_score(summary: Summary):
    return summary.T


class Score(NamedTuple):
    smooth_loss: jnp.ndarray
    hard_score: jnp.ndarray
    n_missed: jnp.ndarray
    miss_total: jnp.ndarray


def score(summary: Summary, course: Course, obj: ObjParams) -> Score:
    hard_miss, smooth_miss = gate_miss(summary, course, obj)
    e = expected_score(summary)
    n_missed = jnp.sum(hard_miss > 0)
    miss_total = jnp.sum(hard_miss)
    return Score(
        smooth_loss=e + obj.lam_gate * jnp.sum(smooth_miss**2),
        hard_score=jnp.where(n_missed > 0, obj.dq_base + miss_total, e),
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
        summary, _ = rollout(policy.act, params, course, p, n_steps)
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
    return jax.jit(lambda params, course: rollout(policy.act, params, course, p, n_steps))
