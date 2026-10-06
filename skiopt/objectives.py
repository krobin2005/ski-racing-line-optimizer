"""Objectives and the batched evaluator every optimizer calls (PLAN.md §5, §9).

Until the DNF model and gates land (days 3-6), both objectives are the course time T:
P_finish = 1 so E[score] = T, and there are no gates to miss. The evaluator's interface is the
final one, so optimizers written now do not change when the objectives fill in."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from skiopt.rollout import rollout
from skiopt.types import Course, SimParams, Summary


def expected_score(summary: Summary):
    return summary.T


def smooth_loss(summary: Summary):
    return expected_score(summary)


def hard_score(summary: Summary):
    return expected_score(summary)


class Evaluation(NamedTuple):
    smooth_loss: jnp.ndarray   # (B,) mean over courses
    hard_score: jnp.ndarray    # (B,) mean over courses
    summary: Summary           # fields (B, C)


def stack_courses(courses: list[Course]) -> Course:
    """Stack courses along a new leading axis (all must have equal array shapes)."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *courses)


def make_evaluator(policy, p: SimParams, n_steps: int):
    """Return jitted `evaluate(params_batch, courses) -> Evaluation`.

    params_batch has shape (B, n_params); courses is a stacked Course with leading axis C.
    Every candidate runs on every course; scores are averaged over courses."""

    def one(params, course):
        summary, _ = rollout(policy.act, params, course, p, n_steps)
        return smooth_loss(summary), hard_score(summary), summary

    over_courses = jax.vmap(one, in_axes=(None, 0))
    over_params = jax.vmap(over_courses, in_axes=(0, None))

    @jax.jit
    def evaluate(params_batch, courses: Course) -> Evaluation:
        sl, hs, summary = over_params(params_batch, courses)
        return Evaluation(sl.mean(axis=1), hs.mean(axis=1), summary)

    return evaluate


def make_trajectory(policy, p: SimParams, n_steps: int):
    """Jitted single-candidate rollout returning (Summary, StepOutput) for telemetry."""
    return jax.jit(lambda params, course: rollout(policy.act, params, course, p, n_steps))
