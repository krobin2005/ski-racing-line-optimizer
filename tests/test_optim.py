"""Optimizer checks (PLAN.md §13, track B): spline conversion, and every method improves on the
fall line of the easy course within a small budget."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skiopt import courses as C
from skiopt.budget import Budget
from skiopt.config import load_config
from skiopt.objectives import ObjParams, make_evaluator, make_trajectory, make_value_and_grad, stack_courses
from skiopt.optim.common import Problem
from skiopt.optim.registry import METHODS
from skiopt.policies import SplineLine
from skiopt.rollout import n_steps_for
from skiopt.types import SimParams


class NullLog:
    def progress(self, **row):
        self.last = row


@pytest.fixture(scope="module")
def problem():
    cfg = load_config()
    p, obj = SimParams.from_config(cfg), ObjParams.from_config(cfg)
    spec = C.easy()
    n = n_steps_for(spec.length, p.dy)
    policy = SplineLine(n, spec.length, spec.x_start, p.dy, 10.0)
    return Problem(evaluate=make_evaluator(policy, p, n, obj), value_and_grad=make_value_and_grad(policy, p, n, obj),
                   courses=stack_courses([C.to_course(spec)]), policy=policy, obj=obj), p, n


def test_spline_line_is_followed_exactly_until_the_clip(problem):
    prob, p, n = problem
    policy = prob.policy
    params = jnp.asarray(5.0 * np.sin(policy.knots[1:] / 40.0))  # smooth: turns far rounder than 12 m
    _, traj = make_trajectory(policy, p, n)(params, C.to_course(C.easy()))
    err = np.abs(np.asarray(traj.telemetry.x) - np.asarray(policy.line(params))[:n])
    assert err.max() < 0.05  # only the forced straight first metre differs


def test_gradient_flows_through_the_spline(problem):
    prob, _, _ = problem
    _, grads = prob.value_and_grad(prob.policy.zero()[None], prob.courses)
    assert np.all(np.isfinite(np.asarray(grads))) and float(jnp.abs(grads).max()) > 0


SMALL = {
    "adam": {"lr": 0.1, "lr_final": 0.01, "eval_every": 10},
    "cmaes": {"population": 16, "sigma": 0.4},
    "cmaes_smooth": {"population": 16, "sigma": 0.4},
    "ga": {"population": 32, "sigma": 1.0},
    "hybrid": {"cma_fraction": 0.3, "cmaes": {"population": 16, "sigma": 0.4}, "adam": {"eval_every": 10}},
    "random_search": {"population": 16, "sigma": 0.2},
}


@pytest.mark.parametrize("name", sorted(SMALL))
def test_method_improves_on_the_fall_line_and_respects_the_budget(problem, name):
    prob, _, _ = problem
    fall = float(prob.evaluate(prob.policy.zero()[None], prob.courses).hard_score[0])
    budget = Budget(1500, grad_step_cost=3.0)
    best, score, history = METHODS[name](prob, SMALL[name], budget, jax.random.key(0), NullLog())
    assert budget.used <= budget.total
    assert score < fall
    assert np.all(np.diff(history) <= 0)  # best-so-far never gets worse
    assert float(prob.evaluate(best[None], prob.courses).hard_score[0]) == pytest.approx(score, rel=1e-5)


def test_same_seed_gives_the_same_champion(problem):
    prob, _, _ = problem
    runs = [METHODS["cmaes"](prob, SMALL["cmaes"], Budget(500), jax.random.key(3), NullLog()) for _ in range(2)]
    np.testing.assert_array_equal(np.asarray(runs[0][0]), np.asarray(runs[1][0]))
