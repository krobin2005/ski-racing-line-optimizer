"""Numerics checks (PLAN.md §13, track B): batching, gradients, determinism."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import make_params
from skiopt import courses, policies
from skiopt.objectives import make_evaluator, stack_courses
from skiopt.rollout import rollout


def test_vmapped_batch_equals_python_loop(base_params):
    n = 150
    policy = policies.OpenLoopKappa(n)
    course = courses.constant_pitch(n, 22)
    evaluate = make_evaluator(policy, base_params, n)
    batch = 0.01 * jax.random.normal(jax.random.key(1), (5, n))

    batched = np.asarray(evaluate(batch, stack_courses([course])).hard_score)
    looped = [float(evaluate(batch[i : i + 1], stack_courses([course])).hard_score[0]) for i in range(5)]
    np.testing.assert_allclose(batched, looped, rtol=1e-6)


def test_multiple_courses_are_averaged(base_params):
    n = 120
    policy = policies.OpenLoopKappa(n)
    c1, c2 = courses.constant_pitch(n, 15), courses.constant_pitch(n, 25)
    evaluate = make_evaluator(policy, base_params, n)
    params = jnp.zeros((1, n))
    both = float(evaluate(params, stack_courses([c1, c2])).hard_score[0])
    one = float(evaluate(params, stack_courses([c1])).hard_score[0])
    two = float(evaluate(params, stack_courses([c2])).hard_score[0])
    assert both == pytest.approx((one + two) / 2, rel=1e-6)


def test_gradient_matches_finite_differences():
    """d T / d kappa from jax.grad agrees with central differences in float64."""
    with jax.enable_x64(True):
        p = make_params()
        n = 120
        course = courses.constant_pitch(n, 20)
        policy = policies.OpenLoopKappa(n)
        key = jax.random.key(3)
        kappa = 0.02 + 0.01 * jax.random.normal(key, (n,), dtype=jnp.float64)
        direction = jax.random.normal(jax.random.key(4), (n,), dtype=jnp.float64)

        T = jax.jit(lambda k: rollout(policy.act, k, course, p, n)[0].T)
        analytic = float(jnp.dot(jax.grad(T)(kappa), direction))
        eps = 1e-6
        numeric = float((T(kappa + eps * direction) - T(kappa - eps * direction)) / (2 * eps))
        assert analytic == pytest.approx(numeric, rel=1e-5)


def test_same_inputs_give_identical_results(base_params):
    n = 100
    policy = policies.OpenLoopKappa(n)
    evaluate = make_evaluator(policy, base_params, n)
    batch = 0.01 * jax.random.normal(jax.random.key(7), (8, n))
    c = stack_courses([courses.constant_pitch(n, 20)])
    a = np.asarray(evaluate(batch, c).hard_score)
    b = np.asarray(evaluate(batch, c).hard_score)
    assert np.array_equal(a, b)


def test_padded_steps_after_finish_are_inactive(base_params):
    """A 100 m course rolled out for 150 steps gives the same result as for 100 steps."""
    policy = policies.OpenLoopKappa(150)
    course = courses.constant_pitch(100, 20)
    padded, traj = rollout(policy.act, jnp.zeros(150), course, base_params, 150)
    exact, _ = rollout(policies.OpenLoopKappa(100).act, jnp.zeros(100), course, base_params, 100)
    assert float(padded.T) == pytest.approx(float(exact.T), rel=1e-6)
    assert float(jnp.sum(traj.active)) == 100
