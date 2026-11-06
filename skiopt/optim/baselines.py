"""Search-free and minimal baselines used to exercise the pipeline. They are not part of the
method comparison (PLAN.md §9)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from skiopt.budget import Budget
from skiopt.optim.common import Problem, Tracker


def fall_line(prob: Problem, method, budget: Budget, key, log, init=None):
    """Evaluate the policy's zero line (the straight fall line) once."""
    del method, key
    params = prob.policy.zero() if init is None else init
    ev = prob.evaluate(params[None], prob.courses)
    budget.charge_rollouts(prob.n_courses)
    tracker = Tracker(log, budget)
    tracker.update(params[None], ev)
    return tracker.result()


def random_search(prob: Problem, method, budget: Budget, key, log, init=None):
    """(1 + lambda) local random search on the hard score: sample `population` Gaussian
    perturbations of the incumbent, keep the best. Starts from the fall line."""
    pop, sigma = int(method["population"]), float(method["sigma"])
    tracker = Tracker(log, budget)
    best = prob.policy.zero() if init is None else init
    ev = prob.evaluate(best[None], prob.courses)
    budget.charge_rollouts(prob.n_courses)
    tracker.update(best[None], ev)
    while budget.remaining >= pop * prob.n_courses:
        key, sub = jax.random.split(key)
        candidates = tracker.best_params + sigma * jax.random.normal(sub, (pop, prob.policy.n_params))
        ev = prob.evaluate(candidates, prob.courses)
        budget.charge_rollouts(pop * prob.n_courses)
        tracker.update(candidates, ev)
    return tracker.result()


def coach_line(prob: Problem, method, budget: Budget, key, log, init=None):
    """A hand-built racer's line (PLAN.md §6): the arc policy's default, every apex `margin0` m
    outside its turning pole on its gate line, joined by circular arcs. No search; one rollout per
    course. Only defined for the arc policy, where it is exactly the line every optimizer starts from."""
    del method, key, init
    if getattr(prob.policy, "name", None) != "arc":
        raise ValueError("coach_line needs `policy: arc` (a spline cannot draw even arcs at 20 m spacing)")
    params = prob.policy.zero()
    ev = prob.evaluate(params[None], prob.courses)
    budget.charge_rollouts(prob.n_courses)
    tracker = Tracker(log, budget)
    tracker.update(params[None], ev)
    return tracker.result()
