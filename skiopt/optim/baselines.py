"""Search-free and minimal baselines used to exercise the pipeline. They are not part of the
method comparison (PLAN.md §9); Adam, CMA-ES and the GA arrive on days 3-5 with the same shape."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from skiopt.budget import Budget
from skiopt.policies import zeros


def fall_line(evaluate, courses, policy, method, budget: Budget, key, log):
    """Evaluate the all-zeros policy (the straight fall line for open-loop curvature)."""
    del method, key
    params = zeros(policy)
    result = evaluate(params[None], courses)
    budget.charge_rollouts(courses.length.shape[0])
    score = float(result.hard_score[0])
    log.progress(generation=0, evals=budget.used, best=score, mean=score,
                 T=float(result.summary.T[0].mean()), D=float(result.summary.D[0].mean()),
                 v_mean=float(result.summary.v_mean[0].mean()), wall_clock=budget.wall_clock)
    return params, score, [score]


def random_search(evaluate, courses, policy, method, budget: Budget, key, log):
    """(1 + lambda) local random search on the hard score: sample `population` Gaussian
    perturbations of the incumbent, keep the best. Starts from the fall line."""
    pop, sigma = int(method["population"]), float(method["sigma"])
    n_courses = courses.length.shape[0]
    best = zeros(policy)
    best_score = float(evaluate(best[None], courses).hard_score[0])
    budget.charge_rollouts(n_courses)
    history, generation = [best_score], 0
    while budget.remaining >= pop * n_courses:
        key, sub = jax.random.split(key)
        candidates = best + sigma * jax.random.normal(sub, (pop, policy.n_params))
        result = evaluate(candidates, courses)
        budget.charge_rollouts(pop * n_courses)
        i = int(jnp.argmin(result.hard_score))
        if float(result.hard_score[i]) < best_score:
            best, best_score = candidates[i], float(result.hard_score[i])
        history.append(best_score)
        generation += 1
        log.progress(generation=generation, evals=budget.used, best=best_score,
                     mean=float(result.hard_score.mean()), T=float(result.summary.T[i].mean()),
                     D=float(result.summary.D[i].mean()), v_mean=float(result.summary.v_mean[i].mean()),
                     wall_clock=budget.wall_clock)
    return best, best_score, history


METHODS = {"fall_line": fall_line, "random_search": random_search}
