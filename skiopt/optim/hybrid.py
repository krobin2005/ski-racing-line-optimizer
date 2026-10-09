"""Hybrid (PLAN.md §9): CMA-ES on the hard score for a fraction f of the budget, then Adam from the
best CMA-ES candidate for the rest. Both phases log to the same progress file."""

from __future__ import annotations

import jax

from skiopt.budget import Budget
from skiopt.optim.common import Problem
from skiopt.optim.evo import cmaes
from skiopt.optim.gradient import adam


def hybrid(prob: Problem, method: dict, budget: Budget, key, log, init=None):
    f = float(method.get("cma_fraction", 0.3))
    k_cma, k_adam = jax.random.split(key)
    cma_cfg = {**method.get("cmaes", {}), "_stop_at": budget.used + f * (budget.total - budget.used)}
    best, best_score, history = cmaes(prob, cma_cfg, budget, k_cma, log, init)
    adam_cfg = {**method.get("adam", {}), "restarts": 1}
    best2, score2, history2 = adam(prob, adam_cfg, budget, k_adam, log, init=best)
    if score2 < best_score:
        best, best_score = best2, score2
    return best, best_score, history + [min(best_score, h) for h in history2]
