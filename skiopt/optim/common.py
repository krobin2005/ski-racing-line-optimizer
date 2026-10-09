"""Shared pieces for every optimizer: the problem bundle and per-generation logging.

Every method has the signature

    method(prob: Problem, method_cfg: dict, budget: Budget, key, log, init=None)
        -> (best_params, best_hard_score, best_so_far_history)

and draws all rollouts from `budget` (PLAN.md §9), so comparisons are at equal cost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np

from skiopt.objectives import Evaluation, ObjParams
from skiopt.types import Course


@dataclass
class Problem:
    evaluate: Callable         # (params_batch, courses, obj=None) -> Evaluation
    value_and_grad: Callable   # (params_batch, courses, obj=None) -> (smooth_loss, grads)
    courses: Course            # stacked, leading axis C
    policy: object
    obj: ObjParams

    @property
    def n_courses(self) -> int:
        return int(self.courses.length.shape[0])

    def init_population(self, key, size: int, sigma: float):
        """Equal starts for every method (PLAN.md §9): the fall line plus Gaussian noise."""
        import jax

        return self.policy.zero() + sigma * jax.random.normal(key, (size, self.policy.n_params))


# Method-specific progress columns. Every row carries all of them (blank when unused) so the
# hybrid's CMA-ES and Adam phases can share one progress.csv.
EXTRA_COLUMNS = ("sigma", "step", "lam_gate", "smooth", "grad_norm")


class Tracker:
    """Best-so-far by hard score, plus the per-generation progress row."""

    def __init__(self, log, budget):
        self.log, self.budget = log, budget
        self.best_params, self.best_score, self.history = None, float("inf"), []
        self.generation = 0

    def update(self, params_batch, ev: Evaluation, **extra) -> None:
        unknown = set(extra) - set(EXTRA_COLUMNS)
        if unknown:
            raise ValueError(f"unknown progress columns {sorted(unknown)}; add them to EXTRA_COLUMNS")
        hard = np.asarray(ev.hard_score)
        i = int(np.argmin(hard))
        if hard[i] < self.best_score:
            self.best_score, self.best_params = float(hard[i]), jnp.asarray(params_batch[i])
        self.history.append(self.best_score)
        s = ev.summary
        p_dnf = -np.expm1(-np.asarray(s.H[i]))
        self.log.progress(
            generation=self.generation, evals=self.budget.used, best=self.best_score,
            gen_best=float(hard[i]), gen_mean=float(np.mean(hard)),
            T=float(np.mean(s.T[i])), D=float(np.mean(s.D[i])), v_mean=float(np.mean(s.v_mean[i])),
            P_DNF=float(np.mean(p_dnf)), missed=int(np.sum(ev.n_missed[i])), max_G=float(np.max(s.max_G[i])),
            wall_clock=round(self.budget.wall_clock, 3),
            **{k: extra.get(k, "") for k in EXTRA_COLUMNS},
        )
        self.generation += 1

    def result(self):
        return self.best_params, self.best_score, self.history
