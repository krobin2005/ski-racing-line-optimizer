"""Evaluation budget in rollout-equivalents (PLAN.md §9): one forward rollout = 1, one gradient
step = `grad_step_cost`. Every method draws from the same counter, so comparisons are fair."""

from __future__ import annotations

import time


class Budget:
    def __init__(self, total: float, grad_step_cost: float = 3.0):
        self.total = float(total)
        self.grad_step_cost = float(grad_step_cost)
        self.used = 0.0
        self._t0 = time.perf_counter()

    def charge_rollouts(self, n: int) -> None:
        self.used += n

    def charge_grad_steps(self, n: int, batch: int = 1) -> None:
        self.used += n * batch * self.grad_step_cost

    @property
    def remaining(self) -> float:
        return max(self.total - self.used, 0.0)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.total

    @property
    def wall_clock(self) -> float:
        return time.perf_counter() - self._t0
