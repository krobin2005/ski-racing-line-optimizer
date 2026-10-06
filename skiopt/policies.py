"""Policies: params -> commanded curvature at each step, all through `act(params, obs, k)`.

Open-loop kappa is the simplest possible policy (one curvature value per step) and drives the
day 1-2 pipeline and tests. The spline, MLP and coach-line policies follow (PLAN.md §6)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


class OpenLoopKappa:
    """params[k] is the commanded curvature at step k. Zeros = the straight fall line."""

    name = "open_loop_kappa"

    def __init__(self, n_steps: int):
        self.n_steps = n_steps

    @property
    def n_params(self) -> int:
        return self.n_steps

    def init(self, key, sigma: float = 0.0):
        return sigma * jax.random.normal(key, (self.n_steps,))

    @staticmethod
    def act(params, obs, k):
        del obs
        return params[k]


POLICIES = {OpenLoopKappa.name: OpenLoopKappa}


def make_policy(name: str, n_steps: int):
    try:
        return POLICIES[name](n_steps)
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; known: {sorted(POLICIES)}") from None


def zeros(policy) -> jnp.ndarray:
    return jnp.zeros((policy.n_params,))
