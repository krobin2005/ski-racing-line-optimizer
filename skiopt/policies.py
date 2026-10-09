"""Policies: params -> commanded curvature at each step (PLAN.md §6).

Every policy has `decode(params) -> rollout params` and `act(rollout_params, obs, k) -> kappa`.
`decode` runs once per candidate before the rollout, so open-loop policies can turn their
parameters into a whole curvature sequence up front instead of at every step.

- OpenLoopKappa: one curvature value per metre (tests and the smoke pipeline).
- SplineLine: a cubic spline x(y) through control points, converted to curvature exactly with the
  step function's own finite differences, so the skier follows the spline until the 12 m clip.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline


class OpenLoopKappa:
    """params[k] is the commanded curvature at step k. Zeros = the straight fall line."""

    name = "open_loop_kappa"

    def __init__(self, n_steps: int, **_):
        self.n_steps = n_steps

    @property
    def n_params(self) -> int:
        return self.n_steps

    def init(self, key, sigma: float = 0.0):
        return sigma * jax.random.normal(key, (self.n_steps,))

    def zero(self):
        return jnp.zeros((self.n_params,))

    @staticmethod
    def decode(params):
        return params

    @staticmethod
    def act(params, obs, k):
        del obs
        return params[k]


class SplineLine:
    """x(y) is a cubic spline through (0, x_start) and control points every `knot_spacing` metres
    to the finish. The start slope is clamped to 0 (the skier leaves the start straight down the
    fall line) and the end is natural. params = control-point x values (m).

    The spline is linear in its control values, so it is precomputed as a basis matrix B with
    x_grid = B @ [x_start, params]. Curvature then follows from the same scheme as the dynamics:
    psi_k = atan((x_{k+1} - x_k) / dy), ds_k = dy / cos(psi_k), kappa_k = (psi_{k+1} - psi_k) / ds_k.
    """

    name = "spline"

    def __init__(self, n_steps: int, length: float, x_start: float = 0.0, dy: float = 1.0,
                 knot_spacing: float = 10.0, **_):
        self.n_steps, self.dy, self.x_start = n_steps, dy, float(x_start)
        n_ctrl = max(int(round(length / knot_spacing)), 2)
        self.knots = np.linspace(0.0, float(length), n_ctrl + 1)
        grid = np.arange(n_steps + 1) * dy
        basis = np.empty((n_steps + 1, n_ctrl + 1))
        for j in range(n_ctrl + 1):
            e = np.zeros(n_ctrl + 1)
            e[j] = 1.0
            basis[:, j] = CubicSpline(self.knots, e, bc_type=((1, 0.0), (2, 0.0)))(grid)
        self.basis = jnp.asarray(basis)

    @property
    def n_params(self) -> int:
        return len(self.knots) - 1

    def zero(self):
        """The straight fall line from the start."""
        return jnp.full((self.n_params,), self.x_start)

    def init(self, key, sigma: float = 0.0):
        return self.zero() + sigma * jax.random.normal(key, (self.n_params,))

    def line(self, params):
        """x on the dy grid, length n_steps + 1."""
        return self.basis[:, 0] * self.x_start + self.basis[:, 1:] @ params

    def decode(self, params):
        x = self.line(params)
        psi = jnp.arctan(jnp.diff(x) / self.dy)
        psi = psi.at[0].set(0.0)  # the rollout starts with heading 0
        ds = self.dy / jnp.cos(psi)
        return jnp.append(jnp.diff(psi) / ds[:-1], 0.0)

    @staticmethod
    def act(kappa, obs, k):
        del obs
        return kappa[k]


POLICIES = {OpenLoopKappa.name: OpenLoopKappa, SplineLine.name: SplineLine}


def make_policy(name: str, n_steps: int, **kwargs):
    try:
        cls = POLICIES[name]
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; known: {sorted(POLICIES)}") from None
    return cls(n_steps, **kwargs)


def zeros(policy) -> jnp.ndarray:
    return policy.zero()
