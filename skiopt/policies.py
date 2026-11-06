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


def curvature_from_line(x, dy, kappa_c=None):
    """Exact curvature commands for a line sampled on the dy grid (length n_steps + 1), using the
    step function's own scheme. In 2D: psi_k = atan(dx/dy), ds_k = dy / cos(psi_k),
    kappa_k = (psi_{k+1} - psi_k) / ds_k. With a curving course frame (3D, kappa_c per step), the
    scale factor h = 1 - x kappa_c enters and the frame's own turning is added back:
    psi_k = atan(dx / (h dy)), d_sigma_k = h dy / cos(psi_k), kappa_k = (dpsi_k + kappa_c dy) / d_sigma_k.
    The first heading is forced to 0 (the rollout starts there)."""
    if kappa_c is None:
        h, kc = 1.0, 0.0
    else:
        kc = kappa_c[: x.shape[0] - 1]
        h = jnp.maximum(1.0 - x[:-1] * kc, 0.2)
    psi = jnp.arctan(jnp.diff(x) / (h * dy))
    psi = psi.at[0].set(0.0)
    d_sigma = h * dy / jnp.cos(psi)
    turn = jnp.diff(psi) + (kc[:-1] * dy if kappa_c is not None else 0.0)
    return jnp.append(turn / d_sigma[:-1], 0.0)


class ArcLine:
    """Apex-and-arc line: one apex per turning gate, joined by circular arcs (PLAN.md §6).

    Between successive apexes (heading along the fall line at each), the line is two equal arcs of
    radius R = (L^2 + w^2) / (4 w), with the inflection halfway. That is the geometry the course
    generator uses to size offsets, so turn radius is constant within each half-turn instead of
    peaking, and the line can only take shapes a carving skier could. After the last apex the line
    runs straight to the finish. Delay gates are not apexes; the line passes them on its traverse.

    params, per turning gate i (2 per gate, or 1 if apex_shift is off):
      u_i  apex distance outside the turning pole: clearance + softplus(u_i)  (always outside it)
      v_i  apex position along the fall line: gate_y + shift_lo + (shift_hi - shift_lo) * sigmoid(v_i)
    zero() puts every apex `margin0` m outside its pole at its gate line: the coach line.
    """

    name = "arc"

    def __init__(self, n_steps: int, spec=None, x_start: float = 0.0, dy: float = 1.0,
                 clearance: float = 0.4, margin0: float = 1.0, apex_shift: bool = True,
                 shift_range=(-6.0, 2.0), kappa_c=None, **_):
        if spec is None:
            raise ValueError("ArcLine needs the course spec (its apexes are the turning gates)")
        import numpy as np

        from skiopt.courses import turning_gates

        t = turning_gates(spec.gate_side)
        self.n_steps, self.dy, self.x_start = n_steps, dy, float(x_start)
        self.apex_y = jnp.asarray(np.asarray(spec.gate_y, dtype=float)[t])
        self.pole_x = jnp.asarray(np.asarray(spec.gate_x, dtype=float)[t])
        self.side = jnp.asarray(np.asarray(spec.gate_side, dtype=float)[t])
        self.n_apex = len(t)
        self.clearance, self.margin0 = float(clearance), float(margin0)
        self.apex_shift = bool(apex_shift)
        self.shift_lo, self.shift_hi = map(float, shift_range)
        self.grid = jnp.arange(n_steps + 1) * dy
        # 3D: curvature of the course frame per step, for the curvilinear conversion.
        self.kappa_c = None if kappa_c is None else jnp.asarray(np.resize(np.asarray(kappa_c, float), n_steps + 1))

    @property
    def n_params(self) -> int:
        return self.n_apex * (2 if self.apex_shift else 1)

    def zero(self):
        """The coach line: every apex margin0 outside its pole, on its gate line."""
        u0 = jnp.log(jnp.expm1(self.margin0 - self.clearance))  # inverse softplus
        v0 = jnp.log(-self.shift_lo / self.shift_hi) if self.shift_hi > 0 else 0.0  # sigmoid -> shift 0
        parts = [jnp.full((self.n_apex,), u0)]
        if self.apex_shift:
            parts.append(jnp.full((self.n_apex,), v0))
        return jnp.concatenate(parts)

    def init(self, key, sigma: float = 0.0):
        return self.zero() + sigma * jax.random.normal(key, (self.n_params,))

    def apexes(self, params):
        u = params[: self.n_apex]
        ax = self.pole_x + self.side * (self.clearance + jax.nn.softplus(u))
        ay = self.apex_y
        if self.apex_shift:
            v = params[self.n_apex:]
            ay = ay + self.shift_lo + (self.shift_hi - self.shift_lo) * jax.nn.sigmoid(v)
        return ay, ax

    def line(self, params):
        ay, ax = self.apexes(params)
        ys = jnp.concatenate([jnp.zeros(1), ay])
        xs = jnp.concatenate([jnp.full(1, self.x_start), ax])
        t = self.grid
        i = jnp.clip(jnp.searchsorted(ys, t, side="right") - 1, 0, self.n_apex - 1)
        ya, yb, xa, xb = ys[i], ys[i + 1], xs[i], xs[i + 1]
        L = yb - ya
        d = xb - xa
        w = jnp.sqrt(d**2 + 1e-8)
        R = (L**2 + w**2) / (4 * w)
        # R - sqrt(R^2 - u^2) written as u^2 / (R + sqrt(R^2 - u^2)): stable when w -> 0 (R -> inf).
        # u is clamped to [0, L/2] (where R >= L/2 keeps the root real) so the branch jnp.where
        # discards never produces an infinite gradient that would turn into NaN.
        def bulge(u):
            u = jnp.clip(u, 0.0, L / 2)
            return u**2 / (R + jnp.sqrt(jnp.maximum(R**2 - u**2, 1e-9)))

        first_half = t < ya + L / 2
        x_arc = jnp.where(first_half, xa + (d / w) * bulge(t - ya), xb - (d / w) * bulge(yb - t))
        return jnp.where(t >= ys[-1], xs[-1], x_arc)

    def decode(self, params):
        return curvature_from_line(self.line(params), self.dy, self.kappa_c)

    @staticmethod
    def act(kappa, obs, k):
        del obs
        return kappa[k]


POLICIES = {OpenLoopKappa.name: OpenLoopKappa, SplineLine.name: SplineLine, ArcLine.name: ArcLine}


def make_policy(name: str, n_steps: int, **kwargs):
    try:
        cls = POLICIES[name]
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; known: {sorted(POLICIES)}") from None
    return cls(n_steps, **kwargs)


def zeros(policy) -> jnp.ndarray:
    return policy.zero()
