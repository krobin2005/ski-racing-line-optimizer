"""Dynamics protocol and the 2D point-mass skier (PLAN.md §4).

Integration is over a fixed spatial step dy down the fall line. Per step:

    kappa  = clip(kappa_cmd, +-1/r_floor)
    ds     = dy / cos(psi)
    x'     = x + tan(psi) dy
    psi'   = psi + kappa ds              (then smoothly limited to |psi| < psi_max)
    a_net  = kappa v^2 + g sin(theta) sin(psi)
    v dv/ds = g sin(theta) cos(psi) - mu g cos(theta) - c_d v^2 - turn losses
    dt     = 2 ds / (v + v')             (exact for constant deceleration along the step)

Speed is advanced through v'^2 = v^2 + 2 F ds, so straight runs at constant pitch match the
analytic v^2 = v0^2 + 2 g sin(theta) s exactly.
"""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp

from skiopt.types import SimParams, State, StepOutput, Telemetry


class Dynamics(Protocol):
    control_dim: int

    def step(self, state: State, control, theta, p: SimParams, y) -> tuple[State, StepOutput]: ...


def softplus_b(z, beta):
    """Smooth max(z, 0) with sharpness beta: softplus(beta z) / beta."""
    return jax.nn.softplus(beta * z) / beta


def lateral_load(kappa, v, theta, psi, g):
    """Signed lateral force per unit mass the snow must supply.

    With heading h = (sin psi, cos psi) and turn normal n = (cos psi, -sin psi), the turn needs
    kappa v^2 along n and gravity already supplies -g sin(theta) sin(psi) along n. So the load is
    lower before the fall line (|psi| shrinking) and higher after it."""
    return kappa * v**2 + g * jnp.sin(theta) * jnp.sin(psi)


def total_g(a_net, theta, g):
    """Total load on the skier in g: lateral load plus the slope-normal component of gravity."""
    return jnp.sqrt(a_net**2 + (g * jnp.cos(theta)) ** 2) / g


def clip_kappa(kappa_cmd, p: SimParams):
    k_max = 1.0 / p.r_floor
    return jnp.clip(kappa_cmd, -k_max, k_max)


def turn_deceleration(kappa, v, a_net, p: SimParams):
    """Speed loss from turning (m/s^2): carving loss on every turn, a rising cost for turns
    tighter than r_typical_lo, and a heavy skid loss below r_skid (PLAN.md §4, turning bands)."""
    abs_k = jnp.abs(kappa)
    tight = softplus_b(abs_k - 1.0 / p.r_typical_lo, p.beta_kappa) * v**2
    skid = softplus_b(abs_k - 1.0 / p.r_skid, p.beta_kappa) * v**2
    return (
        p.w_turn * p.k_turn * jnp.abs(a_net)
        + p.w_tight * p.k_tight * tight
        + p.w_skid * p.k_skid * skid
    )


class PointMass2D:
    """2D point mass in the slope plane; control is the commanded curvature (1/m)."""

    control_dim = 1

    def step(self, state: State, kappa_cmd, theta, p: SimParams, y=0.0) -> tuple[State, StepOutput]:
        x, psi, v = state
        kappa = clip_kappa(kappa_cmd, p)

        cos_psi = jnp.cos(psi)
        ds = p.dy / cos_psi
        x_new = x + jnp.tan(psi) * p.dy
        psi_new = p.psi_max * jnp.tanh((psi + kappa * ds) / p.psi_max)

        a_net = lateral_load(kappa, v, theta, psi, p.g)
        force = (
            p.w_gravity * p.g * jnp.sin(theta) * cos_psi
            - p.w_friction * p.mu * p.g * jnp.cos(theta)
            - p.w_drag * p.c_d * v**2
            - turn_deceleration(kappa, v, a_net, p)
        )
        v_new = jnp.sqrt(jnp.maximum(v**2 + 2.0 * force * ds, p.v_floor**2))
        dt = 2.0 * ds / (v + v_new)

        telemetry = Telemetry(
            x=x, y=jnp.asarray(y, dtype=x.dtype), psi=psi, v=v, kappa=kappa,
            a_net=a_net, G=total_g(a_net, theta, p.g), theta=theta,
        )
        out = StepOutput(dt=dt, ds=ds, active=jnp.ones_like(dt), telemetry=telemetry)
        return State(x_new, psi_new, v_new), out
