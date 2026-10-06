"""Terrain protocol and the 2D pitch-profile terrain (PLAN.md §12).

y is distance along the slope surface down the fall line, so a step dy drops sin(theta)*dy
vertically. A 3D height field later implements the same three functions using x as well."""

from __future__ import annotations

from typing import Protocol

import jax.numpy as jnp

from skiopt.types import ProfileTerrain


class Terrain(Protocol):
    def pitch(self, x, y): ...
    def height(self, x, y): ...
    def normal(self, x, y): ...


def pitch(terrain: ProfileTerrain, x, y):
    """Fall-line pitch (rad) at (x, y). The 2D profile ignores x."""
    del x
    return jnp.interp(y, terrain.y_knots, terrain.theta_knots)


def vertical_drop(terrain: ProfileTerrain, y):
    """Vertical drop (m) from y = 0 down to y: the integral of sin(theta) dy (trapezoid rule)."""
    ys = jnp.linspace(0.0, y, 2001)
    return jnp.trapezoid(jnp.sin(pitch(terrain, 0.0, ys)), ys)


def height(terrain: ProfileTerrain, x, y):
    """Height relative to the start (m, negative downhill)."""
    del x
    return -vertical_drop(terrain, y)


def normal(terrain: ProfileTerrain, x, y):
    """Unit surface normal in (across, down-slope horizontal, up) world axes."""
    th = pitch(terrain, x, y)
    return jnp.stack([jnp.zeros_like(th), jnp.sin(th), jnp.cos(th)])


def constant(theta_rad: float, length: float) -> ProfileTerrain:
    return ProfileTerrain(
        y_knots=jnp.array([0.0, float(length)]),
        theta_knots=jnp.array([float(theta_rad), float(theta_rad)]),
    )
