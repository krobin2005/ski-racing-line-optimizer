"""Core data types. All are NamedTuples of arrays/floats, so they are JAX pytrees and can be
passed through jit, vmap and grad. Course fields have fixed shapes so courses can be stacked."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp


class SimParams(NamedTuple):
    """Every constant the step function needs, as floats (switches are 1.0 / 0.0)."""

    dy: float
    v0: float
    psi_max: float
    psi_knee: float     # heading limit is exact below this (rad), soft between it and psi_max
    v_floor: float
    g: float
    mu: float
    c_d: float          # 0.5 * rho * CdA / m  (1/m)
    k_turn: float
    k_tight: float
    k_skid: float
    beta_kappa: float
    r_typical_lo: float
    r_skid: float
    r_floor: float
    w_gravity: float
    w_friction: float
    w_drag: float
    w_turn: float
    w_tight: float
    w_skid: float

    @classmethod
    def from_config(cls, cfg: dict) -> "SimParams":
        sim, phys, skier = cfg["sim"], cfg["physics"], cfg["skier"]
        terms = phys["terms"]
        return cls(
            dy=float(sim["dy"]),
            v0=float(sim["v0"]),
            psi_max=math.radians(float(sim["psi_max_deg"])),
            psi_knee=math.radians(float(sim["psi_knee_deg"])),
            v_floor=float(sim["v_floor"]),
            g=float(phys["g"]),
            mu=float(phys["mu"]),
            c_d=0.5 * float(phys["rho"]) * float(phys["cda"]) / float(phys["mass"]),
            k_turn=float(phys["k_turn"]),
            k_tight=float(phys["k_tight"]),
            k_skid=float(phys["k_skid"]),
            beta_kappa=float(phys["beta_kappa"]),
            r_typical_lo=float(skier["r_typical_lo"]),
            r_skid=float(skier["r_skid"]),
            r_floor=float(skier["r_floor"]),
            w_gravity=float(terms["gravity"]),
            w_friction=float(terms["friction"]),
            w_drag=float(terms["drag"]),
            w_turn=float(terms["turn"]),
            w_tight=float(terms["tight"]),
            w_skid=float(terms["skid"]),
        )


class State(NamedTuple):
    x: jnp.ndarray      # across the slope (m)
    psi: jnp.ndarray    # heading from the fall line (rad); positive = towards +x
    v: jnp.ndarray      # speed (m/s)


class Telemetry(NamedTuple):
    """Per-step record, start-of-step values. Written to telemetry.csv for the champion."""

    x: jnp.ndarray
    y: jnp.ndarray
    psi: jnp.ndarray
    v: jnp.ndarray
    kappa: jnp.ndarray   # applied curvature, after the r_floor clip
    a_net: jnp.ndarray   # signed lateral load the snow must supply (m/s^2)
    G: jnp.ndarray       # total load in g
    theta: jnp.ndarray   # pitch (rad)


class StepOutput(NamedTuple):
    """What rollout, objectives and optimizers may read from a step (PLAN.md §12).
    Hazard terms and hard-DNF flags join this record with the DNF model (days 5-6)."""

    dt: jnp.ndarray
    ds: jnp.ndarray
    active: jnp.ndarray  # 1.0 before the finish line, 0.0 on padding steps after it
    telemetry: Telemetry


class Gates(NamedTuple):
    """Gate arrays, one entry per gate. Empty (shape (0,)) for gate-less test slopes."""

    y: jnp.ndarray       # down-slope position (m), a multiple of dy
    x_pole: jnp.ndarray  # turning-pole x (m)
    side: jnp.ndarray    # +1: pass on the +x side of the pole, -1: on the -x side
    width: jnp.ndarray   # 4-8 m (ICR 901.2.3)


class ProfileTerrain(NamedTuple):
    """Pitch profile theta(y), piecewise linear between knots. Ignores x (2D)."""

    y_knots: jnp.ndarray
    theta_knots: jnp.ndarray   # rad


class Course(NamedTuple):
    length: jnp.ndarray            # finish line y (m)
    x_start: jnp.ndarray
    piste_half_width: jnp.ndarray  # piste is |x| <= half width (enforced with the DNF model)
    terrain: ProfileTerrain
    gates: Gates


class Summary(NamedTuple):
    """Per-rollout results. T = D / v_mean holds exactly by construction of v_mean;
    the tests check v_mean against the time-average of v instead."""

    T: jnp.ndarray
    D: jnp.ndarray
    v_mean: jnp.ndarray
    v_end: jnp.ndarray
    max_G: jnp.ndarray
