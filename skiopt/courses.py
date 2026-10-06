"""Courses. For now only gate-less test slopes; the FIS rules, validator, generator and the
hand-built library land on days 3-4 (PLAN.md §2)."""

from __future__ import annotations

import math

import jax.numpy as jnp

from skiopt import terrain
from skiopt.types import Course, Gates

PISTE_WIDTH_M = 40.0  # ICR 902.1


def no_gates() -> Gates:
    empty = jnp.zeros((0,))
    return Gates(y=empty, x_pole=empty, side=empty, width=empty)


def constant_pitch(length_m: float, pitch_deg: float, x_start: float = 0.0) -> Course:
    return Course(
        length=jnp.asarray(float(length_m)),
        x_start=jnp.asarray(float(x_start)),
        piste_half_width=jnp.asarray(PISTE_WIDTH_M / 2),
        terrain=terrain.constant(math.radians(pitch_deg), length_m),
        gates=no_gates(),
    )


def from_config(spec: dict) -> Course:
    kind = spec["kind"]
    if kind == "constant_pitch":
        return constant_pitch(spec["length_m"], spec["pitch_deg"], spec.get("x_start", 0.0))
    raise ValueError(f"unknown course kind {kind!r}")
