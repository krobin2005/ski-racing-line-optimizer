"""Core data types. All are NamedTuples of arrays/floats, so they are JAX pytrees and can be
passed through jit, vmap and grad. Course fields have fixed shapes so courses can be stacked."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp


class RiskParams(NamedTuple):
    """DNF model constants from configs/risk.yaml (PLAN.md §5). Switches are 1.0 / 0.0.
    The defaults switch everything off, so code that never loads risk.yaml sees no risk."""

    load_on: float = 0.0
    c_load: float = 0.0
    G_safe: float = 3.0
    beta_load: float = 8.0
    tight_on: float = 0.0
    c_tight: float = 0.0
    r_skid: float = 14.0
    beta_tight: float = 2000.0
    late_on: float = 0.0
    c_late: float = 0.0
    beta_late: float = 2000.0
    kappa_req_cap: float = 0.5
    slow_on: float = 0.0
    c_slow: float = 0.0
    v_slow: float = 35.0 / 3.6
    beta_slow: float = 2.0
    slow_after: float = 60.0
    gmax_on: float = 0.0
    G_max: float = 4.0
    vstop_on: float = 0.0
    v_stop: float = 10.0 / 3.6
    piste_on: float = 0.0
    gate_miss_on: float = 1.0
    wall_weight: float = 100.0
    wall_beta: float = 20.0
    clearance: float = 0.4       # body clearance at the turning pole (configs/fis_gs.yaml), for kappa_req

    @classmethod
    def from_config(cls, cfg: dict | None) -> "RiskParams":
        if not cfg:
            return cls()
        r = cfg["risk"] if "risk" in cfg else cfg
        on = float(r["enabled"])
        t, hard = r["terms"], r["hard"]
        return cls(
            load_on=on * float(t["load"]["enabled"]), c_load=float(t["load"]["c"]),
            G_safe=float(t["load"]["G_safe"]), beta_load=float(t["load"]["beta"]),
            tight_on=on * float(t["tight"]["enabled"]), c_tight=float(t["tight"]["c"]),
            r_skid=float(t["tight"]["r_skid"]), beta_tight=float(t["tight"]["beta"]),
            late_on=on * float(t["late"]["enabled"]), c_late=float(t["late"]["c"]),
            beta_late=float(t["late"]["beta"]), kappa_req_cap=float(t["late"]["kappa_req_cap"]),
            slow_on=on * float(t["slow"]["enabled"]), c_slow=float(t["slow"]["c"]),
            v_slow=float(t["slow"]["v_slow_kmh"]) / 3.6, beta_slow=float(t["slow"]["beta"]),
            slow_after=float(t["slow"]["after_m"]),
            gmax_on=float(hard["g_max"]["enabled"]), G_max=float(hard["g_max"]["value"]),
            vstop_on=float(hard["v_stop"]["enabled"]), v_stop=float(hard["v_stop"]["value_kmh"]) / 3.6,
            piste_on=float(hard["piste"]["enabled"]), gate_miss_on=float(hard["gate_miss"]["enabled"]),
            wall_weight=float(r["wall"]["weight"]), wall_beta=float(r["wall"]["beta"]),
        )


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
    risk: RiskParams = RiskParams()

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
            risk=RiskParams.from_config(cfg.get("risk"))._replace(
                clearance=float(cfg.get("fis", {}).get("model", {}).get("body_clearance_m", 0.4))),
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
    The hazard fields are filled in by the rollout (it knows the course); dynamics leaves them 0."""

    dt: jnp.ndarray
    ds: jnp.ndarray
    active: jnp.ndarray  # 1.0 before the finish line, 0.0 on padding steps after it
    telemetry: Telemetry
    hazard: jnp.ndarray = 0.0    # (4,) hazard accumulated this step (h * ds): load, tight, late, slow
    hard_dnf: jnp.ndarray = 0.0  # 1.0 if a hard limit (G_max, v_stop, piste) is broken this step
    wall: jnp.ndarray = 0.0      # smooth-objective wall penalty this step (already times ds)
    kappa_req: jnp.ndarray = 0.0 # curvature needed to still make the next gate (1/m)


class Gates(NamedTuple):
    """Gate arrays, one entry per gate. Empty (shape (0,)) for gate-less test slopes."""

    y: jnp.ndarray       # down-slope position (m), a multiple of dy
    x_pole: jnp.ndarray  # turning-pole x (m)
    side: jnp.ndarray    # +1: pass on the +x side of the pole, -1: on the -x side
    width: jnp.ndarray   # 4-8 m (ICR 901.2.3)
    valid: jnp.ndarray   # 1.0 for real gates, 0.0 for padding (courses stacked to equal gate counts)


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
    x_gates: jnp.ndarray  # x at each gate line y_g (gates sit on the dy grid, so this is exact)
    H: jnp.ndarray = 0.0          # cumulative hazard sum(h * ds); P_finish = exp(-H)
    H_by_cause: jnp.ndarray = 0.0 # (4,) load, tight, late, slow
    hard_dnf: jnp.ndarray = 0.0   # 1.0 if any hard limit was broken
    wall: jnp.ndarray = 0.0       # total smooth wall penalty
