"""Courses: FIS GS rules, validator, random course generator and hand-built library (PLAN.md §2).

A `CourseSpec` is the plain-Python description of a course (NumPy arrays, colours, metadata). It is
what the generator produces, the validator checks and the plots draw. `to_course` converts it to
the JAX `Course` pytree the simulator runs on, optionally padded so courses can be stacked.

Geometry conventions:
- y runs down the fall line along the slope surface; x runs across, piste centred on x = 0.
- side = +1 means the skier passes on the +x side of the turning pole, -1 on the -x side.
  The outside pole stands at x_pole + side * width, so the pass window is between the two.
- In GS the turning pole is on the inside of the turn, so a gate left of centre is passed
  further left (side = -1) and a gate right of centre further right (side = +1).
- A direction change is a turn. The number of turns is 1 + the number of side switches between
  consecutive gates; a delay gate (same side as the previous gate) adds a gate but no turn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from skiopt.config import CONFIG_DIR, load_yaml
from skiopt.types import Course, Gates, ProfileTerrain

PISTE_WIDTH_M = 40.0  # ICR 902.1


def load_fis(path: Path = CONFIG_DIR / "fis_gs.yaml") -> dict:
    return load_yaml(path)


# --------------------------------------------------------------------------------------------
# Course description


@dataclass
class CourseSpec:
    name: str
    length: float                      # finish line y (m)
    y_knots: np.ndarray                # pitch profile knots (m)
    theta_knots: np.ndarray            # pitch at knots (rad)
    gate_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gate_x: np.ndarray = field(default_factory=lambda: np.zeros(0))       # turning pole x
    gate_side: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gate_width: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gate_colour: list[str] = field(default_factory=list)
    sex: str = "men"
    x_start: float = 0.0
    piste_width: float = PISTE_WIDTH_M
    seed: int | None = None

    @property
    def n_gates(self) -> int:
        return len(self.gate_y)

    @property
    def outside_x(self) -> np.ndarray:
        return self.gate_x + self.gate_side * self.gate_width

    @property
    def vertical_drop(self) -> float:
        return vertical_drop(self.y_knots, self.theta_knots, self.length)

    @property
    def direction_changes(self) -> int:
        return direction_changes(self.gate_side)

    def pitch_at(self, y) -> np.ndarray:
        return np.interp(y, self.y_knots, self.theta_knots)


def vertical_drop(y_knots, theta_knots, y_end: float) -> float:
    """Vertical drop (m) from y = 0 to y_end for a piecewise-linear pitch profile."""
    ys = np.linspace(0.0, y_end, max(int(y_end) * 4, 2))
    return float(np.trapezoid(np.sin(np.interp(ys, y_knots, theta_knots)), ys))


def direction_changes(sides) -> int:
    sides = np.asarray(sides)
    if len(sides) == 0:
        return 0
    return 1 + int(np.sum(sides[1:] != sides[:-1]))


def allowed_direction_changes(vd: float, pct: tuple[float, float]) -> tuple[int, int]:
    """ICR 901.2.4: 11-15% of the vertical drop in metres, rounding the decimals up or down."""
    lo, hi = pct[0] / 100 * vd, pct[1] / 100 * vd
    return math.floor(lo + 0.5), math.floor(hi + 0.5)


def turning_gates(sides) -> np.ndarray:
    """Indices of gates that start a new direction (the first gate, and every side switch).
    Delay gates, which repeat the previous side, are excluded."""
    sides = np.asarray(sides)
    return np.concatenate([[0], np.flatnonzero(sides[1:] != sides[:-1]) + 1]).astype(int) if len(sides) else np.zeros(0, int)


def implied_radius(spec: "CourseSpec", margin: float) -> np.ndarray:
    """Turn radius (m) a skier needs between each pair of successive turning gates.

    Two equal circular arcs covering a fall-line distance L while swinging w across have radius
    R = (L^2 + w^2) / (4 w). The skier passes `margin` outside both turning poles, so
    w = |x_b - x_a| + 2 * margin. One value per pair of successive turning gates."""
    t = turning_gates(spec.gate_side)
    L = np.diff(spec.gate_y[t])
    w = np.abs(np.diff(spec.gate_x[t])) + 2 * margin
    return (L**2 + w**2) / (4 * w)


def offsets_for_radius(spacing, radius, margin: float, min_offset: float) -> np.ndarray:
    """Across-slope distance between successive turning poles that makes `implied_radius` equal
    `radius` for fall-line spacing `spacing` (the smaller root of w^2 - 4 R w + L^2 = 0)."""
    radius = np.maximum(radius, 0.51 * np.asarray(spacing))
    swing = 2 * radius - np.sqrt(4 * radius**2 - np.asarray(spacing) ** 2)
    return np.maximum(swing - 2 * margin, min_offset)


def to_course(spec: CourseSpec, max_gates: int | None = None, max_knots: int | None = None) -> Course:
    """Convert to the JAX pytree. Padding gates are marked invalid and placed at y = 0; padding
    knots repeat the last knot, which leaves the pitch profile unchanged."""
    n = spec.n_gates
    g_pad = (max_gates or n) - n
    k_pad = (max_knots or len(spec.y_knots)) - len(spec.y_knots)
    if g_pad < 0 or k_pad < 0:
        raise ValueError("max_gates / max_knots smaller than the course")

    def pad(a, value=0.0):
        return jnp.asarray(np.concatenate([np.asarray(a, dtype=float), np.full(g_pad, value)]))

    y_knots = np.concatenate([spec.y_knots, np.full(k_pad, spec.y_knots[-1])])
    theta_knots = np.concatenate([spec.theta_knots, np.full(k_pad, spec.theta_knots[-1])])
    return Course(
        length=jnp.asarray(float(spec.length)),
        x_start=jnp.asarray(float(spec.x_start)),
        piste_half_width=jnp.asarray(spec.piste_width / 2),
        terrain=ProfileTerrain(jnp.asarray(y_knots), jnp.asarray(theta_knots)),
        gates=Gates(
            y=pad(spec.gate_y), x_pole=pad(spec.gate_x), side=pad(spec.gate_side, 1.0),
            width=pad(spec.gate_width, 4.0), valid=pad(np.ones(n)),
        ),
    )


def to_courses(specs: list[CourseSpec]) -> Course:
    """Stack several courses (padded to equal gate and knot counts) along a leading axis."""
    import jax

    max_gates = max(s.n_gates for s in specs)
    max_knots = max(len(s.y_knots) for s in specs)
    courses = [to_course(s, max_gates, max_knots) for s in specs]
    return jax.tree.map(lambda *xs: jnp.stack(xs), *courses)


# --------------------------------------------------------------------------------------------
# Validator


@dataclass(frozen=True)
class Violation:
    rule: str        # short code, e.g. "pole_distance"
    article: str     # ICR article, or "model" for our own constraints
    message: str
    gate: int | None = None


def validate(spec: CourseSpec, fis: dict | None = None, dy: float = 1.0) -> list[Violation]:
    """Check a course against the GS rules in configs/fis_gs.yaml. Returns [] when valid."""
    fis = fis or load_fis()
    rules, model = fis["rules"], fis["model"]
    out: list[Violation] = []
    n = spec.n_gates

    vd = spec.vertical_drop
    lo, hi = rules["vertical_drop_m"][spec.sex]
    if not lo <= vd <= hi:
        out.append(Violation("vertical_drop", "901.1", f"vertical drop {vd:.0f} m outside {lo}-{hi} m"))

    if n == 0:
        out.append(Violation("no_gates", "901.2", "course has no gates"))
        return out

    if len({len(spec.gate_y), len(spec.gate_x), len(spec.gate_side), len(spec.gate_width),
            len(spec.gate_colour)}) != 1:
        out.append(Violation("shape", "model", "gate arrays have different lengths"))
        return out

    if not np.all(np.isin(spec.gate_side, (-1, 1))):
        out.append(Violation("side", "model", "every gate side must be +1 or -1"))

    w_lo, w_hi = rules["gate_width_m"]
    for i in np.flatnonzero((spec.gate_width < w_lo) | (spec.gate_width > w_hi)):
        out.append(Violation("gate_width", "901.2.3", f"gate width {spec.gate_width[i]:.1f} m outside {w_lo}-{w_hi} m", int(i)))

    d_min = rules["min_turning_pole_distance_m"]
    dist = np.hypot(np.diff(spec.gate_x), np.diff(spec.gate_y))
    for i in np.flatnonzero(dist < d_min):
        out.append(Violation("pole_distance", "901.2.3",
                             f"turning poles of gates {i} and {i + 1} are {dist[i]:.1f} m apart (< {d_min} m)", int(i + 1)))

    dc_lo, dc_hi = allowed_direction_changes(vd, rules["direction_changes_pct_vd"])
    dc = spec.direction_changes
    if not dc_lo <= dc <= dc_hi:
        out.append(Violation("direction_changes", "901.2.4",
                             f"{dc} direction changes, allowed {dc_lo}-{dc_hi} for {vd:.0f} m vertical drop"))

    colours = rules["colours"]
    first = colours.index(spec.gate_colour[0]) if spec.gate_colour[0] in colours else 0
    for i, c in enumerate(spec.gate_colour):
        if c != colours[(first + i) % len(colours)]:
            out.append(Violation("colour_order", "901.2.2", f"gate {i} is {c}; colours must alternate", i))
            break

    half = spec.piste_width / 2
    for i in np.flatnonzero((np.abs(spec.gate_x) > half) | (np.abs(spec.outside_x) > half)):
        out.append(Violation("piste", "902.1", f"gate {i} extends outside the {spec.piste_width:.0f} m piste", int(i)))

    if n >= 3:
        spread = (np.percentile(dist, 90) - np.percentile(dist, 10)) / np.median(dist)
        if spread < rules["min_distance_spread"]:
            out.append(Violation("variety", "903.1.3",
                                 f"gate distances too uniform (spread {spread:.2f} < {rules['min_distance_spread']})"))

    margin = model["body_clearance_m"]
    radii = implied_radius(spec, margin)
    r_min = model["min_implied_radius_m"]
    t = turning_gates(spec.gate_side)
    for k in np.flatnonzero(radii < r_min):
        out.append(Violation("turn_radius", "model",
                             f"gates {t[k]} to {t[k + 1]} need a {radii[k]:.1f} m turn (< {r_min} m)", int(t[k + 1])))

    if np.any(np.diff(spec.gate_y) <= 0):
        out.append(Violation("gate_order", "model", "gate y positions must strictly increase"))
    off_grid = np.abs(spec.gate_y / dy - np.round(spec.gate_y / dy)) > 1e-9
    for i in np.flatnonzero(off_grid):
        out.append(Violation("grid", "model", f"gate {i} at y = {spec.gate_y[i]} is not a multiple of dy = {dy}", int(i)))
    if spec.gate_y[0] < model["first_gate_min_y_m"]:
        out.append(Violation("first_gate", "model", f"first gate at {spec.gate_y[0]:.0f} m is too close to the start"))
    if spec.length - spec.gate_y[-1] < model["finish_margin_m"]:
        out.append(Violation("finish_margin", "model", f"last gate is {spec.length - spec.gate_y[-1]:.0f} m from the finish"))
    return out


# --------------------------------------------------------------------------------------------
# Generator


def _pitch_profile(rng: np.random.Generator, g: dict, y_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Random piecewise-linear pitch profile: a bounded random walk over knots."""
    lo, hi = np.radians(g["pitch_deg"])
    ys, thetas = [0.0], [rng.uniform(lo + 0.2 * (hi - lo), hi - 0.3 * (hi - lo))]
    while ys[-1] < y_max:
        ys.append(ys[-1] + rng.uniform(*g["pitch_knot_spacing_m"]))
        thetas.append(float(np.clip(thetas[-1] + rng.normal(0.0, np.radians(5.0)), lo, hi)))
    return np.array(ys), np.array(thetas)


def _length_for_drop(y_knots, theta_knots, target: float, dy: float) -> float:
    ys = np.arange(0.0, y_knots[-1], dy)
    drop = np.concatenate([[0.0], np.cumsum(np.sin(np.interp(ys[:-1] + dy / 2, y_knots, theta_knots)) * dy)])
    return float(ys[np.searchsorted(drop, target)])


def _section_spacings(rng: np.random.Generator, g: dict, n_intervals: int) -> np.ndarray:
    """Split the gate intervals into rhythm sections (short / medium / long), at least two kinds."""
    k = int(rng.integers(g["sections"][0], g["sections"][1] + 1))
    k = max(1, min(k, n_intervals // 3))
    cuts = np.sort(rng.choice(np.arange(3, n_intervals - 2), size=k - 1, replace=False)) if k > 1 else []
    sizes = np.diff(np.concatenate([[0], cuts, [n_intervals]])).astype(int)
    kinds = list(g["section_spacing_m"])
    labels = rng.choice(kinds, size=k)
    if k > 1 and len(set(labels)) == 1:
        labels[rng.integers(k)] = rng.choice([s for s in kinds if s != labels[0]])
    return np.concatenate([rng.uniform(*g["section_spacing_m"][lab], size=s) for lab, s in zip(labels, sizes)])


def generate(seed: int, fis: dict | None = None, dy: float = 1.0, name: str | None = None) -> CourseSpec:
    """A random FIS-valid GS course. Deterministic in `seed`. Retries internally until the course
    passes `validate`; raises if it cannot within `max_attempts`."""
    fis = fis or load_fis()
    rules, model, g = fis["rules"], fis["model"], fis["generator"]
    rng = np.random.default_rng(seed)

    for _ in range(g["max_attempts"]):
        y_knots, theta_knots = _pitch_profile(rng, g, y_max=3000.0)
        vd_target = rng.uniform(*g["vertical_drop_m"])
        length = _length_for_drop(y_knots, theta_knots, vd_target, dy)
        theta_end = float(np.interp(length, y_knots, theta_knots))
        keep = y_knots < length
        y_knots = np.append(y_knots[keep], length)
        theta_knots = np.append(theta_knots[keep], theta_end)
        if length > g["max_length_m"]:
            continue
        vd = vertical_drop(y_knots, theta_knots, length)

        dc_lo, dc_hi = allowed_direction_changes(vd, rules["direction_changes_pct_vd"])
        n_turns = int(rng.integers(dc_lo, dc_hi + 1))
        n_delay = int(rng.integers(g["delay_gates"][0], g["delay_gates"][1] + 1))

        # Turning gates first. A delay gate later splits one long interval in two.
        y_first = rng.uniform(*g["first_gate_y_m"])
        available = length - model["finish_margin_m"] - y_first
        spacing = _section_spacings(rng, g, n_turns - 1)
        delay_after = rng.choice(np.arange(1, n_turns - 2), size=n_delay, replace=False) if n_delay else []
        for j in delay_after:
            spacing[j] = rng.uniform(*g["delay_spacing_m"])
        spacing *= available / spacing.sum() * rng.uniform(0.93, 1.0)  # stretch to fit, leave finish room
        normal = np.setdiff1d(np.arange(n_turns - 1), delay_after)
        if spacing[normal].min() < 15.0 or spacing[normal].max() > 42.0:
            continue
        if n_delay and spacing[delay_after].min() < 32.0:  # keep >= 16 m either side of a delay gate
            continue
        turn_y = y_first + np.concatenate([[0.0], np.cumsum(spacing)])

        # Lateral swing from a target turn radius: two arcs of radius R covering a gate interval
        # of length L swing w = 2R - sqrt(4R^2 - L^2) across (see implied_radius).
        margin = model["body_clearance_m"] + g["line_margin_m"]
        radius = np.clip(rng.normal(*g["turn_radius_m"][:2], size=n_turns - 1), *g["turn_radius_m"][2:])
        dx = offsets_for_radius(spacing, radius, margin, g["min_offset_m"])
        sides = np.empty(n_turns)
        sides[0] = rng.choice([-1.0, 1.0])
        sides[1:] = sides[0] * (-1.0) ** np.arange(1, n_turns)
        turn_x = np.concatenate([[0.0], np.cumsum(sides[1:] * dx)])
        turn_x -= 0.5 * (turn_x.max() + turn_x.min())
        width = rng.uniform(*g["gate_width_m"], size=n_turns).round(1)

        # Delay gates: halfway down the interval, on the skier's traverse between the two turning
        # gates, passed on the same side as the previous gate, with the pole set just inside the line.
        gate_y, gate_x, gate_side, gate_width = [list(a) for a in (turn_y, turn_x, sides, width)]
        for j in sorted(delay_after, reverse=True):
            line_x = 0.5 * (turn_x[j] + turn_x[j + 1])
            inset = rng.uniform(*g["delay_pole_inset_m"])
            gate_y.insert(j + 1, 0.5 * (turn_y[j] + turn_y[j + 1]))
            gate_x.insert(j + 1, line_x - sides[j] * (model["body_clearance_m"] + inset))
            gate_side.insert(j + 1, sides[j])
            gate_width.insert(j + 1, round(float(rng.uniform(*g["gate_width_m"])), 1))
        gate_y = np.round(np.array(gate_y) / dy) * dy
        gate_x, sides, width = np.array(gate_x).round(2), np.array(gate_side), np.array(gate_width)
        n_gates = len(gate_y)
        first_colour = int(rng.integers(2))
        colours = [rules["colours"][(first_colour + i) % 2] for i in range(n_gates)]

        spec = CourseSpec(
            name=name or f"gen_{seed}", length=length, y_knots=y_knots, theta_knots=theta_knots,
            gate_y=gate_y, gate_x=gate_x, gate_side=sides, gate_width=width, gate_colour=colours,
            sex=g["sex"], seed=seed,
        )
        if not validate(spec, fis, dy):
            return spec
    raise RuntimeError(f"could not generate a valid course for seed {seed}")


# --------------------------------------------------------------------------------------------
# Hand-built library


def constant_pitch(length_m: float, pitch_deg: float, x_start: float = 0.0) -> Course:
    """Gate-less test slope as a JAX Course (used by the physics tests and the smoke experiment)."""
    return to_course(constant_pitch_spec(length_m, pitch_deg, x_start))


def constant_pitch_spec(length_m: float, pitch_deg: float, x_start: float = 0.0) -> CourseSpec:
    t = math.radians(pitch_deg)
    return CourseSpec(name=f"slope_{pitch_deg:g}deg", length=float(length_m),
                      y_knots=np.array([0.0, float(length_m)]), theta_knots=np.array([t, t]),
                      x_start=float(x_start))


def easy() -> CourseSpec:
    """A gentle, readable GS: constant 20 degree pitch, 300 m vertical drop, open gates in three
    rhythm sections (24 m, 30 m, 22 m apart), offsets set for 20 m turns, 7 m gates, no delays."""
    theta = math.radians(20.0)
    length = float(round(300.0 / math.sin(theta)))          # 877 m
    spacing = np.array([24.0] * 11 + [30.0] * 9 + [20.0] * 13)  # 34 gates, last at 819 m
    gate_y = 25.0 + np.concatenate([[0.0], np.cumsum(spacing)])
    n = len(gate_y)
    sides = np.where(np.arange(n) % 2 == 0, -1.0, 1.0)
    dx = offsets_for_radius(spacing, np.full(n - 1, 20.0), margin=0.7, min_offset=2.0)
    gate_x = np.concatenate([[0.0], np.cumsum(sides[1:] * dx)])
    gate_x = (gate_x - 0.5 * (gate_x.max() + gate_x.min())).round(2)
    return CourseSpec(
        name="easy", length=length, y_knots=np.array([0.0, length]), theta_knots=np.array([theta, theta]),
        gate_y=gate_y, gate_x=gate_x, gate_side=sides, gate_width=np.full(n, 7.0),
        gate_colour=["red" if i % 2 == 0 else "blue" for i in range(n)],
    )


LIBRARY = {"easy": easy}


def spec_from_config(spec: dict) -> CourseSpec:
    kind = spec["kind"]
    if kind == "constant_pitch":
        return constant_pitch_spec(spec["length_m"], spec["pitch_deg"], spec.get("x_start", 0.0))
    if kind == "generated":
        return generate(int(spec["seed"]))
    if kind == "library":
        return LIBRARY[spec["name"]]()
    raise ValueError(f"unknown course kind {kind!r}")


def from_config(spec: dict) -> Course:
    return to_course(spec_from_config(spec))


def no_gates() -> Gates:
    empty = jnp.zeros((0,))
    return Gates(y=empty, x_pole=empty, side=empty, width=empty, valid=empty)
