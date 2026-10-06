"""Physics checks (PLAN.md §13, track A)."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from conftest import make_params, run_line
from skiopt.dynamics import PointMass2D, clip_kappa, lateral_load, turn_deceleration
from skiopt.types import State

G = 9.81


def test_frictionless_straight_run_matches_analytic():
    """Gravity only, constant pitch: v^2 = v0^2 + 2 g sin(theta) s, T = (v - v0) / (g sin(theta))."""
    p = make_params(on=("gravity",), v0=1.0)
    theta, L = math.radians(20), 500.0
    summary, _ = run_line(0.0, p, length=L, pitch_deg=20)

    a = G * math.sin(theta)
    v_end = math.sqrt(1.0 + 2 * a * L)
    assert float(summary.v_end) == pytest.approx(v_end, rel=1e-5)
    assert float(summary.T) == pytest.approx((v_end - 1.0) / a, rel=1e-5)
    assert float(summary.D) == pytest.approx(L, rel=1e-6)


def test_straight_run_reaches_terminal_speed(base_params):
    p = base_params
    theta = math.radians(20)
    v_terminal = math.sqrt(G * (math.sin(theta) - p.mu * math.cos(theta)) / p.c_d)
    summary, _ = run_line(0.0, p, length=4000.0, pitch_deg=20)
    assert float(summary.v_end) == pytest.approx(v_terminal, rel=1e-3)


def test_flat_ground_decelerates_monotonically(base_params):
    p = base_params._replace(v0=20.0)
    _, traj = run_line(0.0, p, length=200.0, pitch_deg=0.0)
    v = np.asarray(traj.telemetry.v)
    v = v[v > p.v_floor + 1e-3]
    assert len(v) > 10 and np.all(np.diff(v) < 0)


def test_turn_cost_bands(base_params):
    """Flat above 18 m, rising monotonically from 18 m to 12 m (carving term switched off)."""
    p = base_params._replace(w_turn=0.0)
    v = 20.0
    cost = lambda r: float(turn_deceleration(1.0 / r, v, 0.0, p))

    rounder = [cost(r) for r in (60.0, 40.0, 30.0, 25.0, 22.0, 18.0)]
    assert max(rounder) < 0.05 * cost(14.0)

    radii = np.linspace(18.0, 12.0, 25)
    costs = np.array([cost(r) for r in radii])
    assert np.all(np.diff(costs) > 0)


def test_curvature_tighter_than_floor_is_clipped(base_params):
    p = base_params
    assert float(clip_kappa(1 / 8, p)) == pytest.approx(1 / p.r_floor)
    assert float(clip_kappa(-1 / 8, p)) == pytest.approx(-1 / p.r_floor)

    state = State(jnp.asarray(0.0), jnp.asarray(0.2), jnp.asarray(20.0))
    theta = jnp.asarray(math.radians(20))
    dyn = PointMass2D()
    s_cmd, _ = dyn.step(state, 1 / 8, theta, p)
    s_floor, _ = dyn.step(state, 1 / p.r_floor, theta, p)
    for a, b in zip(s_cmd, s_floor):
        assert float(a) == pytest.approx(float(b))


def test_lateral_load_is_lower_before_the_fall_line():
    """Turning left (kappa < 0) from heading right (psi > 0) towards the fall line, gravity helps;
    after crossing it (psi < 0) gravity adds to the load."""
    kappa, v, theta = -1 / 20, 20.0, math.radians(20)
    before = abs(float(lateral_load(kappa, v, theta, 0.6, G)))
    at = abs(float(lateral_load(kappa, v, theta, 0.0, G)))
    after = abs(float(lateral_load(kappa, v, theta, -0.6, G)))
    assert before < at < after


def test_heading_stays_below_psi_max(base_params):
    p = base_params
    _, traj = run_line(1 / 13, p, length=200.0)
    assert float(np.max(np.abs(np.asarray(traj.telemetry.psi)))) < p.psi_max


def test_time_distance_and_mean_speed_are_consistent(base_params):
    """T = sum(dt), D = sum(ds) >= straight length, and v_mean = D / T equals the time-average
    of the speed trace."""
    n = 300
    k = np.where((np.arange(n) // 30) % 2 == 0, 1 / 25, -1 / 25)
    summary, traj = run_line(k, base_params, length=float(n))

    dt, ds = np.asarray(traj.dt), np.asarray(traj.ds)
    v = np.append(np.asarray(traj.telemetry.v), float(summary.v_end))
    time_avg_v = np.sum(0.5 * (v[:-1] + v[1:]) * dt) / np.sum(dt)

    assert float(summary.T) == pytest.approx(dt.sum(), rel=1e-6)
    assert float(summary.D) == pytest.approx(ds.sum(), rel=1e-6)
    assert float(summary.D) > n
    assert float(summary.v_mean) == pytest.approx(float(summary.D) / float(summary.T), rel=1e-6)
    assert float(summary.v_mean) == pytest.approx(time_avg_v, rel=1e-5)
