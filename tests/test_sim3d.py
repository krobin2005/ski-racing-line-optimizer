"""3D physics checks (docs/ROADMAP_3D.md §3): analytic cases on planes, crests and compressions."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import make_params
from skiopt import sim3d
from skiopt.sim3d import path_loads

G = 9.81


def flat_grid(Z, ds=1.0, n0=-30.0, dn=0.5):
    return sim3d.make_grid(Z, ds, n0, dn)


def grid_from_fn(f, length=400.0, half=30.0, ds=1.0, dn=0.5):
    s = np.arange(0.0, length, ds)
    n = np.arange(-half, half + dn / 2, dn)
    return flat_grid(f(s[:, None], n[None, :]), ds, n[0], dn)


@pytest.fixture(autouse=True)
def float64():
    """Exactness checks run in float64; the simulator itself runs in float32 (rounding ~1e-4)."""
    with jax.enable_x64(True):
        yield


def test_straight_fall_line_on_a_plane_matches_analytic():
    """Gravity only: v^2 = v0^2 + 2 g sin(theta) * (surface distance), with surface distance s / cos(theta)."""
    p = make_params(on=("gravity",), v0=1.0)
    theta, L = math.radians(25), 300.0
    course = sim3d.plane_course(25, L)
    n = int(L)
    summary, _ = sim3d.rollout3d(lambda k, o, i: k[i], jnp.zeros(n), course, p, n)
    ell = L / math.cos(theta)
    assert float(summary.D) == pytest.approx(ell, rel=1e-5)
    assert float(summary.v_end) == pytest.approx(math.sqrt(1 + 2 * G * math.sin(theta) * ell), rel=1e-5)


def test_plane_loads_reduce_to_2d():
    """On an inclined plane, straight down: N = g cos(theta), no lateral load."""
    grid = grid_from_fn(lambda s, n: -math.tan(math.radians(20)) * s + 0 * n)
    L = path_loads(grid, 100.0, 0.0, 0.0, 0.0, 20.0, G)
    assert float(L["N"]) == pytest.approx(G * math.cos(math.radians(20)), rel=1e-5)
    assert float(L["F_lat"]) == pytest.approx(0.0, abs=1e-5)
    assert float(L["kappa_n"]) == pytest.approx(0.0, abs=1e-6)


def test_circular_turn_on_flat_ground():
    grid = grid_from_fn(lambda s, n: 0 * s + 0 * n)
    v, R = 18.0, 20.0
    L = path_loads(grid, 100.0, 0.0, 0.3, 1 / R, v, G)
    assert float(L["N"]) == pytest.approx(G, rel=1e-6)
    assert float(L["F_lat"]) == pytest.approx(v**2 / R, rel=1e-6)
    assert float(L["kappa_g"]) == pytest.approx(1 / R, rel=1e-6)


@pytest.mark.parametrize("sign,label", [(-1.0, "crest"), (1.0, "compression")])
def test_crest_unloads_and_compression_loads(sign, label):
    """Straight over z = sign (s - s0)^2 / (2 R) at its flat point: N = g + sign v^2 / R."""
    R, s0, v = 40.0, 200.0, 15.0
    grid = grid_from_fn(lambda s, n: sign * (s - s0) ** 2 / (2 * R) + 0 * n)
    L = path_loads(grid, s0, 0.0, 0.0, 0.0, v, G)
    assert float(L["N"]) == pytest.approx(G + sign * v**2 / R, rel=1e-3), label


def test_camber_traverse_needs_an_uphill_edge():
    """Straight along s on a slope that rises 10 degrees to the left: the edge must hold
    g sin(10 deg) towards the uphill side."""
    phi = math.radians(10)
    grid = grid_from_fn(lambda s, n: 0 * s + math.tan(phi) * n)
    L = path_loads(grid, 100.0, 0.0, 0.0, 0.0, 15.0, G)
    assert float(L["F_lat"]) == pytest.approx(G * math.sin(phi), rel=1e-4)
    assert float(L["N"]) == pytest.approx(G * math.cos(phi), rel=1e-4)


def test_fall_away_adds_load_and_camber_removes_it():
    """Turning left (kappa > 0) while the slope falls away to the right (outside of the turn):
    gravity adds to the lateral load. With the slope tilting into the turn, it subtracts."""
    phi, v, R = math.radians(8), 18.0, 20.0
    falls_right = grid_from_fn(lambda s, n: 0 * s + math.tan(phi) * n)    # rises to the left
    rises_right = grid_from_fn(lambda s, n: 0 * s - math.tan(phi) * n)
    away = path_loads(falls_right, 100.0, 0.0, 0.0, 1 / R, v, G)["F_lat"]
    into = path_loads(rises_right, 100.0, 0.0, 0.0, 1 / R, v, G)["F_lat"]
    assert float(away) > v**2 / R * math.cos(phi) > float(into)


def test_rollout_gradient_matches_finite_differences():
    if True:
        p = make_params()
        grid = grid_from_fn(lambda s, n: -0.35 * s + 0.05 * n + 0.8 * np.sin(s / 15.0))
        course = sim3d.Course3D(length=jnp.asarray(150.0), x_start=jnp.asarray(0.0),
                                piste_half_width=jnp.asarray(20.0), terrain=grid,
                                gates=sim3d.plane_course(20, 10).gates)
        n = 150
        kappa = 0.01 * jnp.sin(jnp.arange(n) / 9.0)
        T = jax.jit(lambda k: sim3d.rollout3d(lambda kk, o, i: kk[i], k, course, p, n)[0].T)
        d = jax.random.normal(jax.random.key(0), (n,), dtype=jnp.float64)
        analytic = float(jnp.dot(jax.grad(T)(kappa), d))
        eps = 1e-6
        numeric = float((T(kappa + eps * d) - T(kappa - eps * d)) / (2 * eps))
        assert analytic == pytest.approx(numeric, rel=1e-4)


# --- the Gran Risa course ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gran_risa_build():
    from skiopt.course3d import build_gran_risa

    with jax.enable_x64(False):
        return build_gran_risa(seed=0, smooth_m=8.0)


def test_gran_risa_set_matches_the_2025_race_and_the_rules(gran_risa_build):
    from skiopt import courses as C

    b = gran_risa_build
    assert b.spec.n_gates == 52 and b.spec.direction_changes == 51        # FIS result, run 1
    assert abs(b.meta["vertical_drop_model_m"] - 448) < 6
    assert 1200 < b.meta["surface_length_m"] < 1280
    assert [v for v in C.validate(b.validation_spec) if v.rule != "piste"] == []
    i = b.spec.gate_y.astype(int)
    outer = b.spec.outside_x
    assert np.all(np.minimum(b.spec.gate_x, outer) > b.piste_lo[i]) and np.all(np.maximum(b.spec.gate_x, outer) < b.piste_hi[i])


def test_gran_risa_coach_line_is_realistic(gran_risa_build):
    """Calibration guard: the unoptimized coach line passes every gate, is tracked within 0.5 m on
    the curving frame, and lands within 10% of the real fastest run (76.00 s)."""
    from skiopt.config import load_config
    from skiopt.objectives import ObjParams, make_trajectory, score
    from skiopt.policies import ArcLine
    from skiopt.types import SimParams

    with jax.enable_x64(False):
        b = gran_risa_build
        cfg = load_config()
        p, obj = SimParams.from_config(cfg), ObjParams.from_config(cfg)
        n = int(b.spec.length)
        pol = ArcLine(n, spec=b.spec, x_start=b.spec.x_start, dy=1.0, kappa_c=b.frame.kappa)
        summary, traj = make_trajectory(pol, p, n, rollout_fn=sim3d.rollout3d)(pol.zero(), b.course)
        assert int(score(summary, b.course, obj).n_missed) == 0
        assert np.abs(np.asarray(traj.telemetry.x) - np.asarray(pol.line(pol.zero()))[:n]).max() < 0.5
        assert float(summary.T) == pytest.approx(76.0, rel=0.10)
