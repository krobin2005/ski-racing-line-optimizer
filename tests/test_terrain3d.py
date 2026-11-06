"""Real-terrain pipeline checks (docs/ROADMAP_3D.md): geodesy, the Gran Risa frame and resampling."""

import numpy as np
import pytest

from skiopt import terrain3d as T
from skiopt.geo import latlon_to_utm


def test_utm_on_the_central_meridian():
    e, n = latlon_to_utm(0.0, 9.0)
    assert float(e) == pytest.approx(500000.0) and float(n) == pytest.approx(0.0, abs=1e-6)
    # Meridian arc to 46 N on GRS80 is 5,096,086.0 m; scaled by k0 = 0.9996.
    _, n46 = latlon_to_utm(46.0, 9.0)
    assert float(n46) == pytest.approx(0.9996 * 5096086.0, abs=2.0)


@pytest.fixture(scope="module")
def gran_risa():
    hf, frame, report = T.gran_risa()
    return hf, frame, report, T.resample(hf, frame)


def test_gran_risa_matches_published_facts(gran_risa):
    hf, frame, report, ct = gran_risa
    j0 = int(np.argmin(np.abs(ct.n)))
    start, finish = report["start_offset_m"], frame.length - report["finish_offset_m"]
    z = ct.Z[:, j0]
    drop = np.interp(start, frame.s, z) - np.interp(finish, frame.s, z)
    assert drop == pytest.approx(448, abs=8)                     # published vertical drop
    slope_length = np.sum(np.hypot(np.diff(frame.s), np.diff(z))[int(start):int(finish)])
    assert 1150 < slope_length < 1350                            # published 1,225-1,255 m
    grad = ct.fall_line_gradient()[int(start):int(finish)]
    assert 0.30 < grad.mean() < 0.45                             # published mean ~36%


def test_frame_is_valid_across_the_piste(gran_risa):
    _, frame, report, _ = gran_risa
    assert report["min_centreline_radius_m"] > 25 * 2            # |n| kappa < 0.5 out to 25 m
    assert report["max_deviation_from_osm_m"] < 20
    np.testing.assert_allclose(np.hypot(frame.tx, frame.ty), 1.0, atol=1e-9)


def test_resampled_grid_matches_the_height_field(gran_risa):
    hf, frame, _, ct = gran_risa
    i = np.array([10, 400, 900])
    j = np.array([5, 60, 110])
    E, N = frame.to_map(frame.s[i], ct.n[j])
    np.testing.assert_allclose(ct.Z[i, j], hf.height(E, N), atol=1e-6)
