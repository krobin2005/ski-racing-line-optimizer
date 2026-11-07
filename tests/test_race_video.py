"""Race video (skiopt/analysis/race_video.py): replay logic, and one rendered frame if PyVista is installed."""

import numpy as np
import pytest

from skiopt.analysis import race_video as rv


def _racer(T=10.0, key="cmaes", length=200.0):
    t = np.linspace(0, T, 101)
    s = np.linspace(0, length, 101)
    z = np.zeros_like(t)
    return rv.Racer(key, key, "#ff0000", t, s, s.copy(), z.copy(), -0.5 * s, np.full_like(t, length / T),
                    np.ones_like(t), np.full_like(t, 0.3), np.full_like(t, 0.01), T)


def test_state_interpolates_and_clamps_at_the_finish():
    r = _racer()
    mid = r.at(5.0)
    assert mid["s"] == pytest.approx(100.0) and not mid["finished"]
    assert np.linalg.norm(mid["heading"]) == pytest.approx(1.0)
    end = r.at(99.0)
    assert end["s"] == pytest.approx(200.0) and end["finished"]
    assert r.time_at_s(50.0) == pytest.approx(2.5)


def test_pace_ghost_runs_the_leaders_line_in_schwarzs_time():
    leader = _racer(T=74.5)
    ghost = rv.pace_ghost(leader)
    assert ghost.T == pytest.approx(rv.SCHWARZ_T)
    assert ghost.t[-1] == pytest.approx(rv.SCHWARZ_T)
    np.testing.assert_allclose(ghost.E, leader.E)                    # same line
    assert ghost.v.mean() < leader.v.mean()                            # slower, since 76.00 > 74.5


def test_fis_surnames():
    assert rv._surname("SCHWARZ Marco") == "Schwarz"
    assert rv._surname("MCGRATH Atle Lie") == "McGrath"
    assert rv._surname("PINHEIRO BRAATHEN Lucas") == "Pinheiro Braathen"


def test_one_frame_renders_on_the_gran_risa(tmp_path, monkeypatch):
    pytest.importorskip("pyvista")

    def fake_racers(results, methods, build, hf):
        f = build.frame
        s = np.linspace(0, f.length, 400)
        E, N = f.to_map(s, np.zeros_like(s))
        t = np.linspace(0, 75.0, 400)
        return [rv.Racer("cmaes", "CMA-ES", "#ff3b30", t, s, E, N, hf.height(E, N), np.full_like(t, 17.0),
                         np.ones_like(t), np.full_like(t, 0.5), np.full_like(t, 0.02), 75.0)]

    monkeypatch.setattr(rv, "load_racers", fake_racers)
    out = rv.render(tmp_path / "results", tmp_path / "frame.png", size=(640, 360), still=20.0)
    from PIL import Image

    img = np.asarray(Image.open(out))
    assert img.shape == (360, 640, 3)
    assert img.std() > 20  # not a blank frame
