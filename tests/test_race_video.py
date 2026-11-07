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


class _StraightFrame:
    """A straight course along +N, for testing the finish-area extension."""

    length = 100.0
    s = np.linspace(0, 100.0, 11)
    tx, ty = np.zeros(11), np.ones(11)

    def to_map(self, s, n):
        s = np.clip(s, 0, self.length)
        return -np.asarray(n, float) + 0 * s, s + 0 * np.asarray(n, float)


class _Flat:
    def height(self, E, N):
        return np.zeros_like(np.asarray(E, float))


def test_racers_coast_through_the_finish_and_stop():
    f = _StraightFrame()
    E, N = rv.map_ext(f, [100.0, 130.0], [2.0, 2.0])
    np.testing.assert_allclose(N, [100.0, 130.0])   # the course carries on straight past the line
    np.testing.assert_allclose(E, [-2.0, -2.0])
    t, s, E, N, Z, v = rv.coast(np.array([0.0, 5.0]), np.array([0.0, 100.0]), np.array([0.0, 2.0]),
                                np.array([20.0, 20.0]), 5.0, f, _Flat())
    assert t[0] > 5.0 and np.all(np.diff(s) > 0) and s[0] > 100.0
    assert v[-1] == pytest.approx(0.0, abs=1e-9)
    assert s[-1] - 100.0 == pytest.approx(20.0 ** 2 / (2 * rv.RUNOUT_DECEL))  # stopping distance v^2 / 2a


def test_gate_flags_leave_the_gap_open():
    # Gran Risa gate 2: turning pole at n = 7.58, outside pole at 14.08; the optimized lines pass at about n = 8.0
    (i0, i1), (o0, o1) = rv.flag_spans(7.58, 14.08)
    assert max(i0, i1) <= 7.58 and min(o0, o1) >= 14.08
    (i0, i1), (o0, o1) = rv.flag_spans(-0.49, -7.29)  # a gate turning the other way
    assert min(i0, i1) >= -0.49 and max(o0, o1) <= -7.29
