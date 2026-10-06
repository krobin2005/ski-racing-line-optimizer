"""Course checks (PLAN.md §13, track A): FIS validator, generator, padding and the course plot."""

import copy

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from skiopt import courses as C
from skiopt.config import CONFIG_DIR
from skiopt.objectives import ObjParams, gate_miss, score
from skiopt.run import run_experiment
from skiopt.types import Summary

FIS = C.load_fis()


def rules_broken(spec):
    return {v.rule for v in C.validate(spec, FIS)}


def easy():
    return copy.deepcopy(C.easy())


# --- validator -------------------------------------------------------------------------------


def test_easy_course_is_valid():
    assert C.validate(C.easy(), FIS) == []


def test_allowed_direction_changes_round_to_nearest():
    assert C.allowed_direction_changes(300, (11, 15)) == (33, 45)
    assert C.allowed_direction_changes(346, (11, 15)) == (38, 52)  # 38.06 -> 38, 51.9 -> 52


def test_delay_gates_add_gates_but_not_direction_changes():
    assert C.direction_changes([1, -1, -1, 1, -1]) == 4
    assert list(C.turning_gates([1, -1, -1, 1, -1])) == [0, 1, 3, 4]


def test_turning_poles_too_close():
    spec = easy()
    spec.gate_y[5] = spec.gate_y[4] + 6.0
    spec.gate_x[5] = spec.gate_x[4] + 1.0
    assert "pole_distance" in rules_broken(spec)


def test_gate_width_out_of_range():
    spec = easy()
    spec.gate_width[3] = 9.0
    assert "gate_width" in rules_broken(spec)
    spec.gate_width[3] = 3.5
    assert "gate_width" in rules_broken(spec)


def test_wrong_colour_order():
    spec = easy()
    spec.gate_colour[7] = spec.gate_colour[6]
    assert "colour_order" in rules_broken(spec)


def test_wrong_direction_change_count():
    spec = easy()
    keep = slice(0, 20)  # 20 gates for a 300 m drop; at least 33 are required
    for name in ("gate_y", "gate_x", "gate_side", "gate_width"):
        setattr(spec, name, getattr(spec, name)[keep])
    spec.gate_colour = spec.gate_colour[keep]
    assert "direction_changes" in rules_broken(spec)


def test_gate_outside_piste():
    spec = easy()
    assert spec.gate_side[11] == 1  # outside pole stands at x_pole + 7 m
    spec.gate_x[11] = 15.0          # so it reaches 22 m, past the 20 m piste edge
    assert "piste" in rules_broken(spec)


def test_turn_too_tight_for_the_model():
    spec = easy()
    spec.gate_x = spec.gate_side * 9.0  # 18 m offsets at 20-30 m spacing need turns under 13 m
    assert "turn_radius" in rules_broken(spec)


def test_vertical_drop_out_of_range():
    spec = easy()
    spec.theta_knots = spec.theta_knots * 0.5  # 10 degrees: about 150 m of drop
    assert "vertical_drop" in rules_broken(spec)


def test_gate_off_the_dy_grid():
    spec = easy()
    spec.gate_y = spec.gate_y.copy()
    spec.gate_y[2] += 0.5
    assert "grid" in rules_broken(spec)


# --- generator -------------------------------------------------------------------------------


def test_thousand_generated_courses_are_valid():
    delays = 0
    for seed in range(1000):
        spec = C.generate(seed, FIS)
        assert C.validate(spec, FIS) == [], seed
        assert 1500 >= spec.length >= 500
        delays += spec.n_gates - spec.direction_changes
    assert delays > 50  # delay gates actually occur


def test_generator_is_deterministic():
    a, b = C.generate(42, FIS), C.generate(42, FIS)
    np.testing.assert_array_equal(a.gate_x, b.gate_x)
    np.testing.assert_array_equal(a.gate_y, b.gate_y)
    assert a.gate_colour == b.gate_colour


def test_generated_turns_are_mostly_in_the_typical_band():
    radii = np.concatenate([C.implied_radius(C.generate(s, FIS), 0.4) for s in range(100)])
    assert np.median(radii) == pytest.approx(21, abs=2.5)
    assert radii.min() >= 13.0


# --- conversion and gate scoring -------------------------------------------------------------


def test_padded_courses_keep_their_gates():
    a, b = C.generate(1, FIS), C.generate(2, FIS)
    stacked = C.to_courses([a, b])
    n_max = max(a.n_gates, b.n_gates)
    assert stacked.gates.y.shape == (2, n_max)
    assert float(stacked.gates.valid[0].sum()) == a.n_gates
    assert float(stacked.gates.valid[1].sum()) == b.n_gates


def _summary_at(x_gates):
    x = jnp.asarray(x_gates, dtype=float)
    zero = jnp.asarray(0.0)
    return Summary(T=jnp.asarray(60.0), D=zero, v_mean=zero, v_end=zero, max_G=zero, x_gates=x)


def test_gate_pass_window_includes_clearance():
    spec = easy()
    course = C.to_course(spec)
    obj = ObjParams(clearance=0.4)
    s = spec.gate_side
    inside = spec.gate_x + s * 2.0
    hard, _ = gate_miss(_summary_at(inside), course, obj)
    assert float(hard.sum()) == 0.0

    on_pole = spec.gate_x.copy()  # touching the turning pole: 0.4 m inside the clearance
    hard, _ = gate_miss(_summary_at(on_pole), course, obj)
    np.testing.assert_allclose(np.asarray(hard), 0.4, atol=1e-6)

    beyond = spec.outside_x + s * 1.5  # 1.5 m past the outside pole
    hard, _ = gate_miss(_summary_at(beyond), course, obj)
    np.testing.assert_allclose(np.asarray(hard), 1.5, atol=1e-6)


def test_missed_gate_is_a_dq_and_smooth_loss_grows_with_miss():
    spec = easy()
    course = C.to_course(spec)
    obj = ObjParams()
    clean = score(_summary_at(spec.gate_x + spec.gate_side * 2.0), course, obj)
    assert float(clean.hard_score) == pytest.approx(60.0)
    assert int(clean.n_missed) == 0

    x = spec.gate_x + spec.gate_side * 2.0
    losses = []
    for miss in (0.5, 1.0, 2.0):
        x_bad = x.copy()
        x_bad[3] = spec.gate_x[3] - spec.gate_side[3] * miss  # wrong side of the turning pole
        s = score(_summary_at(x_bad), course, obj)
        assert int(s.n_missed) == 1
        assert float(s.hard_score) > obj.dq_base
        losses.append(float(s.smooth_loss))
    assert losses[0] < losses[1] < losses[2]


# --- runner and plot -------------------------------------------------------------------------


def test_runner_reports_missed_gates_on_a_gated_course(tmp_path):
    exp = {"experiment": "gated", "course": {"kind": "library", "name": "easy"},
           "policy": "open_loop_kappa", "methods": [{"name": "fall_line"}], "seeds": [0]}
    path = tmp_path / "gated.yaml"
    path.write_text(yaml.safe_dump(exp))
    run_experiment(path, results_dir=str(tmp_path / "results"))
    manifest = (tmp_path / "results" / "gated" / "fall_line" / "seed_0" / "manifest.json").read_text()
    import json

    m = json.loads(manifest)
    assert m["gates_missed"] > 0          # the straight fall line misses gates
    assert m["best_score"] > 1000.0       # so it is disqualified
    assert m["config"]["fis"]["icr_edition"].startswith("ICR")


def test_course_plot_renders(tmp_path):
    from skiopt.analysis.plots import plot_course

    spec = C.generate(16, FIS)
    ys = np.arange(0, spec.length)
    line = {"y": ys, "x": np.sin(ys / 20), "v": 15 + ys / 100}
    fig = plot_course(spec, line=line, missed=np.arange(spec.n_gates) == 3)
    out = tmp_path / "course.png"
    fig.savefig(out)
    assert out.stat().st_size > 10_000
