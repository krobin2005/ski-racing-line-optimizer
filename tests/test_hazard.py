"""DNF model checks (PLAN.md §13, track A): hazard terms, hard limits, switches, risk map."""

import copy
import math

import jax.numpy as jnp
import numpy as np
import pytest

from conftest import run_line
from skiopt import courses as C
from skiopt import hazard
from skiopt.config import load_config
from skiopt.objectives import ObjParams, expected_score, p_dnf, t_dnf
from skiopt.types import RiskParams, SimParams


def risk_cfg(**term_switches):
    cfg = copy.deepcopy(load_config())
    for name, on in term_switches.items():
        if name in cfg["risk"]["terms"]:
            cfg["risk"]["terms"][name]["enabled"] = on
        else:
            cfg["risk"]["hard"][name]["enabled"] = on
    return cfg


def params_from(cfg):
    return SimParams.from_config(cfg)


R = RiskParams.from_config(load_config()["risk"])


def test_moderate_straight_run_has_almost_no_risk():
    p = params_from(load_config())
    summary, _ = run_line(0.0, p, length=300.0, pitch_deg=15.0)
    assert float(p_dnf(summary)) < 1e-3
    assert float(summary.hard_dnf) == 0.0


@pytest.mark.parametrize("term,index", [("load", 0), ("tight", 1), ("late", 2), ("slow", 3)])
def test_each_term_rises_monotonically_with_its_driver(term, index):
    G, kappa, kreq, v = 2.0, 1 / 20, 0.0, 20.0
    drivers = {
        "load": [dict(G=g) for g in np.linspace(2.5, 3.8, 14)],
        "tight": [dict(kappa=1 / r) for r in np.linspace(16, 12, 14)],
        "late": [dict(kappa_req=k) for k in np.linspace(0.05, 0.2, 14)],
        "slow": [dict(v=v_) for v_ in np.linspace(12, 4, 14)],
    }[term]
    values = []
    for d in drivers:
        args = dict(G=G, kappa=kappa, kappa_req=kreq, v=v, y=500.0) | d
        values.append(float(hazard.hazard_terms(jnp.asarray(args["G"]), args["kappa"], args["kappa_req"],
                                                args["v"], args["y"], R)[index]))
    assert np.all(np.diff(values) > 0)


def test_slow_risk_is_off_in_the_start_zone():
    h = hazard.hazard_terms(jnp.asarray(1.0), 0.0, 0.0, 5.0, 10.0, R)
    assert float(h[3]) == 0.0


@pytest.mark.parametrize("case", ["g_max", "v_stop", "piste"])
def test_hard_limits_are_detected(case):
    G, v, x = {"g_max": (4.2, 20.0, 0.0), "v_stop": (1.0, 2.0, 0.0), "piste": (1.0, 20.0, 21.0)}[case]
    broken, wall = hazard.hard_limits(jnp.asarray(G), v, x, 20.0, R)
    assert float(broken) == 1.0 and float(wall) > 0
    off = R._replace(**{"g_max": {"gmax_on": 0.0}, "v_stop": {"vstop_on": 0.0}, "piste": {"piste_on": 0.0}}[case])
    broken, wall = hazard.hard_limits(jnp.asarray(G), v, x, 20.0, off)
    assert float(broken) == 0.0 and float(wall) == pytest.approx(0.0, abs=1e-6)


def test_late_line_raises_kappa_required_before_the_gate():
    spec = C.easy()
    course = C.to_course(spec)
    y_gate, x_gate, side = spec.gate_y[3], spec.gate_x[3], spec.gate_side[3]
    x_far_off = x_gate - side * 6.0  # heading straight down, well on the wrong side of the gate
    k = [float(hazard.kappa_required(jnp.asarray(x_far_off), y_gate - d, jnp.asarray(0.0), course, 0.4, 0.5))
         for d in (20.0, 15.0, 10.0, 5.0)]
    assert k[0] > 0 and np.all(np.diff(k) > 0)
    inside = x_gate + side * 2.0  # already lined up inside the pass window: no late risk
    assert float(hazard.kappa_required(jnp.asarray(inside), y_gate - 10, jnp.asarray(0.0), course, 0.4, 0.5)) == 0.0


def _weave(n=400):
    return np.where((np.arange(n) // 25) % 2 == 0, 1 / 13, -1 / 13)  # tight, fast weave: risky


def test_switching_one_term_off_zeroes_only_that_term():
    full = run_line(_weave(), params_from(load_config()), length=400.0, pitch_deg=25)[0]
    assert float(full.H_by_cause[0]) > 0 and float(full.H_by_cause[1]) > 0
    no_load = run_line(_weave(), params_from(risk_cfg(load=False)), length=400.0, pitch_deg=25)[0]
    assert float(no_load.H_by_cause[0]) == 0.0
    np.testing.assert_allclose(np.asarray(no_load.H_by_cause[1:]), np.asarray(full.H_by_cause[1:]), rtol=1e-5)


def test_everything_off_means_no_dnf():
    cfg = risk_cfg(g_max=False, v_stop=False, piste=False)
    cfg["risk"]["enabled"] = False
    summary = run_line(_weave(), params_from(cfg), length=400.0, pitch_deg=25)[0]
    assert float(summary.H) == 0.0 and float(summary.hard_dnf) == 0.0 and float(summary.wall) == 0.0


def test_expected_score_and_dnf_cost():
    course = C.to_course(C.easy())
    obj = ObjParams()
    summary = run_line(0.0, params_from(load_config()), length=300.0, pitch_deg=15.0)[0]
    risky = summary._replace(H=jnp.asarray(-math.log(0.9)))  # P_finish = 0.9
    e = float(expected_score(risky, course, obj))
    assert e == pytest.approx(0.9 * float(summary.T) + 0.1 * float(t_dnf(course, obj)), rel=1e-5)
    assert float(t_dnf(course, obj)) == pytest.approx(1.5 * 877 / (60 / 3.6), rel=1e-6)


def test_risk_map_typical_band_is_low_risk_and_tight_fast_turns_are_not():
    theta = math.radians(20)
    low, _ = hazard.turn_risk(20.0, 70 / 3.6, theta, R)
    high, _ = hazard.turn_risk(15.0, 80 / 3.6, theta, R)
    hard, _ = hazard.turn_risk(13.0, 90 / 3.6, theta, R)
    assert float(low) < 0.001
    assert float(high) > 0.01
    assert float(hard) == 1.0


def test_every_dnf_is_disqualified_and_graded():
    """Leaving the piste scores like a missed gate (DQ_BASE + excess), never like a finish, and
    going further off scores worse."""
    from skiopt.objectives import score

    course = C.to_course(C.constant_pitch_spec(300.0, 20.0))
    p = params_from(load_config())
    obj = ObjParams.from_config(load_config())
    scores = []
    for kappa in (0.004, 0.008):  # gentle drifts that leave the 40 m piste, the second by more
        summary = run_line(kappa, p, length=300.0, pitch_deg=20.0)[0]
        assert float(summary.hard_dnf) == 1.0
        scores.append(float(score(summary, course, obj).hard_score))
    assert scores[0] > obj.dq_base and scores[1] > scores[0]
