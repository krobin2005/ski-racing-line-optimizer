"""Analysis checks: SUMMARY.md and the report figures build from saved runs alone."""

import numpy as np
import yaml

from skiopt.analysis.figures import make_figures
from skiopt.analysis.summarize import a12, holm, summarize, turn_radii
from skiopt.run import run_experiment


def test_summary_and_figures_build_from_a_tiny_experiment(tmp_path):
    exp = {"experiment": "tiny", "course": {"kind": "library", "name": "easy"}, "policy": "spline",
           "methods": [{"name": "fall_line"}, {"name": "cmaes", "population": 16, "sigma": 0.4, "budget": 400}],
           "seeds": [0, 1]}
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(exp))
    run_experiment(path, results_dir=str(tmp_path / "results"))
    exp_dir = tmp_path / "results" / "tiny"

    text = summarize(exp_dir)
    assert "## Headline" in text and "cmaes" in text and "fall_line" in text
    assert (exp_dir / "SUMMARY.md").exists() and (exp_dir / "summary.csv").exists()

    written = make_figures(exp_dir, tmp_path / "figs")
    names = {p.name for p in written}
    assert {"convergence.png", "dv_plane.png", "turn_radii.png", "course_cmaes.png"} <= names
    assert all(p.stat().st_size > 5_000 for p in written)


def test_a12_and_holm():
    assert a12([1, 2], [3, 4]) == 1.0
    assert a12([3, 4], [1, 2]) == 0.0
    assert a12([1, 3], [1, 3]) == 0.5
    adjusted = holm([0.01, 0.04, 0.03])
    assert adjusted[0] == 0.03 and max(adjusted) <= 1.0


def test_average_turn_radius_of_a_constant_arc(tmp_path):
    """Two 20 m arcs of opposite sign: average and tightest radius are both 20 m."""
    k = np.r_[np.full(30, 1 / 20), np.full(30, -1 / 20)]
    rows = ["kappa,ds"] + [f"{x},1.0" for x in k]
    f = tmp_path / "telemetry.csv"
    f.write_text("\n".join(rows))
    np.testing.assert_allclose(turn_radii(f), [20, 20])
    np.testing.assert_allclose(turn_radii(f, kind="tightest"), [20, 20])
