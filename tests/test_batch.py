"""Batch runner checks: jobs split per (method, seed), finished runs skipped, longest first."""

import yaml

from skiopt.batch import job_cost, plan
from skiopt.config import CONFIG_DIR
from skiopt.run import run_experiment


def test_plan_splits_orders_and_skips(tmp_path):
    exp = {"experiment": "b", "course": {"kind": "constant_pitch", "length_m": 120, "pitch_deg": 20},
           "policy": "open_loop_kappa", "results_dir": str(tmp_path / "r"),
           "methods": [{"name": "fall_line"}, {"name": "random_search", "population": 8, "sigma": 0.001, "budget": 32}],
           "seeds": [0, 1]}
    path = tmp_path / "b.yaml"
    path.write_text(yaml.safe_dump(exp))
    jobs = plan([str(path)])
    assert len(jobs) == 4
    assert [j[2]["methods"][0]["name"] for j in jobs][:2] == ["random_search"] * 2  # longest first
    assert all(len(j[2]["seeds"]) == 1 and len(j[2]["methods"]) == 1 for j in jobs)

    run_experiment(path)  # finish everything; the plan is then empty
    assert plan([str(path)]) == []


def test_3d_and_gradient_jobs_cost_more():
    flat = {"course": {"kind": "library"}}
    gr = {"course": {"kind": "gran_risa"}}
    assert job_cost(gr, {"name": "adam"}) > job_cost(flat, {"name": "adam"}) > job_cost(flat, {"name": "cmaes"})
    assert job_cost(flat, {"name": "adam", "restarts": 4}) < job_cost(flat, {"name": "adam"})
    assert CONFIG_DIR.exists()
