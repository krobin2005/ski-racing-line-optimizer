"""Runner and logger checks (PLAN.md §13): a tiny run writes all four files, a second run is
skipped, and the same seed reproduces the same champion."""

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from skiopt.config import CONFIG_DIR
from skiopt.run import run_experiment

SMOKE = CONFIG_DIR / "experiments" / "smoke.yaml"
FILES = ("manifest.json", "progress.csv", "champion.npz", "telemetry.csv")


def test_tiny_run_writes_all_files_and_second_run_is_skipped(tmp_path):
    first = run_experiment(SMOKE, results_dir=str(tmp_path))
    assert not any(r["skipped"] for r in first)

    run = Path(tmp_path) / "smoke" / "random_search" / "seed_0"
    for name in FILES:
        assert (run / name).exists(), name

    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["equipment"]["ski_length_cm"] == 193
    assert manifest["config"]["sim"]["dy"] == 1.0
    with open(run / "telemetry.csv") as f:
        assert len(list(csv.reader(f))) == 1 + 300  # header + one row per metre

    second = run_experiment(SMOKE, results_dir=str(tmp_path))
    assert all(r["skipped"] for r in second)


def _experiment(tmp_path, name, **overrides):
    """Write a copy of the smoke experiment with top-level keys overridden; return its path."""
    cfg = yaml.safe_load(SMOKE.read_text())
    cfg.update(overrides)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_changed_physics_reruns_finished_runs(tmp_path):
    results = str(tmp_path / "results")
    run_experiment(_experiment(tmp_path, "before"), results_dir=results)
    changed = _experiment(tmp_path, "after", physics={"mu": 0.06})
    rerun = run_experiment(changed, results_dir=results)
    assert not any(r["skipped"] for r in rerun)
    assert all(r["skipped"] for r in run_experiment(changed, results_dir=results))


def test_adding_a_seed_keeps_existing_runs(tmp_path):
    results = str(tmp_path / "results")
    run_experiment(_experiment(tmp_path, "two_seeds", seeds=[0, 1]), results_dir=results)
    report = run_experiment(_experiment(tmp_path, "three_seeds", seeds=[0, 1, 2]), results_dir=results)
    assert {(r["method"], r["seed"]) for r in report if not r["skipped"]} == {
        ("fall_line", 2), ("random_search", 2)}


def test_same_seed_reproduces_champion(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run_experiment(SMOKE, results_dir=str(a))
    run_experiment(SMOKE, results_dir=str(b))
    rel = Path("smoke") / "random_search" / "seed_1" / "champion.npz"
    np.testing.assert_array_equal(np.load(a / rel)["params"], np.load(b / rel)["params"])
