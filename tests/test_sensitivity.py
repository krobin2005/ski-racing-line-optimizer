"""Sensitivity study plumbing: the override lands in the right config entry, nothing else changes."""

import yaml

from skiopt.analysis import sensitivity as S
from skiopt.config import CONFIG_DIR, load_config


def test_nested_and_lookup():
    assert S.nested("physics.mu", 0.05) == {"physics": {"mu": 0.05}}
    assert S.lookup({"a": {"b": 3}}, "a.b") == 3


def test_experiments_override_one_entry(tmp_path):
    base = CONFIG_DIR / "experiments" / "baseline.yaml"
    paths = S.experiments(base, "physics.mu", [0.02, 0.06], ["coach_line", "cmaes"], tmp_path)
    assert [p.stem for p in paths] == ["sens_mu_0.02", "sens_mu_0.06"]
    ref = load_config(base)
    for p, mu in zip(paths, (0.02, 0.06)):
        raw = yaml.safe_load(p.read_text())
        assert raw["seeds"] == [0]
        assert [m.get("label", m["name"]) for m in raw["methods"]] == ["coach_line", "cmaes"]
        cfg = load_config(p)
        assert cfg["physics"]["mu"] == mu
        assert cfg["physics"]["k_turn"] == ref["physics"]["k_turn"]   # siblings untouched
        assert cfg["course"] == ref["course"]
