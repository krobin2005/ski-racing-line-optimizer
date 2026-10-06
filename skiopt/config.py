"""Config loading: configs/base.yaml, deep-merged with an experiment file's overrides."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(experiment_path: str | Path | None = None, config_dir: Path = CONFIG_DIR) -> dict:
    cfg = load_yaml(config_dir / "base.yaml")
    if experiment_path is not None:
        cfg = deep_merge(cfg, load_yaml(experiment_path))
    return cfg


def load_equipment(config_dir: Path = CONFIG_DIR) -> dict:
    return load_yaml(config_dir / "equipment.yaml")["equipment"]


def config_hash(cfg: dict) -> str:
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
