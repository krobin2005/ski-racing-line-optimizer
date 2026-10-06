"""Run logging (PLAN.md §10). Each run writes, under results/<experiment>/<method>/seed_<n>/:

    manifest.json   config snapshot + hash, equipment, seed, git commit, versions, totals, `complete`
    progress.csv    one row per generation or gradient step
    champion.npz    best parameters + best-so-far history
    telemetry.csv   the champion line, one row per metre

A run whose manifest says complete is skipped, so re-running an experiment only fills gaps."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from skiopt.config import config_hash
from skiopt.types import StepOutput

TRACKED_PACKAGES = ["jax", "jaxlib", "evosax", "optax", "numpy", "scipy"]


def run_dir(results_dir: str | Path, experiment: str, method: str, seed: int) -> Path:
    return Path(results_dir) / experiment / method / f"seed_{seed}"


def is_complete(path: Path) -> bool:
    manifest = path / "manifest.json"
    if not manifest.exists():
        return False
    try:
        return bool(json.loads(manifest.read_text()).get("complete"))
    except json.JSONDecodeError:
        return False


def git_info(repo: Path) -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}
    except FileNotFoundError:
        return {"commit": None, "dirty": None}


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


class RunLogger:
    def __init__(self, path: Path, cfg: dict, equipment: dict, method: dict, seed: int):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._started = time.perf_counter()
        self._progress_file = open(self.path / "progress.csv", "w", newline="")
        self._progress = None
        self.manifest = {
            "experiment": cfg.get("experiment"),
            "method": method,
            "seed": seed,
            "config_hash": config_hash(cfg),
            "config": cfg,
            "equipment": equipment,
            "git": git_info(Path(__file__).resolve().parent.parent),
            "versions": versions(),
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "complete": False,
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        tmp = self.path / "manifest.json.tmp"
        tmp.write_text(json.dumps(self.manifest, indent=2, default=_jsonable))
        tmp.replace(self.path / "manifest.json")

    def progress(self, **row) -> None:
        row = {k: _jsonable(v) for k, v in row.items()}
        if self._progress is None:
            self._progress = csv.DictWriter(self._progress_file, fieldnames=list(row))
            self._progress.writeheader()
        self._progress.writerow(row)
        self._progress_file.flush()

    def champion(self, params, history) -> None:
        np.savez(self.path / "champion.npz", params=np.asarray(params), best_so_far=np.asarray(history))

    def telemetry(self, traj: StepOutput) -> None:
        tel = traj.telemetry
        active = np.asarray(traj.active) > 0
        cols = {
            "x": tel.x, "y": tel.y, "t": np.cumsum(np.asarray(traj.dt)) - np.asarray(traj.dt),
            "v": tel.v, "psi": tel.psi, "kappa": tel.kappa, "a_net": tel.a_net, "G": tel.G,
            "theta": tel.theta, "dt": traj.dt, "ds": traj.ds,
        }
        cols = {k: np.asarray(v)[active] for k, v in cols.items()}
        with open(self.path / "telemetry.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(zip(*cols.values()))

    def finish(self, **totals) -> None:
        self._progress_file.close()
        self.manifest.update({k: _jsonable(v) for k, v in totals.items()})
        self.manifest["wall_clock_s"] = round(time.perf_counter() - self._started, 3)
        self.manifest["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.manifest["complete"] = True
        self._write_manifest()


def _jsonable(v):
    if isinstance(v, (np.ndarray, np.generic)) or hasattr(v, "__array__"):
        arr = np.asarray(v)
        return arr.item() if arr.ndim == 0 else arr.tolist()
    return v
