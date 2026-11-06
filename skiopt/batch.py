"""Batch runner: split experiments into single (method, seed) jobs and run several at once.

    python -m skiopt.batch configs/experiments/final_easy.yaml configs/experiments/gran_risa.yaml --jobs 4

Each job runs `python -m skiopt.run` on a one-method, one-seed copy of its experiment, so results
land in the usual results/<experiment>/<method>/seed_<n>/ folders and the usual skip-if-done applies:
re-running the batch only does what is missing or stale. Jobs start longest first (gradient methods,
then 3D before 2D), which shortens the total time. A log per job goes to results/_batch_logs/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

from skiopt.config import load_config
from skiopt.logger import run_dir, run_hash, run_status

ROOT = Path(__file__).resolve().parent.parent

# Rough relative cost per job, for longest-first ordering.
METHOD_COST = {"adam": 10, "hybrid": 8, "dp": 1, "cmaes": 2, "cmaes_smooth": 2, "ga": 2, "coach_line": 0.1,
               "fall_line": 0.1, "random_search": 1}


def job_cost(cfg: dict, method: dict) -> float:
    cost = METHOD_COST.get(method["name"], 2)
    if method["name"] == "adam" and method.get("restarts", 1) > 1:
        cost = 4  # batched restarts take far fewer sequential steps
    if method["name"] == "dp" and method.get("refine") == "adam":
        cost = 9
    if cfg["course"]["kind"] == "gran_risa":
        cost *= 3
    return cost


def plan(paths: list[str]):
    jobs = []
    for path in paths:
        raw = yaml.safe_load(Path(path).read_text())
        cfg = load_config(path)
        for method in raw["methods"]:
            for seed in raw["seeds"]:
                label = method.get("label", method["name"])
                status = run_status(run_dir(cfg["results_dir"], cfg["experiment"], label, seed),
                                    run_hash(cfg, method, seed))
                if status == "complete":
                    continue
                one = dict(raw, methods=[method], seeds=[seed])
                jobs.append((job_cost(cfg, method), f"{raw['experiment']}/{label}/seed_{seed}", one))
    jobs.sort(key=lambda j: -j[0])
    return jobs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="+")
    parser.add_argument("--jobs", type=int, default=4, help="jobs run at once")
    args = parser.parse_args(argv)

    jobs = plan(args.experiments)
    logs = ROOT / "results" / "_batch_logs"
    logs.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="skiopt_batch_"))
    print(f"{len(jobs)} jobs to run, {args.jobs} at a time", flush=True)

    running: list[tuple[subprocess.Popen, str, float]] = []
    queue = list(jobs)
    done = failed = 0
    while queue or running:
        while queue and len(running) < args.jobs:
            _, name, cfg = queue.pop(0)
            cfg_path = tmp / (name.replace("/", "__") + ".yaml")
            cfg_path.write_text(yaml.safe_dump(cfg))
            log = open(logs / (name.replace("/", "__") + ".log"), "w")
            proc = subprocess.Popen([sys.executable, "-m", "skiopt.run", str(cfg_path)], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT)
            running.append((proc, name, time.time()))
        time.sleep(5)
        for entry in list(running):
            proc, name, t0 = entry
            if proc.poll() is not None:
                running.remove(entry)
                ok = proc.returncode == 0
                done += ok
                failed += not ok
                print(f"{'done' if ok else 'FAILED'}  {name}  ({(time.time() - t0) / 60:.1f} min)  "
                      f"[{done + failed}/{len(jobs)}, {failed} failed]", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
