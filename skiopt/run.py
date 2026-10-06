"""Experiment runner.

    python -m skiopt.run configs/experiments/smoke.yaml [--force]

Runs every (method, seed) pair in the experiment file. Runs already complete under the current
config are skipped unless --force is given; runs completed under a different config are re-run."""

from __future__ import annotations

import argparse
import sys

import jax

from skiopt import courses as course_lib
from skiopt.budget import Budget
from skiopt.config import load_config, load_equipment
from skiopt.logger import RunLogger, run_dir, run_hash, run_status
from skiopt.objectives import make_evaluator, make_trajectory, stack_courses
from skiopt.optim.baselines import METHODS
from skiopt.policies import make_policy
from skiopt.rollout import n_steps_for
from skiopt.types import SimParams


def run_experiment(experiment_path: str, force: bool = False, results_dir: str | None = None) -> list[dict]:
    cfg = load_config(experiment_path)
    if results_dir is not None:
        cfg["results_dir"] = results_dir
    equipment = load_equipment()
    p = SimParams.from_config(cfg)

    course = course_lib.from_config(cfg["course"])
    courses = stack_courses([course])
    n_steps = n_steps_for(float(course.length), p.dy)
    policy = make_policy(cfg["policy"], n_steps)
    evaluate = make_evaluator(policy, p, n_steps)
    trajectory = make_trajectory(policy, p, n_steps)

    report = []
    for method in cfg["methods"]:
        name = method["name"]
        if name not in METHODS:
            raise ValueError(f"unknown method {name!r}; known: {sorted(METHODS)}")
        for seed in cfg["seeds"]:
            path = run_dir(cfg["results_dir"], cfg["experiment"], name, seed)
            status = run_status(path, run_hash(cfg, method, seed))
            if status == "complete" and not force:
                print(f"skip  {name:16s} seed {seed}  (complete)")
                report.append({"method": name, "seed": seed, "skipped": True})
                continue
            if status == "stale":
                print(f"rerun {name:16s} seed {seed}  (config changed since it last ran)")
            budget = Budget(method.get("budget", 1), cfg["budget"]["grad_step_cost"])
            log = RunLogger(path, cfg, equipment, method, seed)
            best, score, history = METHODS[name](
                evaluate, courses, policy, method, budget, jax.random.key(seed), log
            )
            summary, traj = trajectory(best, course)
            log.champion(best, history)
            log.telemetry(traj)
            log.finish(best_score=score, T=summary.T, D=summary.D, v_mean=summary.v_mean,
                       max_G=summary.max_G, evals=budget.used)
            print(f"done  {name:16s} seed {seed}  T={float(summary.T):.3f} s  "
                  f"v_mean={float(summary.v_mean) * 3.6:.1f} km/h  evals={budget.used:.0f}  "
                  f"{budget.wall_clock:.1f} s")
            report.append({"method": name, "seed": seed, "skipped": False, "score": score})
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", help="path to an experiment YAML in configs/experiments/")
    parser.add_argument("--force", action="store_true", help="re-run runs that are already complete")
    args = parser.parse_args(argv)
    run_experiment(args.experiment, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
