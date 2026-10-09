"""Experiment runner.

    python -m skiopt.run configs/experiments/smoke.yaml [--force]

Runs every (method, seed) pair in the experiment file. Runs already complete under the current
config are skipped unless --force is given; runs completed under a different config are re-run."""

from __future__ import annotations

import argparse
import sys

import jax
import numpy as np

from skiopt import courses as course_lib
from skiopt.budget import Budget
from skiopt.config import load_config, load_equipment
from skiopt.hazard import TERM_NAMES
from skiopt.logger import RunLogger, run_dir, run_hash, run_status
from skiopt.objectives import score as score_run
from skiopt.objectives import ObjParams, make_evaluator, make_trajectory, make_value_and_grad, stack_courses
from skiopt.optim.common import Problem
from skiopt.optim.registry import METHODS
from skiopt.policies import make_policy
from skiopt.rollout import n_steps_for
from skiopt.types import SimParams


def run_experiment(experiment_path: str, force: bool = False, results_dir: str | None = None) -> list[dict]:
    cfg = load_config(experiment_path)
    if results_dir is not None:
        cfg["results_dir"] = results_dir
    equipment = load_equipment()
    p = SimParams.from_config(cfg)

    spec = course_lib.spec_from_config(cfg["course"])
    course = course_lib.to_course(spec)
    courses = stack_courses([course])
    n_steps = n_steps_for(float(course.length), p.dy)
    policy = make_policy(cfg["policy"], n_steps, length=spec.length, x_start=spec.x_start, dy=p.dy,
                         knot_spacing=cfg.get("spline", {}).get("knot_spacing_m", 10.0))
    obj = ObjParams.from_config(cfg)
    prob = Problem(evaluate=make_evaluator(policy, p, n_steps, obj),
                   value_and_grad=make_value_and_grad(policy, p, n_steps, obj),
                   courses=courses, policy=policy, obj=obj)
    trajectory = make_trajectory(policy, p, n_steps)

    report = []
    for method in cfg["methods"]:
        name = method["name"]
        if name not in METHODS:
            raise ValueError(f"unknown method {name!r}; known: {sorted(METHODS)}")
        label = method.get("label", name)  # distinguishes variants of one method, e.g. adam_restarts
        for seed in cfg["seeds"]:
            path = run_dir(cfg["results_dir"], cfg["experiment"], label, seed)
            status = run_status(path, run_hash(cfg, method, seed))
            if status == "complete" and not force:
                print(f"skip  {label:16s} seed {seed}  (complete)")
                report.append({"method": label, "seed": seed, "skipped": True})
                continue
            if status == "stale":
                print(f"rerun {label:16s} seed {seed}  (config changed since it last ran)")
            budget = Budget(method.get("budget", cfg["budget"]["spline_rollouts"]), cfg["budget"]["grad_step_cost"])
            log = RunLogger(path, cfg, equipment, method, seed)
            best, score, history = METHODS[name](prob, method, budget, jax.random.key(seed), log)
            summary, traj = trajectory(best, course)
            log.champion(best, history)
            log.telemetry(traj)
            gates_missed = int(score_run(summary, course, obj).n_missed)
            p_dnf = float(-np.expm1(-float(summary.H)))
            log.finish(best_score=score, T=summary.T, D=summary.D, v_mean=summary.v_mean,
                       max_G=summary.max_G, gates_missed=gates_missed, P_DNF=p_dnf,
                       H_by_cause=dict(zip(TERM_NAMES, np.asarray(summary.H_by_cause).tolist())),
                       hard_dnf=bool(summary.hard_dnf > 0), evals=budget.used)
            print(f"done  {label:16s} seed {seed}  T={float(summary.T):.3f} s  "
                  f"v_mean={float(summary.v_mean) * 3.6:.1f} km/h  missed={gates_missed}  P_DNF={100 * p_dnf:.1f}%  evals={budget.used:.0f}  "
                  f"{budget.wall_clock:.1f} s")
            report.append({"method": label, "seed": seed, "skipped": False, "score": score})
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
