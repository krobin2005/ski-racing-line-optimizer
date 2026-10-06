"""Evaluation speed check against the plan's target of < 50 ms per 1,000 candidates (PLAN.md §11).

    python -m skiopt.bench [--candidates 1000] [--length 1200]
"""

from __future__ import annotations

import argparse
import time

import jax

from skiopt import courses, policies
from skiopt.config import load_config
from skiopt.objectives import make_evaluator, stack_courses
from skiopt.rollout import n_steps_for
from skiopt.types import SimParams


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=1000)
    parser.add_argument("--length", type=float, default=1200.0, help="course length (m)")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)

    p = SimParams.from_config(load_config())
    n = n_steps_for(args.length, p.dy)
    policy = policies.OpenLoopKappa(n)
    evaluate = make_evaluator(policy, p, n)
    course = stack_courses([courses.constant_pitch(args.length, 20)])
    batch = 0.005 * jax.random.normal(jax.random.key(0), (args.candidates, n))

    jax.block_until_ready(evaluate(batch, course).hard_score)  # compile
    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(evaluate(batch, course).hard_score)
        times.append(time.perf_counter() - t0)
    times.sort()
    print(f"{args.candidates} candidates x {n} steps: median {times[len(times) // 2] * 1e3:.1f} ms, "
          f"best {times[0] * 1e3:.1f} ms  (target < 50 ms per 1,000)")


if __name__ == "__main__":
    main()
