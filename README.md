# Ski Racing-Line Optimizer

Finding the fastest line down a Giant Slalom course without crashing, and comparing how different optimizers get there. A 2D point-mass skier runs through FIS-legal GS gates; gradient descent through a differentiable JAX simulator, evolutionary methods (GA, CMA-ES) and neuroevolution of a neural-network controller compete on equal budgets. The objective is expected race time: course time weighted by the probability of finishing.

Evolutionary Computing independent project, DCS 340, Bates College. Kyle and Liam.

**The full implementation plan is in [docs/PLAN.md](docs/PLAN.md).** It is the official copy; change the plan there, in a commit.

## Setup (Mac)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt   # exact versions of everything, incl. transitive deps
```

`requirements.txt` lists the direct dependencies; `requirements-lock.txt` is a full `pip freeze` of a clean install of it. After changing `requirements.txt`, regenerate the lock file in a fresh venv with `pip install -r requirements.txt && pip freeze > requirements-lock.txt`.

Tested with Python 3.13 and 3.14 (JAX 0.11.2, evosax 0.3.1). Every pinned package supports Python 3.12 or newer.

## Running

```bash
pytest                                                   # run the tests (~40 s)
python -m skiopt.run configs/experiments/smoke.yaml      # pipeline smoke test; add --force to re-run
python -m skiopt.bench                                   # evaluation speed vs. the < 50 ms / 1,000 target
python -m skiopt.analysis.plots --generated 7 --out figures/gen_7.png   # plot a course (or --library easy, --run <dir>)
python -m skiopt.analysis.risk_map --out figures/risk_map.png            # tune configs/risk.yaml
python -m skiopt.run configs/experiments/baseline.yaml                  # day-6 gate: every spline method on the easy course
python -m skiopt.analysis.summarize results/baseline                    # tables + race reports -> results/baseline/SUMMARY.md
```

Results go to `results/<experiment>/<method>/seed_<n>/` (git-ignored): `manifest.json`, `progress.csv`, `champion.npz`, `telemetry.csv`. Finished runs are skipped on re-run. Figures and tables are rebuilt from saved results, never by re-running simulations.

### Code map

| Module | What it does |
| --- | --- |
| `skiopt/types.py` | `SimParams`, `State`, `StepOutput`, `Course`, `Gates`, `Summary` (JAX pytrees) |
| `skiopt/terrain.py` | Terrain protocol + θ(y) pitch profile |
| `skiopt/dynamics.py` | 2D point mass: curvature clip, lateral load a_net, turning bands, speed update |
| `skiopt/rollout.py` | Fixed-length `lax.scan` rollout; padding steps after the finish are masked |
| `skiopt/courses.py` | `CourseSpec`, FIS validator, course generator, `easy` library course, padding to stack courses |
| `skiopt/objectives.py` | `evaluate` / `value_and_grad`; E[score] = P_finish·T + P_DNF·T_DNF; every DNF (missed gate, piste, G_max, v_stop) is a graded DQ in the hard score; softplus penalties + walls in the smooth loss |
| `skiopt/analysis/plots.py` | Course plot: gates, piste, start/finish, line coloured by speed, pitch profile |
| `skiopt/hazard.py`, `configs/risk.yaml` | DNF model: 4 hazard terms + hard limits, each switchable and tunable |
| `skiopt/policies.py` | Open-loop curvature and the spline line (MLP and coach line to come) |
| `skiopt/optim/` | `adam` (+ restarts), `cmaes`, `cmaes_smooth`, `ga`, `hybrid`, plus `fall_line` / `random_search` baselines; `registry.py` maps names to methods |
| `skiopt/analysis/risk_map.py` | DNF risk of one turn over radius × speed, for tuning `risk.yaml` |
| `skiopt/analysis/summarize.py` | `SUMMARY.md` + `summary.csv` from saved runs: headline table, pairwise tests, race reports, warnings |
| `skiopt/run.py`, `logger.py`, `budget.py`, `config.py` | Runner, result files, rollout-equivalent budget, YAML config |

## Status

| Step | Plan section | Status |
| --- | --- | --- |
| Setup: repo, environment, plan | — | Done |
| Days 1–2: core simulator + physics tests, evaluator, runner, logger | §8 | Done: 24 tests pass; 1,000 candidates × 1,200 m in ~10 ms |
| Days 3–4: FIS rules, course generator, course plot | §8 | Done: 1,000 generated courses validate; 43 tests pass |
| Days 5–6: hazard model, spline optimizers | §8 | Done: switchable DNF model, risk map, Adam / CMA-ES (2 arms) / GA / hybrid, summary + figures |
| Day 6 gate: baseline runs end to end | §8 | Passed with one open issue: 5 of 6 methods finish cleanly; 57–60 s, 57–59 km/h, ~7% DNF. Turn shape too peaky, see [docs/reports/day6_baseline](docs/reports/day6_baseline/SUMMARY.md) |
| Days 7–8: coach line, deceptive course, hybrid | §8 | |
| Days 9–11: neuroevolution | §8 | |
| Days 12–14: 10-seed runs, figures, report | §8 | |
