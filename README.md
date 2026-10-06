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
pytest                                                   # run the tests (~4 s)
python -m skiopt.run configs/experiments/smoke.yaml      # pipeline smoke test; add --force to re-run
python -m skiopt.bench                                   # evaluation speed vs. the < 50 ms / 1,000 target
```

Results go to `results/<experiment>/<method>/seed_<n>/` (git-ignored): `manifest.json`, `progress.csv`, `champion.npz`, `telemetry.csv`. Finished runs are skipped on re-run. Figures and tables are rebuilt from saved results, never by re-running simulations.

### Code map (days 1–2)

| Module | What it does |
| --- | --- |
| `skiopt/types.py` | `SimParams`, `State`, `StepOutput`, `Course`, `Gates`, `Summary` (JAX pytrees) |
| `skiopt/terrain.py` | Terrain protocol + θ(y) pitch profile |
| `skiopt/dynamics.py` | 2D point mass: curvature clip, lateral load a_net, turning bands, speed update |
| `skiopt/rollout.py` | Fixed-length `lax.scan` rollout; padding steps after the finish are masked |
| `skiopt/objectives.py` | `evaluate(params_batch, courses)`; objectives are T until gates and DNF land |
| `skiopt/policies.py` | Open-loop curvature policy (spline, MLP, coach line to come) |
| `skiopt/optim/baselines.py` | `fall_line` and `random_search`, to exercise the pipeline |
| `skiopt/run.py`, `logger.py`, `budget.py`, `config.py` | Runner, result files, rollout-equivalent budget, YAML config |

## Status

| Step | Plan section | Status |
| --- | --- | --- |
| Setup: repo, environment, plan | — | Done |
| Days 1–2: core simulator + physics tests, evaluator, runner, logger | §8 | Done: 24 tests pass; 1,000 candidates × 1,200 m in ~10 ms |
| Days 3–4: FIS rules, course generator, course plot | §8 | Next |
| Days 5–6: hazard model, spline optimizers | §8 | |
| Day 6 gate: baseline runs end to end | §8 | |
| Days 7–8: coach line, deceptive course, hybrid | §8 | |
| Days 9–11: neuroevolution | §8 | |
| Days 12–14: 10-seed runs, figures, report | §8 | |
