# Ski Racing-Line Optimizer

Finding the fastest line down a Giant Slalom course without crashing, and comparing how different optimizers get there. A 2D point-mass skier runs through FIS-legal GS gates; gradient descent through a differentiable JAX simulator, evolutionary methods (GA, CMA-ES), a CMA-ES → Adam hybrid and a dynamic-programming lattice search compete on equal budgets, in 2D and in 3D on the real Gran Risa (Alta Badia) terrain. The objective is expected race time: course time weighted by the probability of finishing.

Evolutionary Computing independent project, DCS 340, Bates College. Kyle and Liam.

**The full implementation plan is in [docs/PLAN.md](docs/PLAN.md); the plan to reach 3D on the real Gran Risa by November 30 is in [docs/ROADMAP_3D.md](docs/ROADMAP_3D.md).** It is the official copy; change the plan there, in a commit.

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
pytest                                                   # run the tests (97 tests, ~4 min)
python -m skiopt.run configs/experiments/smoke.yaml      # pipeline smoke test; add --force to re-run
python -m skiopt.bench                                   # evaluation speed vs. the < 50 ms / 1,000 target
python -m skiopt.analysis.plots --generated 7 --out figures/gen_7.png   # plot a course (or --library easy, --run <dir>)
python -m skiopt.analysis.risk_map --out figures/risk_map.png            # tune configs/risk.yaml
python -m skiopt.run configs/experiments/baseline.yaml                  # day-6 gate: every spline method on the easy course
python -m skiopt.analysis.summarize results/baseline                    # tables + race reports -> results/baseline/SUMMARY.md
python -m skiopt.batch configs/experiments/final_easy.yaml --jobs 4     # many experiments: one job per (method, seed), N at a time
python -m skiopt.analysis.gran_risa_figures results/gran_risa --out figures/gran_risa   # 3D: line on the map, run trace, splits
python -m skiopt.analysis.report_all                                  # every final report -> docs/reports/final_*
python -m skiopt.analysis.sensitivity configs/experiments/gran_risa.yaml physics.mu 0.02 0.04 0.06 --out docs/reports/sensitivity_mu   # re-optimize at each value
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
| `skiopt/policies.py` | Open-loop curvature, the spline line, and the apex-and-arc line (main; its default is the coach line). MLP to come |
| `skiopt/optim/` | `adam` (+ restarts), `cmaes`, `cmaes_smooth`, `ga`, `hybrid`, plus `fall_line` / `random_search` baselines; `registry.py` maps names to methods |
| `skiopt/analysis/risk_map.py` | DNF risk of one turn over radius × speed, for tuning `risk.yaml` |
| `skiopt/analysis/summarize.py` | `SUMMARY.md` + `summary.csv` from saved runs: headline table, pairwise tests, race reports, warnings |
| `skiopt/run.py`, `logger.py`, `budget.py`, `config.py` | Runner, result files, rollout-equivalent budget, YAML config |

## Status

| Step | Status |
| --- | --- |
| Setup, plan v2.1 | Done |
| Days 1–8 (2D): simulator, FIS courses, DNF model, optimizers, trap course | Done |
| 2D results, easy course, 3 seeds | Done: [arc line (main)](docs/reports/2d_easy_arc/SUMMARY.md), [spline (contrast)](docs/reports/2d_easy_spline/SUMMARY.md). All methods within 1% on the arc line; CMA-ES reaches it ~28x faster than Adam (10-seed final: ~20x). The spline line costs every method 17-30 s. |
| 3D (docs/ROADMAP_3D.md): Gran Risa terrain pipeline | Done: [terrain figures](docs/reports/gran_risa_terrain) |
| 3D simulator (camber, fall-away, rolls, compressions, edge angle) | Done, tested against analytic cases |
| Gran Risa course set (52 gates / 51 turning, as in the 21 Dec 2025 race, run 1) | Done; 3D carving loss calibrated so the DP-optimal line matches the real 76.00 s |
| DP global baseline ("discretize and search") | Done |
| Final runs: 10 seeds, every method, on the easy course, the trap course and the Gran Risa (220 jobs via `python -m skiopt.batch`) | **Done** (Nov 6, 220/220 clean): [easy](docs/reports/final_easy/SUMMARY.md), [trap](docs/reports/final_trap/SUMMARY.md), [Gran Risa](docs/reports/final_gran_risa/SUMMARY.md). CMA-ES on the hard score is best or tied-best everywhere; on the Gran Risa it runs 74.5 s (real winner 76.00 s) and beats the DP lattice (75.9 s). Write-up: [docs/REPORT.md](docs/REPORT.md) §5–6. |
| Sensitivity: snow friction, 3D carving loss | Done: [μ](docs/reports/sensitivity_mu/SENSITIVITY.md), [k_turn](docs/reports/sensitivity_k_turn/SENSITIVITY.md). About 1.3 s per 0.01 of friction. |
| Ablation: sharp (unbiased) gate penalty for the gradient methods | Sharp penalty done (REPORT.md §6: removes the bias, Adam 1.6–1.9 s faster on the Gran Risa but clips gates by mm–cm); 5 cm margin done: all runs clean, Adam's gap to CMA-ES halves on the Gran Risa and DP → Adam beats DP (75.57 vs 75.94 s); with the margin Adam ties CMA-ES on the trap course (52.07 vs 52.05); with a sharp 3D edge-angle wall too, Adam reaches 74.88 s on the Gran Risa (CMA-ES 74.80) — REPORT.md §6 |
| Neural controller | Stretch (cut first, see the roadmap) |
