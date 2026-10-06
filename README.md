# Ski Racing-Line Optimizer

Finding the fastest line down a Giant Slalom course without crashing, and comparing how different optimizers get there. A 2D point-mass skier runs through FIS-legal GS gates; gradient descent through a differentiable JAX simulator, evolutionary methods (GA, CMA-ES) and neuroevolution of a neural-network controller compete on equal budgets. The objective is expected race time: course time weighted by the probability of finishing.

Evolutionary Computing independent project, DCS 340, Bates College. Kyle and Liam.

**The full implementation plan is in [docs/PLAN.md](docs/PLAN.md).** It is the official copy; change the plan there, in a commit.

## Setup (Mac)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.14 and the versions pinned in `requirements.txt` (JAX 0.11.2, evosax 0.3.1).

## Running

Once code lands:

```bash
pytest                                                        # run the tests
python -m skiopt.run configs/experiments/baseline.yaml        # run an experiment
```

Results go to `results/` (git-ignored). Figures and tables are rebuilt from saved results, never by re-running simulations.

## Status

| Step | Plan section | Status |
| --- | --- | --- |
| Setup: repo, environment, plan | — | Done |
| Days 1–2: core simulator + physics tests, evaluator, runner, logger | §8 | Next |
| Days 3–4: FIS rules, course generator, course plot | §8 | |
| Days 5–6: hazard model, spline optimizers | §8 | |
| Day 6 gate: baseline runs end to end | §8 | |
| Days 7–8: coach line, deceptive course, hybrid | §8 | |
| Days 9–11: neuroevolution | §8 | |
| Days 12–14: 10-seed runs, figures, report | §8 | |
