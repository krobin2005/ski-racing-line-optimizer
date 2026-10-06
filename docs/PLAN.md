# Ski Racing-Line Optimizer: Implementation Plan v2.1

**Status:** v2 approved by Kyle and Liam, October 6, 2026; v2.1 revisions requested by Liam the same day (see [What changed in v2.1](#what-changed-in-v21)). This file is the official copy of the plan. The Google Doc version is frozen; change the plan here, in a commit.

**Working arrangement:** for now we work through tracks A and B together rather than splitting them. The track labels below still show which parts depend on each other.

## Contents

0. [What changed from v1](#what-changed-from-v1) and [in v2.1](#what-changed-in-v21)
1. [Context and project decisions](#1-context-and-project-decisions)
2. [FIS rules and equipment](#2-fis-rules-and-equipment)
3. [The optimization problem](#3-the-optimization-problem)
4. [Simulator](#4-simulator)
5. [DNF model, gates and objectives](#5-dnf-model-gates-and-objectives)
6. [Policies](#6-policies)
7. [Project structure](#7-project-structure)
8. [Two-week schedule](#8-two-week-schedule)
9. [Fair comparison and metrics](#9-fair-comparison-and-metrics)
10. [Logging and summary](#10-logging-and-summary)
11. [Risks and mitigations](#11-risks-and-mitigations)
12. [Design for the 3D extension](#12-design-for-the-3d-extension)
13. [Tests and verification](#13-tests-and-verification)

## What changed from v1

v2 fits the project into about two weeks. It fixes v1's internal contradictions and scores a single run instead of two. The core idea, minimizing T = D / v̄ with explicit DNF risk across gradient, evolutionary and hybrid methods, is unchanged.

| Area | v1 | v2 |
| --- | --- | --- |
| Timeline | ≈3.5 weeks, one sequential track | 14 days, two tracks with a fixed cut order |
| Runs | Two runs core (ICR 906.1) | One run only; the two-run rule is noted, not modelled |
| Equipment | Frozen dataclass, hash asserts, override rejection, `validate_fis()` code | One read-only config block; FIS legality checked once by hand (table in §2) |
| Boot lifter | Counted in both the 49 mm stack and boot-sole thickness | Counted once, in the 49 mm stack (it sits between plate and binding) |
| Turns under 12 m | "Instant DNF" and also "clipped, skier drifts" | Clipped for both policies, never a DNF on its own |
| Lateral load | κv² only | Net load κv² + g·sinθ·sinψ, so load peaks after the fall line |
| G-force claim | "1.9 g, comfortably below 2.5 g" | Corrected to total G (2.1–2.9 g in the typical band); G_safe raised to 3.0 g |
| Hazard terms | 6 terms + course multiplier m_course | 4 terms; m_course and h_fast moved to stretch |
| Pole pass | Point mass may touch the turning pole | 0.4 m body clearance from the turning pole |
| Direction changes | Validator counted gates; delay gate wording garbled | Validator counts side switches; a delay gate adds none |
| Physical constants | "Stylized" mass, drag, friction | Literature values + a realism check on run time and mean speed |
| 3D carving | R = R_sc·cos(edge) and κ = sin(edge)/R_sc both used | R = R_sc·cos(edge) everywhere |
| Fairness | Evolution on smooth loss was an ablation | CMA-ES on the smooth loss is a core arm |
| MLP optimizer | Full CMA-ES on ~560 weights | sep-CMA-ES and OpenAI-ES; full CMA-ES only on splines |
| Logging | Parquet, checkpoints, resume, live watcher, LaTeX, presentation mode | JSON + CSV + NPZ, skip-if-done, one report script |
| Extras | Monte Carlo, risk sweep, difficulty sweep, robustness, ablations core | Moved to stretch; coach-line baseline added to core |

### What changed in v2.1

| Area | v2 | v2.1 |
| --- | --- | --- |
| Boot lifter and stack height | Counted in the bearing-surface height; used in 3D | Omitted from the model. A 2D point mass has no edge angle or boot-out, so neither can change turn radius. Stack height is kept as a recorded fact only (§2). |
| DNF limits | Fixed hazard terms and hard limits | Every hazard term and every hard limit has its own on/off switch and tunable constants in `configs/risk.yaml` (§5) |
| Tuning risk | Calibrate c_* once | A `risk_map.py` script plots DNF risk for a single turn across radius × speed, so we can see exactly what the current settings do before running anything (§5) |
| Hard limits in the smooth objective | Only the piste edge had a smooth wall | Every enabled hard limit (G_max, v_stop, piste) gets a steep softplus wall, so gradient methods see it coming |
| Run-time estimate | "A spline run takes about 10 s" | 10 s holds for population methods only; Adam's sequential steps take minutes (§10) |
| Milestone labels | Leftover M0/M2 references | Replaced with schedule days |
| Dependencies | Direct packages pinned | Plus `requirements-lock.txt`, a full `pip freeze`, so transitive packages (flax etc.) are pinned too |

## 1. Context and project decisions

A point-mass skier runs down a slope through a sequence of Giant Slalom gates, and the goal is to minimize course time without crashing. This is an Evolutionary Computing independent project. It compares gradient-based optimization through a differentiable simulator, evolutionary methods (GA, CMA-ES, and neuroevolution of a controller), and a hybrid of the two. This 2D model is stage 1, and its interfaces should let 3D terrain and edge or lean controls replace the 2D versions later.

- **Timeline:** 14 working days, two people (§8). NEAT is stretch-only.
- **Hardware:** Mac laptops, VS Code, `jax[cpu]`.
- **Physics:** stylized, with constants taken from published values where they exist.
- **Neural controller:** trained on randomized FIS-valid courses.
- **Courses:** follow the FIS GS course-setting rules that can be expressed in 2D, plotted in matplotlib.
- **Discipline:** GS only. The discipline still lives in a YAML file, so another discipline later is a new file, not a refactor.
- **Runs:** every result is a single run on one course. Real GS is decided over two runs (ICR 906.1), but a second run adds nothing to the optimizer comparison.
- **Efficiency rule:** if a feature doesn't change a result in the report, it isn't built in the first 14 days.

## 2. FIS rules and equipment

The generator and validator enforce the GS course-setting rules that can be expressed in 2D. Values come from ICR Book IV, July 2025 edition. One five-minute task is to open the July 2026 edition by hand and confirm articles 901–903 are unchanged. Terrain inspection, safety nets and snow preparation are out of scope for a 2D model.

| Rule | Value | ICR article |
| --- | --- | --- |
| Vertical drop, men | 250–450 m | 901.1.1 |
| Vertical drop, women | 250–400 m | 901.1.2 |
| Gate | 2 turning + 2 outside poles with panels, alternating red and blue | 901.2.1–901.2.2 |
| Gate width | 4–8 m | 901.2.3 |
| Nearest turning poles of successive gates | at least 10 m apart | 901.2.3 |
| Direction changes | 11–15% of vertical drop in metres, rounded | 901.2.4 |
| Course width | about 40 m | 902.1 |
| Character of the set | mainly single gates; a mix of long, medium and short turns | 903.1.2–903.1.3 |
| Correct passage | both tips and both feet cross the gate line | 661.4.1 |

How these enter the model:

- **Piste:** a 40 m corridor. Leaving it is a DNF.
- **Direction changes:** each gate has a side (pass left or right). The validator counts side switches between consecutive gates, not gates. A **delay gate** is a gate passed on the same side as the previous one: it adds a gate but no direction change. Red/blue colours still alternate on every gate.
- **Variety:** the validator checks the spread of gate distances, so the generator can't produce a uniform rhythm.
- **Gate pass:** x(y_g) must lie between the turning pole plus a 0.4 m body clearance and the outside pole (§5).
- **Grid:** gate y-positions are rounded to whole multiples of Δy, so the gate check is exact.

### Equipment (fixed, recorded once)

The setup is a constant, identical in every run, stored in one read-only block in `configs/equipment.yaml` and copied into each run's `manifest.json`. No optimizer or experiment config touches it, so v1's hash, assert and override-rejection code is dropped.

| Item | Value | Used in 2D? |
| --- | --- | --- |
| Ski | Völkl Racetiger GS, 193 cm, 30 m sidecut | Sidecut sets the turning bands (§4) |
| Plate + binding | Piston plate, 30 DIN | Recorded only |
| Stack height | 49 mm heel, 47 mm toe | Recorded only |
| Boot | Lange ZB, 25.5 | Recorded only |

FIS legality, checked once by hand against the 2024/25 equipment specifications rather than in code:

| Spec | FIS limit | Setup | Result |
| --- | --- | --- | --- |
| GS ski length, men, COC level | ≥ 193 cm | 193 cm | Legal, at the limit |
| GS sidecut radius, men | ≥ 30 m | 30 m | Legal, at the limit |
| Bearing surface height h_BS (ski base to boot sole) | ≤ 50 mm | 49 mm | Legal, 1 mm margin |

**Boot lifter: omitted (v2.1).** Lifters and stack height change edge angle, leverage and the point where the boot touches the snow. A 2D point mass has none of those. Its turn radius comes only from the 30 m sidecut through the turning bands (§4), so the lifter cannot change any result. It is left out of the model and the FIS check rather than modelled half-way. If the 3D stage adds edge angle and boot-out, measure the real stack and lifter then.

## 3. The optimization problem

Find the line with the highest average speed over the shortest distance, without going down. Course time is exactly distance over average speed:

```
T = D / v̄,   D = Σ Δs
```

The shortest line goes straight at each gate but needs turns tighter than 18 m, which cost speed. Round 18–22 m arcs keep speed but travel further. The optimum gives up a little distance to keep speed.

- **One objective.** Every optimizer minimizes the expected race score (§5). D and v̄ are never combined as a weighted sum, because minimizing T already balances them.
- **Always reported.** Every rollout returns T, D, v̄ and P_DNF, so each result can be explained physically: "method X travelled 3 m further but carried 1.2 km/h more average speed."
- **Signature figure.** The D–v̄ plane: each method's final line is a point, coloured by P_DNF, over iso-time curves T = D / v̄. It shows directly whether a method lost time through distance or through speed. On the deceptive course its two basins should appear as two clusters.

## 4. Simulator

The simulator integrates over a fixed spatial step Δy = 1 m down the fall line, so every rollout has the same length and fits `jax.lax.scan`, `jit` and `vmap` with no masking. The view is top-down in the slope plane: x runs across the slope, y runs down the fall line, and a pitch profile θ(y) supplies the steepness. Heading ψ is measured from the fall line and clipped smoothly to |ψ| < 80°, so y always increases.

- **Start:** x_start, ψ₀ = 0, v₀ ≈ 5 m/s.
- **Piste:** 40 m wide. Leaving it is a DNF (a softplus wall in the smooth objective, an exact DNF in the hard score).
- **State:** (x, ψ, v). **Control:** κ_cmd, the commanded curvature.

Per step:

```
κ      = clip(κ_cmd, ±1/r_floor)                     never tighter than 12 m, for every policy
Δs     = Δy / cos ψ
x'     = x + tan ψ · Δy
ψ'     = ψ + κ · Δs
a_net  = κ v² + g sinθ · sinψ                      lateral load the snow must supply (signed)
tight  = softplus_β(|κ| − 1/r_typical_lo) · v²      tighter than 18 m: rising speed cost
skid   = softplus_β(|κ| − 1/r_skid) · v²            tighter than 14 m: heavy skid loss
v·dv/ds = g sinθ cosψ − μ g cosθ − c_d v² − k_turn|a_net| − k_tight·tight − k_skid·skid
t     += Δs / v
```

The a_net term is new in v2. Gravity's sideways component, g·sinθ·sinψ, lowers the edge load before the fall line and raises it after. At 25° pitch and ψ = 50° that is about 3.2 m/s², roughly 15% of the load. Including it moves peak load and risk to the second half of the turn, which matches measured GS turns.

### Turning bands

The 30 m sidecut links radius to edge angle through R = R_sc·cos(edge). All four thresholds live in the `skier` block of the config and can be tuned.

| Radius | Implied edge angle | Behaviour | Cost |
| --- | --- | --- | --- |
| ≥ 18 m (typical 18–22 m) | 43–53° | Clean carve | Carving term only |
| 14–18 m | 53–62° | Carving at the edge of grip | Speed cost rising as radius shrinks |
| 12–14 m | 62–66° | Heavy skid | Large skid loss + h_tight risk |
| < 12 m | > 66° | Unreachable | Clipped to 12 m for every policy |

A command tighter than 12 m is **clipped, not a DNF**. That removes v1's contradiction and treats both policies alike: the MLP can't command it anyway, and a spline that asks for it drifts off its line and may miss the next gate. A missed gate is a DNF.

### Physical constants

Mass, drag and friction use published values instead of made-up ones. Every number below is approximate and lives in `base.yaml`.

| Constant | Value |
| --- | --- |
| Skier mass m | 85 kg |
| Drag area CdA (GS turning posture) | 0.3 m² |
| Air density ρ (at altitude) | 1.0 kg/m³ |
| c_d = ½ρCdA / m | ≈ 0.0018 1/m |
| Ski–snow friction μ | 0.04 |

Realism check, run once at the day-6 gate: on a standard generated course, a sensible line should take about 60–80 s at a mean speed of about 55–70 km/h. If it doesn't, tune k_turn before anything else.

Each physics and risk term has an on/off switch (`physics.terms.drag`, `risk.enabled`, …) for the analytic tests.

## 5. DNF model, gates and objectives

DNF risk is a hazard rate per metre that accumulates along the run. It stays deterministic and differentiable: P_DNF is computed exactly from the trajectory, with no sampling. v2 cuts the hazard terms from six to four, because grip and G-force are both driven by the same lateral load, and the rest added constants that no data could pin down.

```
G       = √(a_net² + (g cosθ)²) / g                   total load in g
h_load  = c_load  · softplus_β(G − G_safe)               edge washout / body can't hold the load
h_tight = c_tight · softplus_β(|κ| − 1/r_skid)           radius under ~14 m: edge catch
h_late  = c_late  · softplus_β(κ_req − 1/r_skid)         line too round → next gate needs too tight a turn
h_slow  = c_slow  · softplus_β(v_slow − v)               too slow: no edge pressure
P_finish = exp(−Σ (h_load + h_tight + h_late + h_slow) · Δs)
Hard DNF: G > G_max, v < v_stop, a missed gate, or leaving the piste
```

Every term and hard limit above can be switched off and retuned in `configs/risk.yaml` (see "Switches and tuning" below).

- **G limits.** World Cup GS measurements (Gilgien et al.) show average ground reaction forces above 1.5× body weight and peaks up to about 3.8×. v2 sets **G_safe = 3.0 g** and **G_max = 4.0 g**.
- **Why G_safe went up.** v1 claimed 70 km/h on a 20 m arc is "1.9 g, comfortably below 2.5 g". That is lateral load only. Total G on a 20° pitch is 2.1 g at the fall line and 2.4 g after it. At 80 km/h it is 2.7–2.9 g, already over v1's 2.5 g limit, so the typical turning band was itself a crash risk. With 3.0 g, the 18–22 m band is low-risk across the 60–80 km/h range, and risk rises for tight turns at top speed (80 km/h on 18 m after the fall line ≈ 3.2 g).
- **Late line.** κ_req is the curvature of the arc tangent to the current heading that reaches the nearest point of the next gate's pass window: κ_req = 2·e⊥ / d². It is closed-form and differentiable.
- **Too slow.** v_slow ≈ 35 km/h; below v_stop the run is a DNF.
- **Numerics.** Σh·Δs is accumulated in log space, and P_DNF is computed with `-expm1` so small risks don't vanish in float32.
- **Calibration target.** A sensible line on a standard course: a few percent DNF. An aggressive line: roughly 10–30%. These are starting targets; expect to tune (below).

### Switches and tuning (v2.1)

The DNF limits will need fine-tuning, so none of them are hard-coded. Every hazard term and every hard limit lives in `configs/risk.yaml`, has its own `enabled` switch, and exposes the constants that set how risky a turn of a given radius and speed is. Experiment configs may override any of these values, and the values used are copied into each run's `manifest.json`, so every result records the risk settings it was produced with.

```yaml
risk:
  enabled: true              # master switch: false → P_DNF = 0 from hazards (hard limits below still apply if enabled)
  beta: 4.0                  # softplus sharpness for all hazard terms (higher = closer to a hard threshold)
  terms:
    load:  {enabled: true, c: 0.002, G_safe: 3.0}     # risk per metre per g above G_safe
    tight: {enabled: true, c: 0.002, r_skid: 14.0}    # risk per metre per (1/m) of curvature above 1/r_skid
    late:  {enabled: true, c: 0.002}                  # risk per metre per (1/m) of κ_req above 1/r_skid
    slow:  {enabled: true, c: 0.001, v_slow_kmh: 35}  # risk per metre per m/s below v_slow
  hard:                      # exact DNF in the hard score; steep softplus wall in the smooth objective
    g_max:     {enabled: true, value: 4.0}
    v_stop:    {enabled: true, value_kmh: 10}
    piste:     {enabled: true, width_m: 40}
    gate_miss: {enabled: true}                        # disable only for the analytic tests
  wall: {weight: 100.0, beta: 20.0}                   # smooth-objective walls for enabled hard limits
```

The `c` values above are placeholders, not calibrated numbers.

How the knobs map to "how risky is this turn":

| To change… | Turn this knob |
| --- | --- |
| Where fast, tight turns start to get risky | `load.G_safe`; G combines speed and radius through a_net = κv² + g sinθ sinψ |
| How quickly that risk grows past the threshold | `load.c`, and `beta` for how sharp the onset is |
| Risk from tight radius regardless of speed | `tight.r_skid` and `tight.c` |
| Risk from a late, too-round line | `late.c` |
| Risk from going too slow | `slow.v_slow_kmh` and `slow.c` |
| Whether a limit can end the run outright | `hard.<limit>.enabled` and its value |

**Risk map.** `python -m skiopt.analysis.risk_map [--config configs/risk.yaml]` takes a few seconds and plots, for a single 90° turn on a reference pitch:

- the DNF probability for each turn radius (10–30 m) and speed (40–90 km/h), with each hazard term as its own panel
- contours at 1%, 5% and 20%, and the 18–22 m × 60–80 km/h typical band outlined

The turn's DNF probability is P = 1 − exp(−∫h ds) over the arc. Tuning means editing `risk.yaml`, re-running the map, and checking that the typical band stays low-risk while tight, fast turns become risky. That check comes before any optimizer run.

**Toggle tests.** These are in §13:
- Disabling a term makes its contribution exactly zero.
- With `risk.enabled: false` and all hard limits off, P_DNF = 0 for every line.
- Re-enabling the settings restores the original numbers.

### Gates

Each gate has a y position, a turning-pole x, a side (pass left or right), a width w of 4–8 m, and a colour. A gate is passed when

```
x(y_g) lies between x_p + side·c and x_p + side·w,   c = 0.4 m body clearance
```

Without the clearance, the optimizer puts the point mass exactly on the turning pole, which no real racer can do.

### Objectives

A crash and a missed gate both count as a DNF. The core quantity is the expected race score:

```
E[score] = P_finish · T + P_DNF · T_DNF
```

T_DNF = 1.5× a reference winning time. With T around 70 s, each extra 1% of DNF risk costs about 0.35 s, so a line must gain more than that to justify the risk. The risk-appetite sweep over T_DNF is a stretch item.

- **Smooth objective** (gradient methods, and the CMA-ES smooth-loss arm): E[score] + λ·Σ softplus(gate-miss distance)², with λ annealed upward, plus a steep softplus wall for each enabled hard limit (G_max, v_stop, piste). Without the walls, gradient methods get no warning as they approach a limit that ends the run.
- **Hard score** (all reporting, and evolutionary fitness): a run that misses a gate scores DQ_BASE + Σ miss distance, so evolution still sees progress among disqualified runs. Runs that pass every gate score E[score].

## 6. Policies

All policies share one interface, `policy(params, obs, k) → κ`, and run through one rollout function.

- **Spline (open-loop).** A cubic spline x(y) through N control points is converted to ψ_k and κ_k with the same finite-difference scheme as the step function, so the conversion is exact until the 12 m clip.
- **MLP (closed-loop).** 16 → 16 → 16 → 1, about 560 parameters. Output κ = (1/r_floor)·tanh(·).
    - Inputs (16): dx, dy, side and width for the next 3 gates; ψ, v, θ; distance to the nearer piste edge.
    - Inputs are normalized before the network: distances divided by 30 m, v by 25 m/s, angles left in radians. Without this, gate distances in the tens of metres swamp the other inputs.
- **Coach line (baseline, no budget).** A hand-built racer's line: constant 20 m arcs, apex a fixed distance above each gate. It answers "do the optimizers beat a sensible human line?" for almost no work.

## 7. Project structure

About 20 modules. Each file is tagged with its track (§8): [A] simulator and courses, [B] optimizers and analysis. Package versions are pinned in `requirements.txt`.

```
ski-racing-line-optimizer/
  requirements.txt            # pinned direct deps: jax[cpu], optax, evosax, numpy, scipy, matplotlib, pyyaml, pandas, pytest
  requirements-lock.txt       # full pip freeze incl. transitive deps (flax etc.), for exact installs
  configs/
    base.yaml                 # physics constants, Δy, skier bands, budgets, seeds
    risk.yaml                 # hazard terms + hard limits, each with an on/off switch (§5)
    fis_gs.yaml               # GS course rules + ICR article numbers
    equipment.yaml            # fixed setup, recorded only
    experiments/*.yaml        # one per experiment: methods, courses, budget, seeds
  skiopt/
    types.py                  # State, Course, Gate, Params pytrees                      [A]
    terrain.py                # Terrain protocol + θ(y) profile                          [A]
    dynamics.py               # Dynamics protocol + 2D point mass, turn costs            [A]
    hazard.py                 # 4 hazard terms, hard DNFs, per-term switches, log-space P_finish [A]
    courses.py                # FIS rules, validate(), generator, hand-built library     [A]
    rollout.py                # lax.scan rollout, vmap over params and courses           [A]
    objectives.py             # E[score], smooth_loss, hard_score                        [A]
    policies.py               # spline, MLP, coach line                                  [B]
    budget.py                 # rollout-equivalent counter + wall-clock                  [B]
    optim/gradient.py         # Adam on splines, BPTT on MLP                             [B]
    optim/evo.py              # evosax wrappers (CMA-ES, sep-CMA-ES, OpenAI-ES) + own GA [B]
    optim/hybrid.py           # CMA-ES → Adam                                            [B]
    run.py                    # CLI: python -m skiopt.run configs/experiments/x.yaml     [B]
    logger.py                 # manifest.json, progress.csv, champion.npz, skip-if-done  [B]
    analysis/plots.py         # course + line plot, D–v̄ figure, convergence             [A+B]
    analysis/summarize.py     # tables, Mann–Whitney + A12 → SUMMARY.md                  [B]
    analysis/risk_map.py      # DNF risk of one turn over radius × speed, for tuning     [A]
  tests/
  results/                    # git-ignored
  notebooks/demo.ipynb        # loads saved results only
```

The course plot shows the slope plane with red and blue GS gates, start and finish, and the optimized line coloured by speed. A second panel colours the line by hazard rate to show where risk builds up.

## 8. Two-week schedule

The two tracks meet at one interface, `evaluate(params_batch, courses)`. We're currently working through both tracks together; if we split them later, track B builds against a stub of the simulator until track A's rollout lands on day 2. Spline results freeze on day 8; days 9–14 go to neuroevolution and the report.

| Days | Track A · simulator and courses | Track B · optimizers and analysis |
| --- | --- | --- |
| 1–2 | Core sim + physics tests | Evaluator stub, runner, logger |
| 3–4 | FIS rules, generator, course plot | Spline: Adam, CMA-ES, GA (days 3–5) |
| 5–6 | Hazard model with switches, risk map, calibration | Hybrid, smooth-loss arm, summary (days 6–8) |
| **End of day 6** | **Gate: baseline runs end to end on the easy course** | |
| 7–8 | Coach line + deceptive course | (continues) |
| **End of day 8** | **Gate: spline results frozen** | |
| 9–11 | Both: neuroevolution (sep-CMA-ES, OpenAI-ES, GA), BPTT baseline, held-out test | |
| 12–14 | Both: 10-seed runs, figures, report | |

The day-6 gate is the go/no-go check. If the spline baseline isn't producing an optimized line on the easy course by then, cut from the list below before starting anything new.

**Cut order if behind** (first cut first):

1. GA on the MLP
2. Hybrid (CMA-ES → Adam)
3. BPTT controller baseline
4. Deceptive course

After these cuts, Adam, CMA-ES (both arms) and GA on splines, sep-CMA-ES and OpenAI-ES on the MLP, and the coach line are still a complete project.

**Stretch** (only after day 14 or if well ahead): risk-appetite sweep over T_DNF, Monte Carlo race simulation, difficulty sweep, course-risk multiplier m_course and h_fast, execution-noise robustness, ablation table, snow-condition presets, NEAT, animation.

## 9. Fair comparison and metrics

Every method optimizes a flat parameter vector through one batched evaluator, `evaluate(params_batch, courses) → (smooth_loss, hard_score)`, and is scored the same way at the end.

| Method | Policy | Trains on | Notes |
| --- | --- | --- | --- |
| Adam | Spline | Smooth loss | + random-restart Adam at equal budget |
| CMA-ES | Spline | Hard score | Full covariance (N control points is small) |
| CMA-ES (smooth) | Spline | Smooth loss | Core in v2: separates optimizer from objective |
| GA | Spline | Hard score | Own simple implementation |
| Hybrid | Spline | CMA-ES then Adam | f = 0.3 of the budget on CMA-ES |
| sep-CMA-ES | MLP | Hard score, course batch | Replaces full CMA-ES at ~560 dims |
| OpenAI-ES | MLP | Hard score, course batch | |
| GA | MLP | Hard score, course batch | First to cut if behind |
| BPTT | MLP | Smooth loss | Gradient baseline for the controller |
| Coach line | Fixed | — | No budget |

Full CMA-ES is dropped for the MLP because learning a 560 × 560 covariance takes on the order of n² ≈ 300k evaluations, which is at or past the budget.

- **Equal budgets** in rollout-equivalents: one forward rollout = 1, one gradient step ≈ 3 (measured once when Adam first runs, days 3–5). Splines: 2×10⁵ per course. Controller: 2×10⁶ (controller × course) rollouts. Wall-clock is reported too.
- **Equal starts:** a straight fall line plus noise for splines, Gaussian weights for the MLP.
- **Equal tuning:** a small, equal hyperparameter grid per method, tuned only on a tuning course that is never tested.
- **Seeds:** 10 per method and course. Within a generation every candidate sees the same course batch.
- **Generalization:** the controller is tested on 50 held-out courses and compared with a per-course spline (Adam) on each.
- **Final scoring:** always the hard score through the same rollout.

Metrics per method: T, D, v̄, analytic P_DNF, E[score], DNF cause shares, disqualification rate, evaluations to reach within 1% of the best known time, and median turn radius. Across seeds: mean ± std and median with IQR. Each pair of methods gets a Mann–Whitney U test (Holm-corrected) with the Vargha–Delaney A12 effect size.

## 10. Logging and summary

Each run saves everything once, and every figure and table is rebuilt from saved files. Runs are cheap, so v2 drops checkpoints, resume logic, Parquet, the live watcher, LaTeX tables and the presentation loader. At the day-2 target of under 50 ms per 1,000 candidates, a population-method spline run (CMA-ES, GA) takes about 10 s and a controller run about 2 minutes. Adam is slower in wall-clock: its ~66k gradient steps run one after another, so a spline run takes a few minutes. That is still short enough that a killed run is simply re-run.

`results/<experiment>/<method>/seed_<n>/` holds:

| File | Contents |
| --- | --- |
| `manifest.json` | Config snapshot + hash, equipment block, seed, git commit, library versions, wall-clock, total evals, a `complete` flag |
| `progress.csv` | One row per generation or step: evals, best/mean score, T, D, v̄, P_DNF, gate-miss rate, σ or grad norm |
| `champion.npz` | Best parameters + best-so-far history |
| `telemetry.csv` | Champion line per metre: x, y, t, v, ψ, κ, a_net, G, each hazard term, cumulative P_DNF |

- **Skip if done.** The runner skips any run whose manifest is marked complete, so re-running an experiment only fills the gaps.
- **Gate splits** are computed from `telemetry.csv` when needed rather than stored separately.
- **`summarize.py`** writes `SUMMARY.md` with the headline table (one row per method), the pairwise tests, a short race report per champion (time, D, v̄, max G, min radius, share of turns in 18–22 m, risk by cause), and warnings when results look physically wrong:
    - fewer than 50% of turns in the 18–22 m band
    - a sensible line's P_DNF outside the target range
    - a champion sitting at the 12 m clip or at G_max
    - run time or mean speed outside the realism range

## 11. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Running out of time | End-to-end baseline by day 6; fixed cut order (§8) |
| Gate penalty flat or non-differentiable | Softplus miss-distance for gradients, graded DQ for evolution, λ annealed upward |
| Gradient methods stuck in local optima | Part of the finding; random-restart Adam at equal budget |
| Exploding gradients over ~1,000–1,500 scan steps | Gradient clipping; truncated BPTT over gate-to-gate segments as fallback |
| DNF model dominates (all lines too safe or all crash) | Tune `risk.yaml` with the risk map before optimizer runs; calibrate at the day-6 gate against the target DNF rates; switch individual terms off to find the culprit |
| Hazard vanishes in float32 | Log-space accumulation, `-expm1` for P_DNF |
| Deceptive course isn't deceptive | Check first: random-restart Adam must find ≥ 2 optima with a measurable time gap |
| Slow evaluation | jit + vmap; target < 50 ms per 1,000 candidates, measured on day 2 |
| evosax API changes | Versions pinned in `requirements.txt`, full tree in `requirements-lock.txt`; thin wrappers so the `cma` package can be swapped in |
| FIS values out of date | Check the 2026 ICR edition once by hand |

**Deceptive course.** A run of open, low-offset gates on a steep pitch, then a rhythm change into a large-offset gate. The tight, greedy line through the early gates carries too much speed into that gate and pays a big skid loss. The faster line starts the turn earlier and higher, giving up a little time early. Gradient descent from the fall line is expected to settle on the greedy line.

## 12. Design for the 3D extension

The 2D interfaces stay 3D-ready without building anything 3D now.

- **Terrain protocol:** `height / pitch / normal(x, y)`. 2D ignores x; 3D adds a height field and leaves the dynamics untouched.
- **Dynamics protocol:** `step(state, control, terrain, params)` + a `control_spec`. Policies output values in [−1, 1]^d and the dynamics scales them. 2D: d = 1 (curvature). 3D: d = 2 (edge angle, lean).
- **One carving formula everywhere:** R = R_sc·cos(edge), so κ = 1 / (R_sc·cos(edge)). v1's κ ≈ sin(edge)/R_sc is dropped because it contradicts the 2D bands. The edge angles logged in 2D carry straight into 3D.
- **Equipment in 3D:** stack height and any lifter would set the boot-out edge limit and edge leverage, and the heel-to-toe ramp would set fore-aft stance. They are omitted in 2D (§2); measure them when 3D starts.
- **Opaque state:** rollout, objectives and optimizers read only a fixed `StepOutput` (dt, ds, gate signals, hazard terms, hard DNF, telemetry), never x, ψ or v directly.
- **Gate crossing** = crossing a gate plane: y = y_g in 2D, a vertical plane through the poles in 3D.

## 13. Tests and verification

The tests are trimmed to the ones that catch real bugs or give a report figure.

**Physics** (track A)

- No friction, no drag, constant θ, straight line: v² = 2g·sinθ·s and the analytic time.
- Straight-run terminal speed = √(g(sinθ − μ cosθ) / c_d).
- Turn cost: flat above 18 m, rising monotonically from 18 to 12 m; a command below 12 m is clipped to 12 m.
- a_net is lower before the fall line than after it for the same κ and v.
- T = D / v̄ to floating-point precision on every rollout.

**DNF model** (track A)

- A moderate straight run on an easy pitch gives P_DNF ≈ 0.
- Each hazard term rises monotonically with its driver (G, |κ|, κ_req, v below v_slow).
- G > G_max, v < v_stop, a missed gate and leaving the piste each give P_DNF ≈ 1.
- A line that stays straight too long before a gate gets rising κ_req and h_late before it misses.
- Switches: disabling one term makes its contribution exactly zero and leaves the others unchanged. With `risk.enabled: false` and every hard limit off, P_DNF = 0 for any line. Disabling an enabled hard limit also removes its smooth-objective wall.
- The risk map runs from `risk.yaml` alone, and its 18–22 m × 60–80 km/h band matches the calibration target.

**Courses** (track A)

- 1,000 generated courses all pass `validate`.
- Hand-built invalid courses fail with the expected violation: poles too close, width out of range, wrong colour order, wrong direction-change count. Delay gates don't count as direction changes.

**Numerics and optimizers** (track B)

- `jax.grad` agrees with finite differences in float64.
- A vmapped batch equals a Python loop over the same candidates.
- The same seed gives identical results on the same machine.
- Each optimizer improves on the sphere function and on the easy course within a small budget.
- A tiny run writes all four result files, and a second run is skipped.

**Known optima** (both, report figures)

- **Cycloid:** turn costs and risk off, tiny v₀, small lateral offset. Adam on the spline matches the analytic brachistochrone time within about 1%.
- **No gates:** with risk switched off, every method converges to the straight fall line.

**Realism** (at the day-6 gate): a turn-radius histogram with the 18–22 m band shaded, plus run time and mean speed against the 60–80 s and 55–70 km/h targets.

### End-to-end check

1. `pytest` passes.
2. `python -m skiopt.run configs/experiments/baseline.yaml` produces an optimized-line plot.
3. Re-running a finished experiment does no simulation.
4. The full suite (10 seeds) produces convergence plots, final-line figures, the D–v̄ figure, the generalization plot, and a `SUMMARY.md` with no unresolved warnings.

Moved to stretch with their features: the Monte Carlo vs. analytic P_DNF test, the m_course tests, and the "raising T_DNF never raises P_DNF" test (which needs a tolerance, since it tests stochastic optimizer output).
