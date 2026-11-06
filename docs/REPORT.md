# Racing-Line Optimization for Giant Slalom: Evolutionary vs. Gradient Methods, from a 2D Model to the Real Gran Risa

*Kyle Robin and Liam Baron — DCS 340 Evolutionary Computing, Bates College — draft, November 2026*

> **Draft status.** Results in §5 are the 10-seed final runs (`configs/experiments/final_easy.yaml`, `final_trap.yaml`, `gran_risa.yaml`; 220 runs, none failed). Full tables, significance tests and figures: `docs/reports/final_easy/`, `final_trap/`, `final_gran_risa/`, `sensitivity_mu/`, `sensitivity_k_turn/`. Every number cites a saved result or configuration file, so nothing has to be re-run to check it.

## Abstract

A Giant Slalom (GS) racer wants the line with the highest average speed over the shortest distance, without crashing. The problem is non-convex, constrained by gates and physical limits, and sensitive to how a line is described. It suits a comparison of evolutionary computation and gradient descent.

We build a differentiable point-mass GS simulator, first on a tilted plane (2D), then on the real terrain of the Gran Risa in Alta Badia (3D), with FIS-valid course sets and an explicit, switchable model of DNF risk. We then compare Adam, CMA-ES, a genetic algorithm, a CMA-ES → Adam hybrid and a dynamic-programming lattice search, all at equal evaluation budgets.

Four findings stand out (10 seeds per method, Mann–Whitney tests with Holm correction):
1. **Representation matters more than the optimizer.** Describing the line as circular arcs between gate apexes instead of a cubic spline makes every method 17–30 s faster and turns realistic.
2. **Evolution wins, on every course.** CMA-ES on the true score is best or tied-best on all three courses, and reaches within 1% of the best line about 20× sooner than Adam on the easy course. On the trap course and the Gran Risa, Adam never gets within 1%.
3. **Evolution beats the grid.** On the Gran Risa, CMA-ES (74.51 s), the CMA-ES → Adam hybrid (74.59 s) and the GA (74.82 s) all beat the dynamic-programming lattice optimum (75.94 s), because they can place apexes between the lattice points.
4. **The calibrated 3D model lands close to the real race.** The best line runs 74.4–74.6 s on the Gran Risa; Marco Schwarz's fastest run 1 on 21 December 2025 was 76.00 s, 2% slower, a plausible gap between an optimal line and a real racer.

## 1. Introduction

GS racing is decided by hundredths of a second, and the racing line is the main thing a racer controls. The trade-off is physical and exact: course time T equals distance D divided by mean speed v̄.
- **A short line** points straight at each gate, but needs tight turns that scrub speed through skidding.
- **A round line** keeps speed, but travels further.
- **The best line** gives up a little distance to keep speed.

Real racers also face a third factor: **risk**. Too much speed into a turn, too tight a radius, a late line or too much G-force can end the run. A line that is 0.3 s faster but crashes 40% of the time is a bad line. We therefore optimize the *expected* race score, not raw time (§3.3).

**Why evolutionary computation?** The objective is:
- **Multimodal:** a trap course shows gradient descent landing in different optima from different starts (§5.4).
- **Hard-constrained:** gates, the piste, G-force and the boot-out edge angle.
- **Partly non-smooth:** gate misses and curvature limits.

Gradient methods need a smoothed surrogate; evolutionary methods can work on the true objective. Comparing them at equal cost, on the same simulator and with the same final scoring, is the core of this project.

**Contributions:**
1. A differentiable GS simulator in 2D and 3D on real terrain (JAX, batched: about 12 ms per 1,000 candidate lines in 2D, 165 ms in 3D).
2. FIS-valid course generation (ICR Book IV, July 2025), extended with a skiability rule.
3. An explicit, configurable DNF model.
4. A fair comparison of seven optimizers.
5. A Gran Risa model calibrated to the real 2025 World Cup race.

## 2. Courses and equipment

### 2.1 FIS rules

Courses follow the GS course-setting rules that can be expressed in 2D (ICR July 2025):

| Rule | Value | ICR article |
| --- | --- | --- |
| Vertical drop | 250–450 m | 901.1 |
| Gate width | 4–8 m | 901.2.3 |
| Distance between turning poles of successive gates | ≥ 10 m | 901.2.3 |
| Direction changes | 11–15% of the vertical drop | 901.2.4 |
| Gate colours | alternating red and blue | 901.2.2 |
| Piste width | about 40 m | 902.1 |
| Character of the set | a variety of turns | 903.1.3 |

A gate is passed when the skier's position at the gate line lies between the turning pole plus a 0.4 m body clearance and the outside pole, a point-mass version of ICR 661.4.1.

**A skiability rule (our addition).** A FIS-legal set can still require turns tighter than any ski can carve. Two successive turning gates a distance L apart down the hill, with a lateral swing w, need a turn of radius R = (L² + w²) / 4w. The validator rejects any pair needing less than 13 m, and the generator works backwards from a target radius.

**Courses used:**
- **Easy:** constant 20° pitch, 300 m drop, 34 gates.
- **Trap (deceptive):** a fast, steep 30° section with a quick rhythm, then a rhythm change into a large-offset pair of gates.
- **Random generated courses**, for the generator tests.
- **The Gran Risa** (§4).

### 2.2 Equipment

The skier's setup is fixed in every run:
- Völkl Racetiger GS, 193 cm, 30 m sidecut radius, which is exactly the FIS minimum for men's GS at Continental Cup level;
- piston plate, 30 DIN binding;
- Lange ZB 25.5 boot;
- 49/47 mm stack height (heel/toe).

The sidecut sets the turning bands (§3.1). The stack height sets the boot-out edge angle in 3D (§4.3).

## 3. The 2D model

### 3.1 Dynamics

A point mass moves down a slope in steps of Δy = 1 m along the fall line. The state is the lateral position x, the heading ψ and the speed v; the control is the path curvature κ. Each step:
- clips the curvature at the 12 m floor;
- advances position and heading exactly;
- updates speed through v·dv/ds = g·sinθ·cosψ − μ·g·cosθ − c_d·v² − turn losses.

The lateral load the snow must supply is a_net = κv² + g·sinθ·sinψ. Gravity's sideways pull makes it lower before the fall line and higher after it.

**Turning bands**, tied to the 30 m sidecut through R ≈ R_sc·cos(edge):

| Radius | Behaviour |
| --- | --- |
| ≥ 18 m | clean carve (the typical 18–22 m of men's Continental Cup GS) |
| 14–18 m | a rising speed cost |
| < 14 m | heavy skidding |
| < 12 m | clipped (impossible) |

The tight-turn and skid costs are softplus functions, so the model stays differentiable.

**Constants:**
- 85 kg skier, CdA 0.3 m², air density 1.0 kg/m³ (so c_d ≈ 0.0018 1/m);
- μ = 0.04, consistent with the official race conditions: hard snow at −3 °C;
- start speed 5 m/s.

The carving loss was calibrated so a sensible line lands in 60–80 s and 55–70 km/h on standard courses: k_turn = 0.08 in 2D, k_tight = 2.0, k_skid = 3.0.

**Verification.** 91 automated tests, including:
- exact agreement with the analytic straight run and terminal speed;
- the turning bands;
- gradients checked against finite differences in float64;
- batched equals sequential;
- determinism.

### 3.2 DNF model

Crash risk is a hazard rate h (per metre) accumulated along the run, so P_finish = exp(−Σh·Δs). This keeps the model deterministic and differentiable. Four terms, each switchable in `configs/risk.yaml`:

| Term | Cause |
| --- | --- |
| Load | total G above 3.0 g (edge washout, body load) |
| Tight | radius under 14 m |
| Late | the next gate needs a turn tighter than 14 m: κ_req = 2e⊥ / d² from the current position and heading |
| Slow | speed under 35 km/h, outside the start zone |

**Hard limits**, each also switchable:
- G above 4.0 g (World Cup GS peaks reach about 3.8 body weights; Gilgien et al.);
- speed under 10 km/h;
- leaving the piste;
- missing a gate.

A risk map (`analysis/risk_map.py`) shows the DNF probability of a single 90° turn for every radius and speed. It was used to tune the constants before any optimization: the 18–22 m band at 60–80 km/h is near zero risk, and 18 m at 80 km/h is about 1% per turn.

### 3.3 Objective

The expected race score is E[score] = P_finish·T + P_DNF·T_DNF, where T_DNF = 1.5 × a reference winning time (course length ÷ 60 km/h). With T around 70 s, each extra 1% of DNF risk costs about 0.35 s.
- **Gradient methods** minimize a smooth loss: E[score] + λ·Σ softplus(gate miss)² + steep walls at each hard limit. λ is annealed from 0.1 to 100.
- **Evolutionary methods and all reporting** use the hard score: every DNF (a missed gate, the piste, G, speed) scores 1000 + a graded excess, so evolution still sees progress among disqualified lines. Finishing lines score E[score].

## 4. The 3D Gran Risa model

### 4.1 Terrain and course frame

**Terrain.** The South Tyrol 2.5 m digital terrain model (CC0) is fetched over WCS. The 0.5 m LiDAR model only covers settlement areas and has no data on the slope.

**Piste.** The OpenStreetMap piste line and outline (ODbL) are smoothed into a course centreline with a minimum radius of 66 m, staying within 15 m of the mapped line. Course coordinates are s along the centreline and n across it, and the terrain is resampled onto an (s, n) grid.

**Checked against the published facts:**

| | Model | Published |
| --- | --- | --- |
| Vertical drop | 450 m | 448 m |
| Slope length | ~1,238 m | 1,225–1,255 m |
| Mean gradient | 36–40% | ~36% |
| Maximum gradient | ~70% | 69% |
| Piste width | median 40 m (14–63 m) | ~40 m (ICR 902.1) |

### 4.2 Dynamics on the surface

The skier is a point mass on the snow surface. From the surface's slope and curvature along the path:
- **Normal load** N = v²κ_n + g·u_z. Compressions add load and rolls unload; N < 0 means airborne.
- **Lateral load** F_lat = v²κ_g + g·b_z, where b_z carries camber and fall-away.
- **Drive** −g·T_z, gravity along the path.

The curvature control stays horizontal, so the 2D line descriptions carry over. On a plane the model reduces exactly to 2D. Tests cover:
- planes, crests and compressions;
- camber traverses and fall-away turns;
- banked turns;
- 3D gradients checked against finite differences.

The terrain is smoothed at 8 m (groomed snow has no metre-scale bumps), and its derivatives are precomputed and blended smoothly (C¹).

### 4.3 Edge angle and boot-out

The edge angle is atan(|F_lat| / N). The boot touches the snow at about atan(stack / overhang): with a 49 mm stack, a 90 mm boot sole and a 65 mm ski waist, that is atan(49 / 12.5) ≈ 75.7°.
- DNF risk rises from 71.7°;
- above 80.7° the run ends.

Optimized Gran Risa lines peak at 68–70°, just short of where the risk begins (§5.5).

### 4.4 The course set and calibration

**What the real race gives us.** The FIS official results for 21 December 2025, run 1 (codex 0045):
- 52 gates, 51 of them turning gates;
- course setter Nathan Bryant (USA);
- fastest run 76.00 s by Marco Schwarz, with splits 22.25 / 40.82 / 56.93 s;
- hard snow, −3 °C.

Gate coordinates are not published, so the set is **generated**:
- the real gate and turning-gate counts;
- FIS-valid, inside the real piste edges;
- poles placed with exact designed offsets while steering towards the smoothed piste middle;
- gates narrowed where the piste is narrow;
- every turn at least 14.5 m in horizontal course coordinates.

**Two calibrations, each against a real target:**
1. **Course turn targets** (mean 18 m): the unoptimized coach line then runs the real race pace under the 2D carving loss, 76.2 s at 58.5 km/h against 76.00 s at 58.6 km/h.
2. **3D carving loss** (k_turn = 0.18): chosen so the DP-optimal line lands at the real winning time. The DP line runs 69.5 / 72.9 / 75.9 / 79.0 / 81.9 s at k_turn = 0.14 / 0.16 / 0.18 / 0.20 / 0.22 (`docs/reports/sensitivity_k_turn/`), about 1.5 s per 0.01.

Calibrating on an *optimized* line rests on the assumption that the winner skis close to the optimal line. The final runs then found lines 1.4 s faster than the DP's (§5.5), so the best line now sits 2% under the real time. We kept k_turn = 0.18 rather than re-tune it after seeing the results. k_turn ≈ 0.19 would put the best line at 76.0 s, We expect the ranking of methods to hold there, since k_turn slows every line through the same term, but we have not re-run the comparison at 0.19.

## 5. Methods and experiments

### 5.1 Line representations

- **Spline:** a cubic spline x(y) through control points every 15 m. 10 m gave CMA-ES too many dimensions to find a clean line.
- **Apex-and-arc:** one apex per turning gate, with two parameters each:
  - the distance outside the pole: clearance + softplus(u), so always outside it;
  - the position along the fall line: 6 m above to 3 m below the gate line.

  Successive apexes are joined by two equal circular arcs, R = (L² + w²) / 4w. The default parameters give the **coach line**, a hand-built racer's line every method starts from.

Both representations are converted to curvature commands exactly, so the simulated skier follows the planned line to within centimetres until the 12 m clip.

### 5.2 Optimizers

All at an equal budget of 2×10⁵ rollout-equivalents (one gradient step counts as 3 rollouts), from equal starts, with final scoring on the same hard score:

| Method | Details |
| --- | --- |
| **Adam** | learning rate 0.1 → 0.01, gate penalty λ 0.1 → 100, gradient clipping |
| **Adam with restarts** | 4 parallel starts sharing the budget |
| **CMA-ES (hard score)** | evosax, population 32 |
| **CMA-ES (smooth loss)** | the same λ schedule as Adam, to separate the optimizer from the objective |
| **GA** | population 128, tournament of 3, 4 elites, arithmetic crossover, Gaussian mutation with a decaying step |
| **Hybrid** | CMA-ES for 30% of the budget, then Adam |
| **DP lattice search** (3D) | each apex chooses from 6 margins × 6 positions; segments are simulated with the full 3D physics from the actual entry speed. It keeps the cheapest label per exit-speed band rather than a cost/speed Pareto front, because on steep terrain a *slower* entry is what keeps the next turns feasible. Segments that drift off their end apex are rejected. |
| **DP → Adam** (3D) | the lattice result refined by Adam |

### 5.3 The easy course *(10 seeds)*

| Method | E[score] (s) | T (s) | D (m) | v̄ (km/h) | DNF risk | Rollouts to within 1% of best |
| --- | --- | --- | --- | --- | --- | --- |
| **CMA-ES (hard)** | **45.29 ± 0.01** | 44.76 | 935.4 | 75.2 | 1.5% | 4,240 |
| **Hybrid** | **45.29 ± 0.01** | 44.77 | 935.4 | 75.2 | 1.5% | 4,272 |
| GA | 45.35 ± 0.01 | 44.83 | 935.7 | 75.1 | 1.5% | 65,408 |
| Adam | 45.67 ± 0.01 | 45.10 | 937.7 | 74.8 | 1.7% | 84,272 |
| Adam, 4 restarts | 45.84 ± 0.06 | 45.26 | 938.2 | 74.6 | 1.7% | never |
| CMA-ES (smooth) | 45.91 ± 0.57 | 45.25 | 938.2 | 74.6 | 2.0% | 87,200 (6 of 10 seeds) |
| Coach line | 48.50 | 47.61 | 952.5 | 72.0 | 2.9% | — |

Every method finishes clean on all 10 seeds. CMA-ES and the hybrid are tied (p = 0.23) and beat every other method (Holm-corrected p = 0.003, A12 = 1.0); the GA beats both Adam variants. CMA-ES gets within 1% of the best line about **20× sooner than Adam**.

**Representation beats optimizer** *(3 seeds, `docs/reports/2d_easy_spline/`)*: with the same budget and the spline line, every method was 17–30 s slower (Adam 63.4 s, CMA-ES on the smooth loss 61.9 s, GA 75.3 s with 2 of 3 seeds clean). Spline lines "straighten, then hook", averaging about 28 m per turn with peaks at the 12 m clip. Arc lines turn at 21–22 m on average, with tightest points of about 20 m, inside the 18–22 m band men's Continental Cup racers report.

### 5.4 The trap course *(10 seeds)*

| Method | E[score] (s) | T (s) | Rollouts to within 1% of best |
| --- | --- | --- | --- |
| **CMA-ES (hard)** | **52.05 ± 0.06** | 51.23 | 17,280 |
| Hybrid | 52.12 ± 0.11 | 51.26 | 16,160 |
| GA | 52.28 ± 0.07 | 51.41 | 120,704 |
| Adam, 4 restarts | 52.56 ± 0.06 | 51.80 | 1 of 10 seeds |
| CMA-ES (smooth) | 52.57 ± 0.03 | 51.82 | never |
| Adam | 52.83 ± **0.73** | 51.94 | never |
| Coach line | DQ (misses gates) | — | — |

The deceptive course is built so that the obvious line is wrong: the coach line misses gates on it. Adam's spread across seeds is 12× CMA-ES's, because each start lands in a different local optimum (in an earlier test, 8 Adam starts gave 8 different answers). Restarts make Adam consistent but not better. CMA-ES and the hybrid again tie (p = 0.18) and beat everything else (p ≤ 0.011).

### 5.5 The Gran Risa *(10 seeds)*

| Line | E[score] (s) | T (s) | D (m) | v̄ (km/h) | DNF risk | Rollouts to within 1% |
| --- | --- | --- | --- | --- | --- | --- |
| **CMA-ES (hard)** | **74.80 ± 0.06** | 74.51 | 1311.3 | 63.4 | 1.0% | 9,040 |
| Hybrid | 74.88 ± 0.10 | 74.59 | 1311.5 | 63.3 | 1.0% | 8,896 |
| GA | 75.12 ± 0.07 | 74.82 | 1312.2 | 63.1 | 1.0% | 101,120 |
| DP lattice | 76.21 | 75.94 | 1314.3 | 62.3 | 1.0% | — (≈ 4,800 rollouts) |
| DP → Adam | 76.21 | 75.94 | 1314.3 | 62.3 | 1.0% | — |
| CMA-ES (smooth) | 77.25 ± 0.47 | 77.06 | 1316.5 | 61.5 | 0.7% | never |
| Adam | 77.53 ± 0.24 | 77.35 | 1317.3 | 61.3 | 0.7% | never |
| Coach line | 95.88 | 94.89 | 1333.8 | 50.6 | 12.0% | — |
| *Real fastest run 1 (Marco Schwarz)* | | *76.00* | | | | |

All 80 optimizer runs finish clean. The ranking is the same as in 2D: CMA-ES ≈ hybrid (p = 0.23) > GA > DP > the two smooth-loss methods, which tie with each other (p = 0.28). Every other pair differs at p ≤ 0.002.

- **The continuous searches beat the lattice** by 1.1–1.4 s. The DP is optimal only on its grid of 6 margins × 6 apex positions per gate; CMA-ES places apexes between those points.
- **DP → Adam did not improve the DP line at all** (identical in every seed). §6 explains why.
- **The best lines are physically plausible:** peak speed about 80 km/h on the steep upper section, peak load 2.5–2.6 g (the hard limit is 4 g), peak edge angle 67–70° (boot-out risk starts at 71.7°), and the normal load never approaches zero, so no line goes airborne (`run_trace.png`). Their turns average 20–28 m, mostly a little rounder than the 18–22 m band: on this steep, fast course the optimum gives up some distance to keep speed.
- **Risk** on the optimized lines comes almost entirely from load (92–96%). On the coach line it comes from being slow (93%): it scrubs speed in the flatter sections, down to about 20 km/h in the finish.
- **The finish section:** optimized lines carry about 45 km/h through the flatter finish. A cautious check against the official splits (the timing points' positions are not published): if the third timing point is where the DP line reaches Schwarz's third split (56.93 s), about 896 m down the course, his last section of about 260 m took 19.07 s, about 49 km/h. The model may be slightly too slow on flat ground (`split_times.png`).

### 5.6 Snow conditions *(DP line, `docs/reports/sensitivity_mu/`)*

The race was run on hard snow at −3 °C; our default ski–snow friction is μ = 0.04. Re-optimizing the DP line at other values:

| μ | 0.02 | 0.03 | **0.04** | 0.05 | 0.06 | 0.08 |
| --- | --- | --- | --- | --- | --- | --- |
| DP line T (s) | 73.59 | 74.78 | **75.94** | 77.27 | 78.55 | 81.75 |
| v̄ (km/h) | 64.3 | 63.3 | 62.3 | 61.2 | 60.2 | 57.9 |
| DNF risk | 1.3% | 1.1% | 1.0% | 1.0% | 1.1% | 2.1% |
| Coach line T (s) | 91.42 | 93.09 | 94.89 | 96.85 | 99.00 | 104.16 |

Each 0.01 of friction costs the optimized line about 1.2–1.6 s, so snow conditions matter as much as most of the differences between methods. The unoptimized coach line loses about 1.5× as much, because it already runs slowly in the flat sections, where friction dominates gravity. Slow snow also raises the coach line's risk of a too-slow DNF (9.6% → 19.2%).

## 6. Discussion

**Evolution wins, and the objective is why.** CMA-ES on the true (hard) score is best or tied-best on every course, and Adam is never better than fourth. The cleanest evidence that the *objective* matters as much as the optimizer is CMA-ES itself: run on the smooth loss Adam uses, it falls to Adam's level on all three courses (Gran Risa 77.25 s vs. 74.80 s on the hard score).

**The smooth gate penalty is biased.** To make gate misses differentiable, the smooth loss adds λ · softplus(β·miss)² / β² per gate, with β = 5 /m and λ annealed from 0.1 to 100. Softplus is never exactly zero: a line passing *exactly* at the pass limit still pays (ln 2 / 5)² × 100 ≈ 1.9 s per gate at the final λ. So the methods that descend the smooth loss pass every gate wider than they need to. Their Gran Risa lines are 5–6 m longer than CMA-ES's (1317 m vs. 1311 m), and the logged runs show it directly: once every gate is passed, Adam's line gets *slower* as λ rises (DP → Adam: 77.7 s at λ ≈ 7.5, 82.2 s at λ = 100, with no gate missed). The hard score has no such bias, which is why CMA-ES, GA and the CMA-ES phase of the hybrid do not suffer from it.

**Why DP → Adam did nothing.** The refinement starts from the DP line, which passes every gate, but its λ schedule starts at 0.1. At that weight dropping gates is cheap: Adam immediately gave up 47 of them for speed, then had to win them back as λ grew, and by then the bias above made its clean lines slower than the DP line it started from. The run keeps the best clean line seen, so it returned the DP line unchanged. The same happens in the hybrid's Adam phase: in all 30 hybrid runs (3 courses × 10 seeds) its best clean line was 0.3–2 s *worse* than the CMA-ES phase it started from, so the hybrid's result is its CMA-ES result. The ablations below test the fix: a sharper penalty (β = 200, bias ≈ 0.001 s per gate), and for DP → Adam a penalty that starts at full weight.

**Ablation 1: a sharp penalty** (β = 200, `configs/experiments/gran_risa_sharp*.yaml`, `final_easy_sharp.yaml`; 3 seeds, same budgets). Removing the bias removes the wide lines, but swaps them for the opposite fault: Adam now ends a few millimetres to centimetres *outside* the gates. A quadratic penalty's optimum sits where the time saved per metre of overshoot equals 2λ·miss, so at λ = 100 a small overshoot is always worth it.

| run (β = 200) | T of the final line | gates missed (total miss) | clean? | β = 5 result |
|---|---|---|---|---|
| Gran Risa, Adam, seeds 0/1/2 | 75.48 / 75.78 / 75.63 s | 13 / 16 / 16 (7–10 cm) | 0 of 3 | 77.35 s mean, all clean |
| Gran Risa, DP → Adam (λ = 100 from the start) | 74.97 s | 5.5 cm in total | no; result stays the DP line, 75.94 s | 75.94 s |
| easy, Adam, seeds 0/1/2 | 45.22 (clean) / 44.73 / 44.73 s | 0 / 4 / 2 (≤ 1.2 cm) | 1 of 3 | 45.10 s mean, all clean |
| Gran Risa, CMA-ES on the smooth loss | 79.73 ± 2.14 s | 0 | 3 of 3 | 77.06 s |
| easy, CMA-ES on the smooth loss | 45.58 ± 0.48 s | 0 | 3 of 3 | 45.25 s |

So the bias explains most of the gradient methods' gap: unbiased, Adam's Gran Risa lines are 1.6–1.9 s faster than with β = 5 (1.0–1.3 s behind CMA-ES's 74.51 s), and on the easy course it reaches 44.73 s, level with CMA-ES's clean 44.76 s, apart from the millimetres it clips. Starting from the DP line at full weight, Adam takes a full second off it (75.94 → 74.97 s) while staying within 5.5 cm of every gate. What the gradient methods lack is an exact way to stay feasible, not a good direction to move in. CMA-ES on the smooth loss does not gain: on the Gran Risa it gets worse (79.73 ± 2.14 s vs. 77.06 s, with a wide spread over 3 seeds), and on the easy course it is unchanged within noise. We have not isolated why; one candidate is that the β = 5 penalty's pull towards the middle of each gate also steered CMA-ES's early search.

**Ablation 2: a sharp penalty with a 5 cm margin** (`configs/experiments/*_margin*.yaml`; 3 seeds). The smooth miss is measured to a gate window shrunk by 5 cm on each side (`objective.miss_margin`), so the small overshoot of ablation 1 lands inside the real gate. The hard score, and so every number below, still uses the real gate.

| method | course | E[score] (s) | T (s) | clean | β = 5 (10 seeds) | CMA-ES, hard score |
|---|---|---|---|---|---|---|
| Adam | Gran Risa | 76.04 ± 0.04 | 75.84 ± 0.05 | 3 of 3 | 77.53 / 77.35 | 74.80 / 74.51 |
| CMA-ES (smooth) | Gran Risa | 76.06 ± 0.15 | 75.86 ± 0.16 | 3 of 3 | 77.25 / 77.06 | |
| DP → Adam | Gran Risa | 75.79 | 75.57 | yes | 76.21 / 75.94 (= DP) | |
| Adam | easy | 45.37 ± 0.01 | 44.82 ± 0.02 | 3 of 3 | 45.67 / 45.10 | 45.29 / 44.76 |
| CMA-ES (smooth) | easy | 46.00 ± 1.01 | 45.29 ± 0.75 | 3 of 3 | 45.91 / 45.25 | |

Every run is now clean, and the gradient methods' Gran Risa lines are 1312.6 m long instead of 1317 m, close to CMA-ES's 1311.3 m. Adam's gap to CMA-ES on the hard score halves on the Gran Risa (2.7 → 1.2 s E[score]) and drops from 0.38 to 0.08 s on the easy course. The DP → Adam refinement now does what it was meant to: it improves on the DP line it starts from (75.94 → 75.57 s), the only time a gradient phase has beaten its starting point in this study. CMA-ES on the smooth loss now matches Adam on the Gran Risa (76.06 vs. 76.04 s); on the easy course two of its seeds are level with Adam and one (seed 1, 47.16) lands in a worse basin.

So the gradient methods' handicap is mostly the constraint handling. What remains (Adam 1.2 s behind CMA-ES on the Gran Risa, at about 25× the wall-clock time) is the start-dependence of local search: Adam starts from a random perturbation of the default line (σ = 0.5), and the DP start ends 0.3 s better than those starts.

**Ablation 3: the margin penalty on the trap course and in the hybrid** (`final_trap_margin.yaml`, `*_margin_hybrid.yaml`; 3 seeds). The hybrid's Adam phase now starts at full weight from CMA-ES's clean line.

| run (β = 200, 5 cm margin) | E[score] (s) | T (s) | β = 5 (10 seeds) | CMA-ES, hard score |
|---|---|---|---|---|
| Adam, trap | 52.07 ± 0.01 | 51.23 ± 0.02 | 52.83 ± 0.73 | 52.05 ± 0.06 |
| hybrid, trap: Adam phase vs. its CMA-ES phase | +0.03 to +0.10 | | +0.3 to +2 (all 30 runs, 3 courses) | |
| hybrid, easy: Adam phase vs. its CMA-ES phase | +0.14 to +0.16 | | | |
| hybrid, Gran Risa: Adam phase vs. its CMA-ES phase | +0.85 to +0.99 | | | |

On the trap course, the deceptive one, Adam with the margin penalty is now level with CMA-ES (52.07 vs. 52.05 s) and its spread over seeds drops from 0.73 to 0.01 s. In 2D, then, the whole gap between gradient descent and evolution came from the penalty. The hybrid's Adam phase still never improves on its CMA-ES phase, but in 2D it now ends within 0.03–0.16 s of it, close to the 5 cm margin's own cost.

On the Gran Risa it still ends almost 1 s worse, and the logs show why: its *first* logged line is already 1 s slower than the CMA-ES line it started from, so the smooth loss itself prefers a slower line there. The culprit is a second biased term that only exists in 3D: the smooth wall at the hard edge-angle limit (`edge_max`, 80.7°). It uses the shared wall sharpness (β = 20), which suits an excess in metres or g but is soft for one in radians, so a line 11° below the limit still pays for it. Evaluated on the saved champions with λ = 100, the smooth loss exceeds the hard score by 5.3 s on the DP → Adam line, and the edge wall alone charges CMA-ES's best line about 2.8 s more than Adam's, more than the 1.3 s CMA-ES's line actually saves. With a sharp edge wall (`sim3d.risk.wall_beta_edge` = 1146 /rad, i.e. 20 /deg) the smooth loss of the DP → Adam line is within 0.04 s of its hard score. (CMA-ES's line still costs more in the smooth loss, but only because it passes the gates closer than the 5 cm margin: about 0.25 s per gate, as intended.)

*Ablation 4 (`configs/experiments/gran_risa_edgewall*.yaml`): margin penalty plus a sharp edge wall for Adam, CMA-ES on the smooth loss, the hybrid and DP → Adam on the Gran Risa. Running.*

**When does gradient descent help?** Only for polishing inside the right basin, and only with an unbiased objective. Its gradients are cheap and exact (one step costs about 3 rollouts), but on the trap course and the Gran Risa it never got within 1% of the best line, and its answer depends on the start (the trap course's 12× larger spread across seeds).

**Why the DP matters.** It gives a global reference on a grid in about 4,800 rollouts and about 50 s, and it is the "discretize and search" approach classically used for racing lines. The continuous methods beating it by 1.1–1.4 s shows how much a 6 × 6 grid per gate leaves on the table.

**Snow matters as much as the method.** Moving μ by ±0.01 changes the optimized time by about ±1.3 s, the same size as the gap between the GA and the DP. A real comparison between two racers' lines would need the snow held fixed, as it is in this model.

**Limitations:**
- a point mass, not an articulated skier (no angulation; boot-out from geometry alone);
- stylized risk constants (tuned with the risk map, not fitted to crash data);
- the DNF time penalty is fixed (1.5 × the course length at 60 km/h, ≈ 103 s on the Gran Risa); a line slower than that, like the coach line on slow snow (104 s at μ = 0.08), scores *better* with more risk;
- generated gates, since real positions are unpublished;
- a 2.5 m terrain model, smoothed at 8 m;
- snow conditions as one friction coefficient, with no change in carving grip;
- the official split times can't be compared section by section, because timing-point positions aren't published.

## 7. Conclusion and outlook

A differentiable, calibrated simulator turns a GS racing line into a well-posed optimization problem. With a good line representation, evolution finds better lines than gradient descent, far sooner, on a plane, on a deceptive course and on the real Gran Risa, where the best line runs 74.5 s against the real winner's 76.00 s. Much of gradient descent's handicap comes from the smooth surrogate it needs, not from the gradient itself.

Next steps:
- an unbiased differentiable gate constraint (augmented Lagrangian or a hinge penalty), so the hybrid's gradient phase can actually polish;
- two controls (edge and lean) instead of curvature alone;
- digitized real gate positions;
- snow presets that change carving grip as well as glide;
- a neural controller that generalizes across courses (the stretch goal).

## References

- FIS International Ski and Snowboard Competition Rules, Book IV (Alpine), Edition July 2025.
- FIS Specifications for Alpine Competition Equipment, 2024/2025.
- FIS official results and analysis, Alta Badia men's GS, 21 December 2025, codex 0045 (Longines Timing).
- Gilgien M. et al. (2018). The external forces acting on world cup alpine skiers. *Frontiers in Physiology* 9:145.
- Spörri J. et al. Course setting and injury risk in GS (gate offset vs. vertical gate distance).
- Hansen N. The CMA evolution strategy (tutorial); evosax (JAX evolution strategies).
- Kingma D. P., Ba J. (2015). Adam: a method for stochastic optimization.
- Autonomous Province of Bolzano – South Tyrol, DigitalTerrainModel-2.5m (CC0). © OpenStreetMap contributors (ODbL).
- Snyder J. P. (1987). Map Projections: A Working Manual. USGS Professional Paper 1395.
