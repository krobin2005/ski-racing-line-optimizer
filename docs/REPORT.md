# Racing-Line Optimization for Giant Slalom: Evolutionary vs. Gradient Methods, from a 2D Model to the Real Gran Risa

*Kyle Robin and Liam Baron — DCS 340 Evolutionary Computing, Bates College — draft, November 2026*

> **Draft status.** Sections 1–4 describe the model and methods as implemented. Results marked *(3 seeds, preliminary)* come from the day-6 runs and will be replaced by the 10-seed final runs (`configs/experiments/final_easy.yaml`, `final_trap.yaml`, `gran_risa.yaml`). Every number cites a saved result or configuration file, so nothing has to be re-run to check it.

## Abstract

A Giant Slalom (GS) racer wants the line with the highest average speed over the shortest distance, without crashing. The problem is non-convex, constrained by gates and physical limits, and sensitive to how a line is described. It suits a comparison of evolutionary computation and gradient descent.

We build a differentiable point-mass GS simulator, first on a tilted plane (2D), then on the real terrain of the Gran Risa in Alta Badia (3D), with FIS-valid course sets and an explicit, switchable model of DNF risk. We then compare Adam, CMA-ES, a genetic algorithm, a CMA-ES → Adam hybrid and a dynamic-programming lattice search, all at equal evaluation budgets.

Three findings stand out:
1. **Representation matters more than the optimizer.** Describing the line as circular arcs between gate apexes instead of a cubic spline makes every method 17–30 s faster and turns realistic.
2. **Evolution is far more sample-efficient** on this problem. CMA-ES reaches within 1% of the best line about 28× sooner than Adam.
3. **The 3D model, calibrated against the real race, reproduces the actual winning time.** The optimized line runs 75.9 s on the Gran Risa; Marco Schwarz's fastest run 1 on 21 December 2025 was 76.00 s.

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
2. **3D carving loss** (k_turn = 0.18): the optimized line lands at the real winning time. The DP-optimal line runs 72.9 / 75.9 / 79.0 / 81.9 s at k_turn = 0.16 / 0.18 / 0.20 / 0.22.

Calibrating on the *optimized* line rests on the assumption that the winner skis close to the optimal line.

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

### 5.3 Representation beats optimizer *(easy course, 3 seeds, preliminary)*

| Method | Spline time (s) | Arc time (s) | Arc E[score] (s) | Arc DNF risk |
| --- | --- | --- | --- | --- |
| Adam | 63.4 | 45.1 | 45.68 | 1.7% |
| CMA-ES (hard) | 65.4 (2/3 clean) | 44.8 | 45.34 | 1.6% |
| CMA-ES (smooth) | 61.9 | 45.3 | 45.99 | 2.0% |
| GA | 75.3 (2/3 clean) | 44.9 | 45.38 | 1.5% |
| Hybrid | — | 44.8 | 45.34 | 1.5% |
| Coach line | — | 47.6 | 48.50 | 2.9% |

**Turn shape:** with the spline, every method produced "straighten, then hook" lines, averaging about 28 m per turn with peaks at the 12 m clip. The arc line produces turns averaging 21–22 m with tightest points of about 20 m. That is inside the 18–22 m band that men's Continental Cup racers report.

**Sample efficiency** (arc line): to come within 1% of the best known score, CMA-ES needed about 3,000 rollouts, the hybrid about 3,300 and Adam about 85,000.

### 5.4 The trap course

On the deceptive course, 8 independent Adam starts ended at 8 different optima, from 53.7 s down to lines that miss gates. Gradient descent's answer depends on where it starts. The final 10-seed runs compare all methods there.

### 5.5 The Gran Risa *(preliminary)*

| Line | Time | DNF risk |
| --- | --- | --- |
| DP-optimal, k_turn 0.18 | 75.9 s | 1.0% |
| GA, one seed (before the boot-out update) | 74.8 s | 4.4% |
| Coach line | 94.9 s | ~12–14% |
| *Real fastest run 1 (Marco Schwarz)* | *76.00 s* | |

- Optimized lines carve at peak edge angles of 68–70°, peak speeds of about 70 km/h on the steep sections and about 45 km/h in the flatter finish.
- The finish speeds are probably lower than in the real race, which suggests the carving loss is somewhat strong on flat ground (§6).

## 6. Discussion

**When does evolution win?** On every course so far, CMA-ES reaches near-optimal lines with far fewer evaluations than Adam. It handles hard constraints directly, without a schedule of penalty weights, and it isn't tied to its starting point. On the spline it struggled while CMA-ES on the smooth loss did not, which shows the **objective** matters as much as the optimizer.

**When does gradient descent win?** Polishing once inside the right basin. That is the role of the hybrid and of DP → Adam.

**Why the DP matters.** It gives a global reference on a grid, about 5,000 evaluations in about 100 s, against which the continuous methods can be judged. It is also the "discretize and search" approach classically used for racing lines.

**Limitations:**
- a point mass, not an articulated skier (no angulation; boot-out from geometry alone);
- stylized risk constants (tuned with the risk map, not fitted to crash data);
- generated gates, since real positions are unpublished;
- a 2.5 m terrain model, smoothed at 8 m;
- no snow-condition variation;
- the official split times can't be compared section by section, because timing-point positions aren't published;
- finish-flat speeds that are probably low.

## 7. Conclusion and outlook

A differentiable, calibrated simulator turns a GS racing line into a well-posed optimization problem. With a good line representation, the optimizers agree on the answer and evolution gets there much faster. On real terrain, the calibrated 3D model reproduces the real winning time.

Next steps:
- two controls (edge and lean) instead of curvature alone;
- digitized real gate positions;
- snow-condition presets;
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
