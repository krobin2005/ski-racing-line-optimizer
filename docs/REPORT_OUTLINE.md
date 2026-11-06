# Report outline (draft, November 6)

**Working title:** *Racing-line optimization for Giant Slalom: evolutionary vs. gradient methods, from a 2D model to the real Gran Risa.*

This is a living outline. Numbers marked "(3 seeds)" are from the day-6 runs; the final tables (Nov 6) use 10 seeds (PLAN.md §9). Every number links to a saved result, so nothing has to be re-run to write the report.

## 1. Introduction
- The problem: fastest line through a GS course, **without going down**.
  - Time = distance ÷ average speed: shorter lines need tighter turns, which cost speed.
  - DNF risk is explicit: expected race score = P_finish·T + P_DNF·T_DNF.
- Why it is an evolutionary-computing problem:
  - multimodal (the trap course);
  - hard constraints (gates, piste, G, edge angle);
  - non-smooth physics.
- Contributions:
  1. a differentiable GS simulator in 2D, then 3D on real terrain;
  2. a fair method comparison at equal budgets;
  3. evidence that the line representation matters more than the optimizer;
  4. a calibrated Gran Risa model that reproduces the real 2025 race time.

## 2. Model
- **2D point mass** (PLAN.md §4):
  - turning bands from the 30 m sidecut;
  - lateral load including gravity's sideways pull;
  - calibrated carving and tight-turn losses.
- **DNF model** (§5): four hazard terms plus hard limits, every one switchable; the risk map.
  - Figure: `analysis/risk_map.py` output.
- **3D point mass on real terrain** (`skiopt/sim3d.py`):
  - normal load v²κ_n + g·u_z (compressions and rolls);
  - lateral load v²κ_g + g·b_z (camber and fall-away);
  - edge angle and boot-out from the 49 mm stack.
  - Validated on planes, crests, compressions, camber and banked turns (`tests/test_sim3d.py`).
- **Courses**:
  - FIS GS rules (ICR July 2025, articles 901–903) plus the skiable-turn rule;
  - generator, easy course, trap course;
  - the Gran Risa with the real 2025 race's gate count.

## 3. Methods
- **Line representations**: spline (control points every 15 m) vs. apex-and-arc (one apex per gate, circular arcs).
- **Optimizers**:
  - Adam (+ restarts, gate-penalty annealing);
  - CMA-ES on the hard score and on the smooth loss;
  - GA;
  - hybrid (CMA-ES → Adam);
  - **DP lattice search** (global, "discretize and search");
  - DP → Adam.
- **Fairness** (§9): equal rollout-equivalent budgets, equal starts (the coach line), equal tuning, the same final scoring.

## 4. Results
1. **Representation beats optimizer** (2D, easy course, 3 seeds):
   - Spline line: 61.9–75.3 s, up to 17% DNF; GA and hard-score CMA-ES finished 2 of 3 seeds. Turns averaged 28 m with peaks at the 12 m clip ("straighten, then hook").
   - Arc line: every method 44.8–45.1 s at about 1.5% DNF, with turns averaging 21–22 m.
   - Tables: `docs/reports/2d_easy_spline`, `docs/reports/2d_easy_arc`.
2. **Final 10-seed results** (full write-up: REPORT.md §5; tables: `docs/reports/final_*`):
   - Easy: CMA-ES 45.29 ≈ hybrid 45.29 > GA 45.35 > Adam 45.67 s (E[score]). CMA-ES reaches 1% of the best ~20× sooner than Adam (4,240 vs. 84,272 rollouts).
   - Trap: CMA-ES 52.05 ≈ hybrid 52.12 > GA 52.28 > Adam 52.83 ± 0.73 s; Adam's seed spread is 12× CMA-ES's, and it never gets within 1%. The coach line misses gates.
   - Gran Risa: CMA-ES 74.51 ≈ hybrid 74.59 > GA 74.82 > DP 75.94 > CMA-ES (smooth) 77.06 ≈ Adam 77.35 s; real 76.00 s. All 80 optimizer runs clean, peak edge 67–70° (boot-out risk from 71.7°), peak load ≈ 2.6 g.
3. **Objective vs. optimizer**: CMA-ES on the smooth loss falls to Adam's level on every course. Two biased smooth terms explain it: the softplus gate penalty (≈ 1.9 s per gate at the pass limit at λ = 100) and, in 3D, a soft wall at the edge-angle limit. Ablations (`configs/experiments/*_sharp*`, `*_margin*`, `gran_risa_edgewall*`): a sharp gate penalty with a 5 cm margin ties Adam with CMA-ES on the trap course; adding a sharp edge wall brings Adam to 74.88 s on the Gran Risa (CMA-ES 74.80 s, GA 75.12 s), at ~20× the wall clock.
4. **Snow sensitivity**: about 1.3 s per 0.01 of ski–snow friction for the optimized line (`docs/reports/sensitivity_mu`).
5. **Figures**: `line_on_map_*.png`, `run_trace.png`, `split_times.png`, `convergence.png`, `turn_radii.png`, `sensitivity.png`.
6. **Calibration story** (worth its own subsection): every constant tuned against a real target and documented.
   - Turn-radius bands: Continental Cup racers average 18–22 m.
   - Carving loss: the realism ranges in 2D, the real 76 s in 3D.
   - Course turn targets: the coach line reproduces the real race pace.
   - Edge risk: the boot-out angle from the stack height.

## 5. Discussion
- **When does evolution win?** It needs far fewer evaluations to get close, it handles hard constraints directly, and it isn't trapped by its starting point.
- **When does gradient descent win?** Fine polishing once in the right basin, which is why the hybrid and DP → Adam exist.
- **Why the DP matters:** a global reference on a grid, so the continuous methods can be judged against a near-optimal answer.
- **Limits:**
  - a point mass, not an articulated skier;
  - stylized risk constants;
  - generated gates, because real gate positions aren't published;
  - a 2.5 m terrain model, smoothed at 8 m;
  - no snow or weather variation;
  - finish-flat speeds that are probably low.

## 6. Conclusion and 3D outlook
- Two-control (edge + lean) dynamics; real gate positions digitized from footage; snow conditions; a neural controller across many courses.

## Figure list
| Figure | Source |
| --- | --- |
| Course plot (2D) | `analysis/plots.py` |
| Risk map | `analysis/risk_map.py` |
| Convergence, D–v̄ plane, turn radii | `analysis/figures.py` |
| Gran Risa terrain map, profile, camber / curvature | `analysis/terrain_maps.py` |
| Gran Risa line on map, run trace | `analysis/gran_risa_figures.py` |
