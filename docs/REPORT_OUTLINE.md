# Report outline (draft, November 6)

**Working title:** *Racing-line optimization for Giant Slalom: evolutionary vs. gradient methods, from a 2D model to the real Gran Risa.*

This is a living outline. Numbers marked "(3 seeds)" are preliminary; the final tables use 10 seeds (PLAN.md §9). Every number links to a saved result, so nothing has to be re-run to write the report.

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
2. **Sample efficiency** (2D arc): CMA-ES reaches 1% of the best in about 3,000 evaluations; Adam needs about 85,000; the hybrid about 3,300.
3. **Objective vs. optimizer** (2D spline): CMA-ES on the smooth loss finished every seed while CMA-ES on the hard score did not.
4. **Trap course**: 8 Adam starts ended at 8 different optima (53.7 s to DQ). Gradient descent's result depends on where it starts.
5. **Gran Risa (3D), preliminary, seed 0**:

   | Line | Time | DNF risk |
   | --- | --- | --- |
   | DP | 76.13 s | 4.2% |
   | GA | 74.76 s | 4.4% |
   | Coach line | 94.9 s | 14.4% |
   | *Real fastest run 1 (Marco Schwarz)* | *76.00 s* | |

   Peak edge angles of 68–69° sit just under the 70° boot-out angle.
   - Remaining methods and seeds: pending.
   - Figures: `line_on_map_*.png`, `run_trace.png`.
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
