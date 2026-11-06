# Roadmap: 3D on the Gran Risa by November 30

**Goal:** by November 30, 2026, optimize a racing line in 3D on the real Gran Risa (Alta Badia) terrain, with a GS course set on it. Camber, fall-away, rolls and compressions must affect speed, G-force and DNF risk. The 2D project becomes the validated foundation and the "stage 1" chapter of the report.

**Time available:** 25 days (November 5–30).

**Rule for every task:** if it doesn't make the Gran Risa run more realistic or the method comparison more convincing, it waits.

---

## 1. What is (and isn't) available about the Gran Risa

| Need | Availability | Plan |
| --- | --- | --- |
| Course facts | Public. Start ≈ 1,868–1,871 m, finish ≈ 1,420 m, **vertical drop 448 m**, length ≈ 1,225–1,255 m (sources disagree), max gradient ≈ 69% (≈ 35°), average ≈ 36%. | Use as validation targets. |
| Real race times | Public. December 21, 2025 GS: Marco Schwarz won in 2:35.02 over two runs, **about 77 s per run**. | Realism target for the 3D sim: the optimized line should land near 75–80 s. |
| Terrain | **Open data.** South Tyrol publishes digital terrain models: 2.5 m for the whole province, and 0.5 m LiDAR for settlement areas, via WCS/WMS services, mostly CC0. | Download a tile around the piste, crop it, and build a height field. Use 2.5 m if the 0.5 m model doesn't cover the slope. |
| Piste outline and centreline | OpenStreetMap ski pistes (`piste:type=downhill`) usually trace the Gran Risa. | Centreline from OSM, smoothed; piste width from OSM or a fixed 40 m. |
| **Exact gate positions of a real race** | **Not published.** FIS result PDFs give the number of gates and turning gates, the course setter, and start/finish altitudes, but not gate coordinates. | Two options (decision needed, §6). (a) **FIS-faithful set**: our generator, run on the real terrain along the real centreline, constrained to the real gate count from the FIS result PDF. (b) **Digitized set**: gates placed by hand from TV footage or course maps. More authentic, roughly 1–2 days of manual work, and still approximate. |

Being exact about the gates is impossible, so the report should say **"a FIS-valid course set on the real Gran Risa terrain, with the gate count of the December 2025 race"** rather than "the exact 2025 course set."

## 2. What 3D changes in the model

The 2D model already has the hooks for this (PLAN.md §12): a `Terrain` protocol with `height`, `pitch` and `normal`; a `Dynamics` protocol; and an opaque `StepOutput`. The optimizers, evaluator, budget, logger and analysis do not change.

| Piece | 2D now | 3D |
| --- | --- | --- |
| Coordinates | y down the fall line, x across | **Curvilinear along the course centreline**: s along the course, n across, and the height z(s, n) from the terrain. Still a fixed-length scan in s. |
| Terrain | pitch profile θ(y) | Height field from the DTM: bicubic interpolation in JAX, differentiable. It gives the slope gradient (fall line *and* camber) and the surface curvature (rolls, compressions). |
| Gravity | g sinθ along the fall line | Gravity projected on the local surface: a component along the heading (drive) and one across it (**camber / fall-away**). Fall-away (the slope dropping away on the outside of the turn) adds to the lateral load; camber that tilts into the turn helps. |
| Normal load | g cosθ | g·(surface normal component) + **v²·κ_vertical**. A compression adds G; a roll (convex crest) unloads the skier. When the load reaches 0 the skier is airborne. |
| G-force and edge angle | G from a_net and g cosθ | Same idea with the 3D normal load. The **edge angle** follows as atan(lateral load / normal load). The edge limit from the **stack height (49/47 mm) becomes physically meaningful** here: boot-out caps the edge angle. |
| DNF model | load, tight, late, slow | Same four terms plus: **airborne / unloading** over rolls, and **edge angle above boot-out**. Each has a switch and a tunable constant (PLAN.md §5). |
| Line representation | Apex-and-arc line in (x, y) | The same apex-and-arc line in (s, n). The parameters keep their meaning: distance outside the pole and apex shift. |
| Controls | curvature | Keep curvature as the control and *derive* edge and lean from it. PLAN.md §12's two-control version (edge + lean) is a stretch goal. |

## 3. Schedule

| Days | Work | Done when |
| --- | --- | --- |
| **Nov 5–8** | Finish 2D: arc baseline summary; 10-seed final runs on easy + deceptive overnight; freeze the 2D results and figures for the report. | `docs/reports/` holds the final 2D tables. |
| **Nov 6–10** | Terrain pipeline: fetch the Gran Risa DTM tile, crop, build `HeightFieldTerrain` (bicubic, JAX). OSM centreline → curvilinear frame. Plot elevation, slope and camber maps. | The map shows the Gran Risa's known features: the steep upper walls and the final pitch to La Villa. |
| **Nov 10–14** | 3D dynamics on the surface: projected gravity, normal load with vertical curvature, edge angle, airborne check. | Tests pass: a tilted plane reproduces the 2D results exactly; a banked turn matches the analytic answer; a crest at speed v unloads at v² = g·R. |
| **Nov 13–16** | Gran Risa course set: generator along the real centreline with the real gate count; validator in curvilinear coordinates. | FIS-valid set; vertical drop ≈ 448 m. |
| **Nov 16–19** | DNF model in 3D (airborne, boot-out); risk map now includes camber. Calibrate against the real **~77 s** winning run time. | Coach line within ~10% of 77 s; risk in the target range. |
| **Nov 19–24** | Optimizers on the Gran Risa: Adam, CMA-ES (both arms), GA, hybrid, plus the **dynamic-programming global baseline** (§5). 10 seeds. | `SUMMARY.md` and figures for the Gran Risa. |
| **Nov 24–28** | 3D figures (line on the terrain map, speed / G / edge along the run), report draft. | Report draft complete. |
| **Nov 28–30** | Buffer. | — |

**Cut order if behind** (first cut first):
1. The neural controller (it isn't needed for one fixed course; see §5).
2. The two-control (edge + lean) version.
3. A hand-digitized gate set.
4. The 0.5 m terrain (use 2.5 m).

What stays after those cuts is still a complete 3D project.

## 4. Risks

| Risk | Mitigation |
| --- | --- |
| The 0.5 m DTM may not cover the slope | Use the province-wide 2.5 m DTM; GS-scale camber and rolls (≥ 10 m) survive at 2.5 m. |
| DTM has groomed-snow vs. bare-ground differences | Accept (snow depth roughly 0.5–2 m is smooth at GS scale); say so in the report. |
| Curvilinear frame breaks where the centreline bends sharply | Smooth the centreline (curvature radius ≫ piste half-width); cap lateral range to the piste. |
| 3D gradients get rough (terrain interpolation) | Bicubic interpolation (C¹); evolution / hybrid handle roughness, which is itself a result. |
| Evaluation gets slower | The current 2D speed is ≈ 12 ms per 1,000 candidates. The height-field lookup should cost about 2–4× that, which is fine. |

## 5. Method recommendation (answer to "neural network vs. gradient vs. evolution vs. quantized")

**For one fixed course, a neural network is the wrong primary tool.** A controller network is for *generalizing* across many courses or conditions it hasn't seen. For a single known course, optimizing the line directly is simpler, exact and interpretable, and our 2D results say the **representation of the line matters more than the optimizer**: switching from spline to arc gained about 18 s for every method. Keep neuroevolution as an optional, cut-first chapter.

**Gradient descent alone is not enough.** On the trap course, 8 Adam starts ended at 8 different optima, from 53.7 s down to lines that missed gates. Real 3D terrain (camber, rolls) makes the landscape *more* multimodal, not less. The strongest setup is:

1. **Global search** (CMA-ES or a multi-start) to find the right basin;
2. **Gradient refinement** (Adam) inside it, i.e. the **hybrid** we already have.

That is also the cleanest Evolutionary Computing story: evolution for structure, gradients for polish.

**"Quantized":**
- If that means **quantum** optimization (annealing / QAOA): not recommended. The problem is continuous, physics-based and small, it doesn't map naturally to a QUBO, and current hardware offers no advantage at this size.
- If it means **discretizing** the problem: **yes, add it.** A dynamic-programming search over a lattice (each gate × a handful of apex offsets × speed buckets) finds the *global* optimum on that grid in seconds. It's the classic racing-line approach, gives a trustworthy "best possible" reference for every other method, and makes a great starting point for gradient refinement.

## 6. Decisions needed

1. **Gate set:** (a) FIS-faithful generated set with the real gate count *(recommended, fits the timeline)*, or (b) hand-digitized from race footage.
2. **Which race:** December 21, 2025, run 1 (Marco Schwarz) is the most recent. Its FIS result PDF gives the gate count, start and finish.
3. **Neural controller:** cut it, or keep it only as a stretch goal?

## Sources

- [Gran Risa (Wikipedia)](https://en.wikipedia.org/wiki/Gran_Risa)
- [Alta Badia World Cup (official)](https://www.alta-badia.org/en/leisure-activities/winter-sports/alpine-ski-world-cup/)
- [Alpine Ski World Cup: Alta Badia preview](https://www.alpineskiworldcup.com/post/alta-badia-alpine-ski-world-cup-races-preview)
- [South Tyrol DTM 0.5 m (INSPIRE record)](https://inspire-geoportal.ec.europa.eu/srv/api/records/p_bz:Elevation:DigitalTerrainModel-0.5m)
- [South Tyrol geocatalogue DTM 0.5 m](https://geonetwork1.civis.bz.it/geonetwork/srv/api/records/p_bz:Elevation:DigitalTerrainModel-0.5m)
- [Schwarz wins Alta Badia GS, Dec 21 2025 (CBC)](https://www.cbc.ca/sports/olympics/winter/alpine-skiing/marco-schwarz-giant-slalom-victory-alta-badia-italy-9.7024495)
- [FIS news: Schwarz wins Alta Badia GS](https://www.fis-ski.com/alpine-skiing/news/2025-26/schwarz-ends-epic-drought-with-stunning-alta-badia-gs-win)
