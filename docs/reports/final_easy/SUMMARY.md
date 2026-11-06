# Summary: final_easy

Course: `{"kind": "library", "name": "easy"}` · policy `arc` · best known E[score] 45.277 s

## Headline

Mean ± std over seeds, finished runs only (no missed gate, no hard DNF).

| method | seeds | finish rate | E[score] (s) | T (s) | D (m) | v̄ (km/h) | P_DNF (%) | evals to 1% of best | wall clock (s) |
|---|---|---|---|---|---|---|---|---|---|
| adam | 10 | 10/10 | 45.67 ± 0.01 | 45.10 ± 0.02 | 937.7 ± 0.1 | 74.8 ± 0.0 | 1.7 ± 0.0 | 84272 (10/10) | 813 ± 9 |
| adam_restarts | 10 | 10/10 | 45.84 ± 0.06 | 45.26 ± 0.05 | 938.2 ± 0.6 | 74.6 ± 0.0 | 1.7 ± 0.1 | never | 223 ± 2 |
| cmaes | 10 | 10/10 | 45.29 ± 0.01 | 44.76 ± 0.02 | 935.4 ± 0.0 | 75.2 ± 0.0 | 1.5 ± 0.0 | 4240 (10/10) | 52 ± 4 |
| cmaes_smooth | 10 | 10/10 | 45.91 ± 0.57 | 45.25 ± 0.44 | 938.2 ± 2.2 | 74.6 ± 0.5 | 2.0 ± 0.5 | 87200 (6/10) | 51 ± 0 |
| coach_line | 10 | 10/10 | 48.50 ± 0.00 | 47.61 ± 0.00 | 952.5 ± 0.0 | 72.0 ± 0.0 | 2.9 ± 0.0 | never | 2 ± 0 |
| ga | 10 | 10/10 | 45.35 ± 0.01 | 44.83 ± 0.02 | 935.7 ± 0.1 | 75.1 ± 0.0 | 1.5 ± 0.0 | 65408 (10/10) | 38 ± 0 |
| hybrid | 10 | 10/10 | 45.29 ± 0.01 | 44.77 ± 0.02 | 935.4 ± 0.1 | 75.2 ± 0.0 | 1.5 ± 0.0 | 4272 (10/10) | 586 ± 5 |

## Pairwise tests (hard score, lower is better)

Mann–Whitney U, Holm-corrected. A12 = chance the first method beats the second.

| first | second | A12 | p (Holm) |
|---|---|---|---|
| adam | adam_restarts | 1.00 | 0.00274 |
| adam | cmaes | 0.00 | 0.00274 |
| adam | cmaes_smooth | 0.47 | 0.85 |
| adam | coach_line | 1.00 | 0.00134 |
| adam | ga | 0.00 | 0.00274 |
| adam | hybrid | 0.00 | 0.00274 |
| adam_restarts | cmaes | 0.00 | 0.00274 |
| adam_restarts | cmaes_smooth | 0.32 | 0.372 |
| adam_restarts | coach_line | 1.00 | 0.00134 |
| adam_restarts | ga | 0.00 | 0.00274 |
| adam_restarts | hybrid | 0.00 | 0.00274 |
| cmaes | cmaes_smooth | 1.00 | 0.00274 |
| cmaes | coach_line | 1.00 | 0.00134 |
| cmaes | ga | 1.00 | 0.00274 |
| cmaes | hybrid | 0.74 | 0.227 |
| cmaes_smooth | coach_line | 1.00 | 0.00134 |
| cmaes_smooth | ga | 0.00 | 0.00274 |
| cmaes_smooth | hybrid | 0.00 | 0.00274 |
| coach_line | ga | 0.00 | 0.00134 |
| coach_line | hybrid | 0.00 | 0.00134 |
| ga | hybrid | 0.00 | 0.00274 |

## Race reports (each method's best run)

- **adam** (seed 9): T 45.09 s · D 937.6 m · v̄ 74.9 km/h · max G 2.85 · median turn radius 21.7 m (tightest point 19.8 m) · turns averaging 18–22 m: 57% · P_DNF 1.7% (risk from load 100%)
- **adam_restarts** (seed 4): T 45.17 s · D 937.5 m · v̄ 74.7 km/h · max G 2.85 · median turn radius 21.6 m (tightest point 18.7 m) · turns averaging 18–22 m: 63% · P_DNF 1.7% (risk from load 100%)
- **cmaes** (seed 6): T 44.72 s · D 935.3 m · v̄ 75.3 km/h · max G 2.85 · median turn radius 22.2 m (tightest point 20.0 m) · turns averaging 18–22 m: 29% · P_DNF 1.6% (risk from load 100%)
- **cmaes_smooth** (seed 9): T 44.92 s · D 936.6 m · v̄ 75.1 km/h · max G 2.85 · median turn radius 21.9 m (tightest point 19.9 m) · turns averaging 18–22 m: 51% · P_DNF 1.6% (risk from load 100%)
- **coach_line** (seed 0): T 47.61 s · D 952.5 m · v̄ 72.0 km/h · max G 2.89 · median turn radius 19.9 m (tightest point 17.9 m) · turns averaging 18–22 m: 94% · P_DNF 2.9% (risk from load 100%)
- **ga** (seed 1): T 44.81 s · D 935.6 m · v̄ 75.2 km/h · max G 2.83 · median turn radius 22.2 m (tightest point 18.9 m) · turns averaging 18–22 m: 37% · P_DNF 1.5% (risk from load 100%)
- **hybrid** (seed 7): T 44.73 s · D 935.3 m · v̄ 75.3 km/h · max G 2.85 · median turn radius 22.2 m (tightest point 20.0 m) · turns averaging 18–22 m: 31% · P_DNF 1.6% (risk from load 100%)

## Warnings

- ⚠ adam: mean speed 74.9 km/h outside (55.0, 70.0)
- ⚠ adam_restarts: mean speed 74.7 km/h outside (55.0, 70.0)
- ⚠ cmaes: only 29% of turns average 18–22 m
- ⚠ cmaes: mean speed 75.3 km/h outside (55.0, 70.0)
- ⚠ cmaes_smooth: mean speed 75.1 km/h outside (55.0, 70.0)
- ⚠ coach_line: mean speed 72.0 km/h outside (55.0, 70.0)
- ⚠ ga: only 37% of turns average 18–22 m
- ⚠ ga: mean speed 75.2 km/h outside (55.0, 70.0)
- ⚠ hybrid: only 31% of turns average 18–22 m
- ⚠ hybrid: mean speed 75.3 km/h outside (55.0, 70.0)
