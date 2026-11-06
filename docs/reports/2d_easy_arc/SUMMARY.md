# Summary: baseline

Course: `{"kind": "library", "name": "easy"}` · policy `arc` · best known E[score] 45.303 s

## Headline

Mean ± std over seeds, finished runs only (no missed gate, no hard DNF).

| method | seeds | finish rate | E[score] (s) | T (s) | D (m) | v̄ (km/h) | P_DNF (%) | evals to 1% of best | wall clock (s) |
|---|---|---|---|---|---|---|---|---|---|
| adam | 3 | 3/3 | 45.68 ± 0.02 | 45.10 ± 0.02 | 937.7 ± 0.1 | 74.8 ± 0.0 | 1.7 ± 0.0 | 84546 (3/3) | 1029 ± 59 |
| adam_restarts | 3 | 3/3 | 45.74 ± 0.19 | 45.16 ± 0.15 | 937.8 ± 1.3 | 74.8 ± 0.2 | 1.7 ± 0.1 | 732 (2/3) | 273 ± 27 |
| cmaes | 3 | 3/3 | 45.34 ± 0.05 | 44.79 ± 0.01 | 935.5 ± 0.1 | 75.2 ± 0.0 | 1.6 ± 0.1 | 3008 (3/3) | 96 ± 11 |
| cmaes_smooth | 3 | 3/3 | 45.99 ± 0.64 | 45.30 ± 0.46 | 938.4 ± 2.9 | 74.6 ± 0.5 | 2.0 ± 0.5 | 105936 (2/3) | 97 ± 13 |
| coach_line | 3 | 3/3 | 48.50 ± 0.00 | 47.61 ± 0.00 | 952.5 ± 0.0 | 72.0 ± 0.0 | 2.9 ± 0.0 | never | 3 ± 4 |
| ga | 3 | 3/3 | 45.38 ± 0.02 | 44.87 ± 0.02 | 935.7 ± 0.1 | 75.1 ± 0.0 | 1.5 ± 0.1 | 48384 (3/3) | 80 ± 4 |
| hybrid | 3 | 3/3 | 45.34 ± 0.03 | 44.82 ± 0.03 | 935.6 ± 0.2 | 75.2 ± 0.0 | 1.5 ± 0.0 | 3328 (3/3) | 770 ± 67 |

## Pairwise tests (hard score, lower is better)

Mann–Whitney U, Holm-corrected. A12 = chance the first method beats the second.

| first | second | A12 | p (Holm) |
|---|---|---|---|
| adam | adam_restarts | 0.33 | 1 |
| adam | cmaes | 0.00 | 1 |
| adam | cmaes_smooth | 0.56 | 1 |
| adam | coach_line | 1.00 | 1 |
| adam | ga | 0.00 | 1 |
| adam | hybrid | 0.00 | 1 |
| adam_restarts | cmaes | 0.00 | 1 |
| adam_restarts | cmaes_smooth | 0.56 | 1 |
| adam_restarts | coach_line | 1.00 | 1 |
| adam_restarts | ga | 0.00 | 1 |
| adam_restarts | hybrid | 0.00 | 1 |
| cmaes | cmaes_smooth | 1.00 | 1 |
| cmaes | coach_line | 1.00 | 1 |
| cmaes | ga | 0.78 | 1 |
| cmaes | hybrid | 0.56 | 1 |
| cmaes_smooth | coach_line | 1.00 | 1 |
| cmaes_smooth | ga | 0.00 | 1 |
| cmaes_smooth | hybrid | 0.00 | 1 |
| coach_line | ga | 0.00 | 1 |
| coach_line | hybrid | 0.00 | 1 |
| ga | hybrid | 0.11 | 1 |

## Race reports (each method's best run)

- **adam** (seed 1): T 45.08 s · D 937.6 m · v̄ 74.9 km/h · max G 2.85 · median turn radius 21.7 m (tightest point 19.8 m) · turns averaging 18–22 m: 54% · P_DNF 1.7% (risk from load 100%)
- **adam_restarts** (seed 2): T 45.07 s · D 937.0 m · v̄ 74.8 km/h · max G 2.84 · median turn radius 22.3 m (tightest point 19.1 m) · turns averaging 18–22 m: 46% · P_DNF 1.6% (risk from load 100%)
- **cmaes** (seed 0): T 44.78 s · D 935.4 m · v̄ 75.2 km/h · max G 2.84 · median turn radius 22.2 m (tightest point 19.4 m) · turns averaging 18–22 m: 37% · P_DNF 1.5% (risk from load 100%)
- **cmaes_smooth** (seed 1): T 44.98 s · D 936.3 m · v̄ 74.9 km/h · max G 2.88 · median turn radius 22.1 m (tightest point 19.4 m) · turns averaging 18–22 m: 46% · P_DNF 1.7% (risk from load 100%)
- **coach_line** (seed 0): T 47.61 s · D 952.5 m · v̄ 72.0 km/h · max G 2.89 · median turn radius 19.9 m (tightest point 17.9 m) · turns averaging 18–22 m: 94% · P_DNF 2.9% (risk from load 100%)
- **ga** (seed 1): T 44.85 s · D 935.6 m · v̄ 75.1 km/h · max G 2.83 · median turn radius 22.1 m (tightest point 19.8 m) · turns averaging 18–22 m: 40% · P_DNF 1.5% (risk from load 100%)
- **hybrid** (seed 2): T 44.79 s · D 935.4 m · v̄ 75.2 km/h · max G 2.84 · median turn radius 22.2 m (tightest point 19.8 m) · turns averaging 18–22 m: 37% · P_DNF 1.5% (risk from load 100%)

## Warnings

- ⚠ adam: mean speed 74.9 km/h outside (55.0, 70.0)
- ⚠ adam_restarts: only 46% of turns average 18–22 m
- ⚠ adam_restarts: mean speed 74.8 km/h outside (55.0, 70.0)
- ⚠ cmaes: only 37% of turns average 18–22 m
- ⚠ cmaes: mean speed 75.2 km/h outside (55.0, 70.0)
- ⚠ cmaes_smooth: only 46% of turns average 18–22 m
- ⚠ cmaes_smooth: mean speed 74.9 km/h outside (55.0, 70.0)
- ⚠ coach_line: mean speed 72.0 km/h outside (55.0, 70.0)
- ⚠ ga: only 40% of turns average 18–22 m
- ⚠ ga: mean speed 75.1 km/h outside (55.0, 70.0)
- ⚠ hybrid: only 37% of turns average 18–22 m
- ⚠ hybrid: mean speed 75.2 km/h outside (55.0, 70.0)
