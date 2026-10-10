# Summary: baseline

Course: `{"kind": "library", "name": "easy"}` · policy `spline` · best known E[score] 58.739 s

## Headline

Mean ± std over seeds, finished runs only (no missed gate, no hard DNF).

| method | seeds | finish rate | E[score] (s) | T (s) | D (m) | v̄ (km/h) | P_DNF (%) | evals to 1% of best | wall clock (s) |
|---|---|---|---|---|---|---|---|---|---|
| adam | 3 | 3/3 | 60.47 ± 0.12 | 59.10 ± 0.12 | 937.9 ± 0.4 | 57.1 ± 0.1 | 6.9 ± 0.0 | never | 878 ± 151 |
| adam_restarts | 3 | 3/3 | 60.45 ± 0.17 | 59.07 ± 0.18 | 937.4 ± 0.3 | 57.1 ± 0.2 | 7.0 ± 0.1 | never | 232 ± 51 |
| cmaes | 3 | 1/3 | 65.00 | 63.14 | 935.4 | 53.3 | 11.8 | never | 64 ± 2 |
| cmaes_smooth | 3 | 3/3 | 59.27 ± 0.70 | 57.89 ± 0.74 | 936.1 ± 1.0 | 58.2 ± 0.7 | 6.5 ± 0.0 | 126784 (2/3) | 90 ± 14 |
| fall_line | 3 | 0/3 | DQ | – | – | – | – | never | 1 ± 1 |
| ga | 3 | 1/3 | 69.07 | 67.58 | 938.0 | 50.0 | 13.2 | never | 45 ± 4 |
| hybrid | 3 | 3/3 | 60.88 ± 1.21 | 59.50 ± 1.11 | 937.0 ± 0.9 | 56.7 ± 1.1 | 7.1 ± 0.9 | never | 695 ± 84 |

## Pairwise tests (hard score, lower is better)

Mann–Whitney U, Holm-corrected. A12 = chance the first method beats the second.

| first | second | A12 | p (Holm) |
|---|---|---|---|
| adam | adam_restarts | 0.56 | 1 |
| adam | cmaes | 1.00 | 1 |
| adam | cmaes_smooth | 0.00 | 1 |
| adam | fall_line | 1.00 | 1 |
| adam | ga | 1.00 | 1 |
| adam | hybrid | 0.33 | 1 |
| adam_restarts | cmaes | 1.00 | 1 |
| adam_restarts | cmaes_smooth | 0.00 | 1 |
| adam_restarts | fall_line | 1.00 | 1 |
| adam_restarts | ga | 1.00 | 1 |
| adam_restarts | hybrid | 0.33 | 1 |
| cmaes | cmaes_smooth | 0.00 | 1 |
| cmaes | fall_line | 1.00 | 1 |
| cmaes | ga | 0.89 | 1 |
| cmaes | hybrid | 0.11 | 1 |
| cmaes_smooth | fall_line | 1.00 | 1 |
| cmaes_smooth | ga | 1.00 | 1 |
| cmaes_smooth | hybrid | 1.00 | 1 |
| fall_line | ga | 0.00 | 1 |
| fall_line | hybrid | 0.00 | 1 |
| ga | hybrid | 0.00 | 1 |

## Race reports (each method's best run)

- **adam** (seed 2): T 59.00 s · D 937.5 m · v̄ 57.2 km/h · max G 3.11 · median turn radius 28.3 m (tightest point 12.0 m) · turns averaging 18–22 m: 3% · P_DNF 6.9% (risk from load 6%, tight 92%, slow 2%)
- **adam_restarts** (seed 2): T 58.89 s · D 937.2 m · v̄ 57.3 km/h · max G 3.11 · median turn radius 28.2 m (tightest point 12.0 m) · turns averaging 18–22 m: 3% · P_DNF 6.9% (risk from load 7%, tight 92%, slow 1%)
- **cmaes** (seed 1): T 59.57 s · D 933.5 m · v̄ 56.4 km/h · max G 3.31 · median turn radius 28.3 m (tightest point 12.0 m) · turns averaging 18–22 m: 3% · P_DNF 8.9% (risk from load 12%, tight 82%, slow 7%) · **3 gates missed**
- **cmaes_smooth** (seed 0): T 57.33 s · D 935.7 m · v̄ 58.8 km/h · max G 3.22 · median turn radius 28.6 m (tightest point 12.0 m) · turns averaging 18–22 m: 0% · P_DNF 6.5% (risk from load 13%, tight 84%, slow 3%)
- **fall_line** (seed 0): T 29.12 s · D 877.0 m · v̄ 108.4 km/h · max G 0.94 · median turn radius nan m (tightest point nan m) · turns averaging 18–22 m: 0% · P_DNF 77.8% (risk from late 100%) · **27 gates missed**
- **ga** (seed 1): T 67.58 s · D 938.0 m · v̄ 50.0 km/h · max G 2.87 · median turn radius 27.5 m (tightest point 12.0 m) · turns averaging 18–22 m: 8% · P_DNF 13.2% (risk from tight 94%, slow 6%)
- **hybrid** (seed 2): T 58.78 s · D 937.4 m · v̄ 57.4 km/h · max G 3.13 · median turn radius 28.3 m (tightest point 12.0 m) · turns averaging 18–22 m: 3% · P_DNF 6.6% (risk from load 7%, tight 92%, slow 1%)

## Warnings

- ⚠ adam: only 3% of turns average 18–22 m
- ⚠ adam_restarts: only 3% of turns average 18–22 m
- ⚠ cmaes_smooth: only 0% of turns average 18–22 m
- ⚠ ga: only 8% of turns average 18–22 m
- ⚠ ga: over a quarter of turns touch the 12 m curvature clip
- ⚠ ga: mean speed 50.0 km/h outside (55.0, 70.0)
- ⚠ hybrid: only 3% of turns average 18–22 m
