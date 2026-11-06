# Summary: final_trap

Course: `{"kind": "library", "name": "deceptive"}` · policy `arc` · best known E[score] 51.954 s

## Headline

Mean ± std over seeds, finished runs only (no missed gate, no hard DNF).

| method | seeds | finish rate | E[score] (s) | T (s) | D (m) | v̄ (km/h) | P_DNF (%) | evals to 1% of best | wall clock (s) |
|---|---|---|---|---|---|---|---|---|---|
| adam | 10 | 10/10 | 52.83 ± 0.73 | 51.94 ± 0.41 | 979.9 ± 3.0 | 67.9 ± 0.7 | 3.0 ± 1.2 | never | 828 ± 4 |
| adam_restarts | 10 | 10/10 | 52.56 ± 0.06 | 51.80 ± 0.07 | 980.5 ± 1.0 | 68.1 ± 0.0 | 2.6 ± 0.1 | 100772 (1/10) | 234 ± 2 |
| cmaes | 10 | 10/10 | 52.05 ± 0.06 | 51.23 ± 0.07 | 975.1 ± 0.5 | 68.5 ± 0.1 | 2.7 ± 0.1 | 17280 (10/10) | 53 ± 1 |
| cmaes_smooth | 10 | 10/10 | 52.57 ± 0.03 | 51.82 ± 0.04 | 981.5 ± 0.5 | 68.2 ± 0.1 | 2.5 ± 0.1 | never | 53 ± 1 |
| coach_line | 10 | 0/10 | DQ | – | – | – | – | never | 2 ± 0 |
| ga | 10 | 10/10 | 52.28 ± 0.07 | 51.41 ± 0.05 | 976.2 ± 0.7 | 68.4 ± 0.1 | 2.9 ± 0.2 | 120704 (10/10) | 39 ± 1 |
| hybrid | 10 | 10/10 | 52.12 ± 0.11 | 51.26 ± 0.06 | 974.9 ± 0.8 | 68.5 ± 0.1 | 2.9 ± 0.3 | 16160 (10/10) | 598 ± 10 |

## Pairwise tests (hard score, lower is better)

Mann–Whitney U, Holm-corrected. A12 = chance the first method beats the second.

| first | second | A12 | p (Holm) |
|---|---|---|---|
| adam | adam_restarts | 0.22 | 0.113 |
| adam | cmaes | 0.00 | 0.00274 |
| adam | cmaes_smooth | 0.17 | 0.0561 |
| adam | coach_line | 1.00 | 0.00134 |
| adam | ga | 0.00 | 0.00274 |
| adam | hybrid | 0.00 | 0.00274 |
| adam_restarts | cmaes | 0.00 | 0.00274 |
| adam_restarts | cmaes_smooth | 0.46 | 0.791 |
| adam_restarts | coach_line | 1.00 | 0.00134 |
| adam_restarts | ga | 0.00 | 0.00274 |
| adam_restarts | hybrid | 0.00 | 0.00274 |
| cmaes | cmaes_smooth | 1.00 | 0.00274 |
| cmaes | coach_line | 1.00 | 0.00134 |
| cmaes | ga | 1.00 | 0.00274 |
| cmaes | hybrid | 0.73 | 0.178 |
| cmaes_smooth | coach_line | 1.00 | 0.00134 |
| cmaes_smooth | ga | 0.00 | 0.00274 |
| cmaes_smooth | hybrid | 0.00 | 0.00274 |
| coach_line | ga | 0.00 | 0.00134 |
| coach_line | hybrid | 0.00 | 0.00134 |
| ga | hybrid | 0.09 | 0.011 |

## Race reports (each method's best run)

- **adam** (seed 1): T 51.84 s · D 981.2 m · v̄ 68.1 km/h · max G 3.17 · median turn radius 20.6 m (tightest point 16.4 m) · turns averaging 18–22 m: 71% · P_DNF 2.5% (risk from load 100%)
- **adam_restarts** (seed 0): T 51.65 s · D 978.5 m · v̄ 68.2 km/h · max G 3.16 · median turn radius 20.8 m (tightest point 16.1 m) · turns averaging 18–22 m: 61% · P_DNF 2.6% (risk from load 100%)
- **cmaes** (seed 8): T 51.15 s · D 975.5 m · v̄ 68.7 km/h · max G 3.35 · median turn radius 21.1 m (tightest point 16.3 m) · turns averaging 18–22 m: 34% · P_DNF 2.7% (risk from load 100%)
- **cmaes_smooth** (seed 2): T 51.79 s · D 982.0 m · v̄ 68.3 km/h · max G 3.21 · median turn radius 20.2 m (tightest point 16.4 m) · turns averaging 18–22 m: 58% · P_DNF 2.5% (risk from load 100%)
- **coach_line** (seed 0): T 56.48 s · D 982.1 m · v̄ 62.6 km/h · max G 4.85 · median turn radius 20.6 m (tightest point 12.0 m) · turns averaging 18–22 m: 53% · P_DNF 52.3% (risk from load 57%, tight 3%, late 34%, slow 6%) · **9 gates missed**
- **ga** (seed 6): T 51.35 s · D 976.5 m · v̄ 68.5 km/h · max G 3.35 · median turn radius 20.8 m (tightest point 16.1 m) · turns averaging 18–22 m: 50% · P_DNF 2.8% (risk from load 100%)
- **hybrid** (seed 4): T 51.19 s · D 975.6 m · v̄ 68.6 km/h · max G 3.33 · median turn radius 21.2 m (tightest point 16.2 m) · turns averaging 18–22 m: 39% · P_DNF 2.7% (risk from load 100%)

## Warnings

- ⚠ cmaes: only 34% of turns average 18–22 m
- ⚠ hybrid: only 39% of turns average 18–22 m
