# Summary: gran_risa

Course: `{"kind": "gran_risa", "seed": 0}` · policy `arc` · best known E[score] 74.718 s

## Headline

Mean ± std over seeds, finished runs only (no missed gate, no hard DNF).

| method | seeds | finish rate | E[score] (s) | T (s) | D (m) | v̄ (km/h) | P_DNF (%) | evals to 1% of best | wall clock (s) |
|---|---|---|---|---|---|---|---|---|---|
| adam | 10 | 10/10 | 77.53 ± 0.24 | 77.35 ± 0.25 | 1317.3 ± 0.6 | 61.3 ± 0.2 | 0.7 ± 0.0 | never | 3585 ± 122 |
| cmaes | 10 | 10/10 | 74.80 ± 0.06 | 74.51 ± 0.06 | 1311.3 ± 0.2 | 63.4 ± 0.0 | 1.0 ± 0.0 | 9040 (10/10) | 152 ± 1 |
| cmaes_smooth | 10 | 10/10 | 77.25 ± 0.47 | 77.06 ± 0.47 | 1316.5 ± 1.2 | 61.5 ± 0.3 | 0.7 ± 0.0 | never | 150 ± 2 |
| coach_line | 10 | 10/10 | 95.88 ± 0.00 | 94.89 ± 0.00 | 1333.8 ± 0.0 | 50.6 ± 0.0 | 12.0 ± 0.0 | never | 2 ± 0 |
| dp | 10 | 10/10 | 76.21 ± 0.00 | 75.94 ± 0.00 | 1314.3 ± 0.0 | 62.3 ± 0.0 | 1.0 ± 0.0 | never | 51 ± 4 |
| dp_adam | 10 | 10/10 | 76.21 ± 0.00 | 75.94 ± 0.00 | 1314.3 ± 0.0 | 62.3 ± 0.0 | 1.0 ± 0.0 | never | 3419 ± 96 |
| ga | 10 | 10/10 | 75.12 ± 0.07 | 74.82 ± 0.07 | 1312.2 ± 0.2 | 63.1 ± 0.1 | 1.0 ± 0.0 | 101120 (10/10) | 81 ± 2 |
| hybrid | 10 | 10/10 | 74.88 ± 0.10 | 74.59 ± 0.10 | 1311.5 ± 0.3 | 63.3 ± 0.1 | 1.0 ± 0.0 | 8896 (10/10) | 2519 ± 109 |

## Pairwise tests (hard score, lower is better)

Mann–Whitney U, Holm-corrected. A12 = chance the first method beats the second.

| first | second | A12 | p (Holm) |
|---|---|---|---|
| adam | cmaes | 0.00 | 0.00201 |
| adam | cmaes_smooth | 0.30 | 0.281 |
| adam | coach_line | 1.00 | 0.00166 |
| adam | dp | 0.00 | 0.00166 |
| adam | dp_adam | 0.00 | 0.00166 |
| adam | ga | 0.00 | 0.00201 |
| adam | hybrid | 0.00 | 0.00201 |
| cmaes | cmaes_smooth | 1.00 | 0.00201 |
| cmaes | coach_line | 1.00 | 0.00166 |
| cmaes | dp | 1.00 | 0.00166 |
| cmaes | dp_adam | 1.00 | 0.00166 |
| cmaes | ga | 1.00 | 0.00201 |
| cmaes | hybrid | 0.74 | 0.227 |
| cmaes_smooth | coach_line | 1.00 | 0.00166 |
| cmaes_smooth | dp | 0.00 | 0.00166 |
| cmaes_smooth | dp_adam | 0.00 | 0.00166 |
| cmaes_smooth | ga | 0.00 | 0.00201 |
| cmaes_smooth | hybrid | 0.00 | 0.00201 |
| coach_line | dp | 0.00 | 0.000446 |
| coach_line | dp_adam | 0.00 | 0.000446 |
| coach_line | ga | 0.00 | 0.00166 |
| coach_line | hybrid | 0.00 | 0.00166 |
| dp | dp_adam | 0.50 | 1 |
| dp | ga | 0.00 | 0.00166 |
| dp | hybrid | 0.00 | 0.00166 |
| dp_adam | ga | 0.00 | 0.00166 |
| dp_adam | hybrid | 0.00 | 0.00166 |
| ga | hybrid | 0.02 | 0.00201 |

## Race reports (each method's best run)

- **adam** (seed 4): T 76.86 s · D 1316.3 m · v̄ 61.7 km/h · max G 2.51 · median turn radius 23.8 m (tightest point 16.5 m) · turns averaging 18–22 m: 32% · P_DNF 0.7% (risk from load 92%, slow 8%)
- **cmaes** (seed 3): T 74.42 s · D 1311.1 m · v̄ 63.4 km/h · max G 2.57 · median turn radius 24.3 m (tightest point 17.9 m) · turns averaging 18–22 m: 19% · P_DNF 1.1% (risk from load 96%, slow 4%)
- **cmaes_smooth** (seed 7): T 76.34 s · D 1314.8 m · v̄ 62.0 km/h · max G 2.62 · median turn radius 23.9 m (tightest point 16.8 m) · turns averaging 18–22 m: 25% · P_DNF 0.7% (risk from load 93%, slow 7%)
- **coach_line** (seed 0): T 94.89 s · D 1333.8 m · v̄ 50.6 km/h · max G 2.70 · median turn radius 20.6 m (tightest point 13.8 m) · turns averaging 18–22 m: 42% · P_DNF 12.0% (risk from load 6%, tight 1%, slow 93%)
- **dp** (seed 0): T 75.94 s · D 1314.3 m · v̄ 62.3 km/h · max G 2.56 · median turn radius 24.1 m (tightest point 17.1 m) · turns averaging 18–22 m: 28% · P_DNF 1.0% (risk from load 93%, slow 7%)
- **dp_adam** (seed 0): T 75.94 s · D 1314.3 m · v̄ 62.3 km/h · max G 2.56 · median turn radius 24.1 m (tightest point 17.1 m) · turns averaging 18–22 m: 28% · P_DNF 1.0% (risk from load 93%, slow 7%)
- **ga** (seed 5): T 74.70 s · D 1311.8 m · v̄ 63.2 km/h · max G 2.57 · median turn radius 24.8 m (tightest point 17.9 m) · turns averaging 18–22 m: 19% · P_DNF 1.0% (risk from load 96%, slow 4%)
- **hybrid** (seed 7): T 74.47 s · D 1311.2 m · v̄ 63.4 km/h · max G 2.52 · median turn radius 25.0 m (tightest point 18.0 m) · turns averaging 18–22 m: 17% · P_DNF 1.0% (risk from load 96%, slow 4%)

## Warnings

- ⚠ adam: only 32% of turns average 18–22 m
- ⚠ cmaes: only 19% of turns average 18–22 m
- ⚠ cmaes_smooth: only 25% of turns average 18–22 m
- ⚠ coach_line: only 42% of turns average 18–22 m
- ⚠ coach_line: mean speed 50.6 km/h outside (55.0, 70.0)
- ⚠ dp: only 28% of turns average 18–22 m
- ⚠ dp_adam: only 28% of turns average 18–22 m
- ⚠ ga: only 19% of turns average 18–22 m
- ⚠ hybrid: only 17% of turns average 18–22 m
