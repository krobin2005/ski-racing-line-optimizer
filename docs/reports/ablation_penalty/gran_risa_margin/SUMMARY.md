# Summary: gran_risa_margin

Course: `{"kind": "gran_risa", "seed": 0}` · policy `arc` · best known E[score] 74.880 s

## Headline

Mean ± std over seeds, finished runs only (no missed gate, no hard DNF).

| method | seeds | finish rate | E[score] (s) | T (s) | D (m) | v̄ (km/h) | P_DNF (%) | evals to 1% of best | wall clock (s) |
|---|---|---|---|---|---|---|---|---|---|
| adam | 3 | 3/3 | 76.04 ± 0.04 | 75.84 ± 0.05 | 1312.6 ± 0.1 | 62.3 ± 0.0 | 0.7 ± 0.0 | never | 3608 ± 4 |
| cmaes_smooth | 3 | 3/3 | 76.06 ± 0.15 | 75.86 ± 0.16 | 1312.8 ± 0.2 | 62.3 ± 0.1 | 0.7 ± 0.0 | never | 158 ± 1 |
| dp_adam | 1 | 1/1 | 75.79 | 75.57 | 1313.0 | 62.6 | 0.8 | never | 3585 |
| hybrid | 3 | 3/3 | 74.94 ± 0.09 | 74.64 ± 0.07 | 1311.5 ± 0.3 | 63.3 ± 0.0 | 1.0 ± 0.1 | 7968 (3/3) | 2337 ± 17 |

## Pairwise tests (hard score, lower is better)

Mann–Whitney U, Holm-corrected. A12 = chance the first method beats the second.

| first | second | A12 | p (Holm) |
|---|---|---|---|
| adam | cmaes_smooth | 0.44 | 1 |
| adam | hybrid | 0.00 | 0.3 |
| cmaes_smooth | hybrid | 0.00 | 0.3 |

## Race reports (each method's best run)

- **adam** (seed 1): T 75.79 s · D 1312.7 m · v̄ 62.4 km/h · max G 2.71 · median turn radius 25.2 m (tightest point 16.2 m) · turns averaging 18–22 m: 26% · P_DNF 0.8% (risk from load 94%, slow 6%)
- **cmaes_smooth** (seed 0): T 75.73 s · D 1312.6 m · v̄ 62.4 km/h · max G 2.72 · median turn radius 24.0 m (tightest point 16.6 m) · turns averaging 18–22 m: 23% · P_DNF 0.8% (risk from load 94%, slow 6%)
- **dp_adam** (seed 0): T 75.57 s · D 1313.0 m · v̄ 62.6 km/h · max G 2.66 · median turn radius 24.8 m (tightest point 16.6 m) · turns averaging 18–22 m: 26% · P_DNF 0.8% (risk from load 93%, slow 7%)
- **hybrid** (seed 2): T 74.59 s · D 1311.4 m · v̄ 63.3 km/h · max G 2.57 · median turn radius 25.0 m (tightest point 17.9 m) · turns averaging 18–22 m: 17% · P_DNF 1.0% (risk from load 95%, slow 5%)

## Warnings

- ⚠ adam: only 26% of turns average 18–22 m
- ⚠ cmaes_smooth: only 23% of turns average 18–22 m
- ⚠ dp_adam: only 26% of turns average 18–22 m
- ⚠ hybrid: only 17% of turns average 18–22 m
