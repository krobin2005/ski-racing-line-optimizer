# Summary: gran_risa_sharp

Course: `{"kind": "gran_risa", "seed": 0}` · policy `arc` · best known E[score] 76.210 s

## Headline

Mean ± std over seeds, finished runs only (no missed gate, no hard DNF).

| method | seeds | finish rate | E[score] (s) | T (s) | D (m) | v̄ (km/h) | P_DNF (%) | evals to 1% of best | wall clock (s) |
|---|---|---|---|---|---|---|---|---|---|
| adam | 3 | 0/3 | DQ | – | – | – | – | never | 3659 ± 25 |
| cmaes_smooth | 3 | 3/3 | 79.97 ± 2.15 | 79.73 ± 2.14 | 1322.2 ± 4.6 | 59.7 ± 1.4 | 1.0 ± 0.3 | never | 157 ± 0 |
| dp_adam | 1 | 1/1 | 76.21 | 75.94 | 1314.3 | 62.3 | 1.0 | 4818 (1/1) | 3680 |

## Pairwise tests (hard score, lower is better)

Mann–Whitney U, Holm-corrected. A12 = chance the first method beats the second.

| first | second | A12 | p (Holm) |
|---|---|---|---|
| adam | cmaes_smooth | 0.00 | 0.1 |

## Race reports (each method's best run)

- **adam** (seed 1): T 75.78 s · D 1312.1 m · v̄ 62.3 km/h · max G 2.74 · median turn radius 26.0 m (tightest point 15.3 m) · turns averaging 18–22 m: 25% · P_DNF 0.7% (risk from load 95%, slow 5%) · **16 gates missed**
- **cmaes_smooth** (seed 1): T 77.48 s · D 1317.2 m · v̄ 61.2 km/h · max G 2.57 · median turn radius 23.6 m (tightest point 14.9 m) · turns averaging 18–22 m: 32% · P_DNF 0.8% (risk from load 90%, slow 10%)
- **dp_adam** (seed 0): T 75.94 s · D 1314.3 m · v̄ 62.3 km/h · max G 2.56 · median turn radius 24.1 m (tightest point 17.1 m) · turns averaging 18–22 m: 28% · P_DNF 1.0% (risk from load 93%, slow 7%)

## Warnings

- ⚠ cmaes_smooth: only 32% of turns average 18–22 m
- ⚠ dp_adam: only 28% of turns average 18–22 m
