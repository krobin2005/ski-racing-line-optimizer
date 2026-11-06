# Summary: gran_risa_edgewall

Course: `{"kind": "gran_risa", "seed": 0}` · policy `arc` · best known E[score] 74.831 s

## Headline

Mean ± std over seeds, finished runs only (no missed gate, no hard DNF).

| method | seeds | finish rate | E[score] (s) | T (s) | D (m) | v̄ (km/h) | P_DNF (%) | evals to 1% of best | wall clock (s) |
|---|---|---|---|---|---|---|---|---|---|
| adam | 3 | 3/3 | 74.88 ± 0.06 | 74.57 ± 0.06 | 1311.6 ± 0.2 | 63.3 ± 0.0 | 1.1 ± 0.0 | 110776 (3/3) | 3263 ± 14 |
| cmaes_smooth | 3 | 3/3 | 75.07 ± 0.07 | 74.76 ± 0.08 | 1312.2 ± 0.2 | 63.2 ± 0.1 | 1.1 ± 0.1 | 136256 (3/3) | 162 ± 1 |
| dp_adam | 1 | 1/1 | 75.22 | 74.90 | 1312.6 | 63.1 | 1.1 | 4879 (1/1) | 3251 |
| hybrid | 3 | 3/3 | 74.92 ± 0.08 | 74.62 ± 0.08 | 1311.6 ± 0.3 | 63.3 ± 0.1 | 1.0 ± 0.0 | 8640 (3/3) | 2336 ± 9 |

## Pairwise tests (hard score, lower is better)

Mann–Whitney U, Holm-corrected. A12 = chance the first method beats the second.

| first | second | A12 | p (Holm) |
|---|---|---|---|
| adam | cmaes_smooth | 1.00 | 0.3 |
| adam | hybrid | 0.78 | 0.4 |
| cmaes_smooth | hybrid | 0.11 | 0.4 |

## Race reports (each method's best run)

- **adam** (seed 0): T 74.53 s · D 1311.5 m · v̄ 63.3 km/h · max G 2.59 · median turn radius 25.0 m (tightest point 18.0 m) · turns averaging 18–22 m: 19% · P_DNF 1.1% (risk from load 97%, slow 3%)
- **cmaes_smooth** (seed 2): T 74.67 s · D 1312.0 m · v̄ 63.2 km/h · max G 2.65 · median turn radius 24.9 m (tightest point 18.0 m) · turns averaging 18–22 m: 23% · P_DNF 1.1% (risk from load 96%, slow 4%)
- **dp_adam** (seed 0): T 74.90 s · D 1312.6 m · v̄ 63.1 km/h · max G 2.64 · median turn radius 24.9 m (tightest point 17.6 m) · turns averaging 18–22 m: 26% · P_DNF 1.1% (risk from load 95%, slow 5%)
- **hybrid** (seed 0): T 74.57 s · D 1311.3 m · v̄ 63.3 km/h · max G 2.61 · median turn radius 24.6 m (tightest point 18.0 m) · turns averaging 18–22 m: 19% · P_DNF 1.0% (risk from load 96%, slow 4%)

## Warnings

- ⚠ adam: only 19% of turns average 18–22 m
- ⚠ cmaes_smooth: only 23% of turns average 18–22 m
- ⚠ dp_adam: only 26% of turns average 18–22 m
- ⚠ hybrid: only 19% of turns average 18–22 m
