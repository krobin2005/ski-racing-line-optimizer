# Summary: final_easy_margin

Course: `{"kind": "library", "name": "easy"}` · policy `arc` · best known E[score] 45.286 s

## Headline

Mean ± std over seeds, finished runs only (no missed gate, no hard DNF).

| method | seeds | finish rate | E[score] (s) | T (s) | D (m) | v̄ (km/h) | P_DNF (%) | evals to 1% of best | wall clock (s) |
|---|---|---|---|---|---|---|---|---|---|
| adam | 3 | 3/3 | 45.37 ± 0.01 | 44.82 ± 0.02 | 935.9 ± 0.1 | 75.2 ± 0.0 | 1.6 ± 0.0 | 96319 (3/3) | 788 ± 4 |
| cmaes_smooth | 3 | 3/3 | 46.00 ± 1.01 | 45.29 ± 0.75 | 938.6 ± 4.5 | 74.6 ± 0.9 | 2.1 ± 0.8 | 138240 (2/3) | 73 ± 0 |
| hybrid | 3 | 3/3 | 45.29 ± 0.01 | 44.77 ± 0.02 | 935.4 ± 0.0 | 75.2 ± 0.0 | 1.5 ± 0.0 | 4320 (3/3) | 585 ± 4 |

## Pairwise tests (hard score, lower is better)

Mann–Whitney U, Holm-corrected. A12 = chance the first method beats the second.

| first | second | A12 | p (Holm) |
|---|---|---|---|
| adam | cmaes_smooth | 1.00 | 0.3 |
| adam | hybrid | 0.00 | 0.3 |
| cmaes_smooth | hybrid | 0.00 | 0.3 |

## Race reports (each method's best run)

- **adam** (seed 1): T 44.80 s · D 935.9 m · v̄ 75.2 km/h · max G 2.85 · median turn radius 22.1 m (tightest point 19.1 m) · turns averaging 18–22 m: 43% · P_DNF 1.6% (risk from load 100%)
- **cmaes_smooth** (seed 2): T 44.88 s · D 936.1 m · v̄ 75.1 km/h · max G 2.84 · median turn radius 22.1 m (tightest point 19.5 m) · turns averaging 18–22 m: 49% · P_DNF 1.6% (risk from load 100%)
- **hybrid** (seed 1): T 44.77 s · D 935.4 m · v̄ 75.2 km/h · max G 2.85 · median turn radius 22.2 m (tightest point 20.0 m) · turns averaging 18–22 m: 34% · P_DNF 1.5% (risk from load 100%)

## Warnings

- ⚠ adam: only 43% of turns average 18–22 m
- ⚠ adam: mean speed 75.2 km/h outside (55.0, 70.0)
- ⚠ cmaes_smooth: only 49% of turns average 18–22 m
- ⚠ cmaes_smooth: mean speed 75.1 km/h outside (55.0, 70.0)
- ⚠ hybrid: only 34% of turns average 18–22 m
- ⚠ hybrid: mean speed 75.2 km/h outside (55.0, 70.0)
