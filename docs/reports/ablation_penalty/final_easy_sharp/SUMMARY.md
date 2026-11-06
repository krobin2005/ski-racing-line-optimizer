# Summary: final_easy_sharp

Course: `{"kind": "library", "name": "easy"}` · policy `arc` · best known E[score] 45.834 s

## Headline

Mean ± std over seeds, finished runs only (no missed gate, no hard DNF).

| method | seeds | finish rate | E[score] (s) | T (s) | D (m) | v̄ (km/h) | P_DNF (%) | evals to 1% of best | wall clock (s) |
|---|---|---|---|---|---|---|---|---|---|
| adam | 3 | 1/3 | 45.83 | 45.22 | 938.1 | 74.7 | 1.8 | 122 (1/3) | 787 ± 7 |
| cmaes_smooth | 3 | 3/3 | 46.42 ± 0.61 | 45.58 ± 0.48 | 940.0 ± 2.7 | 74.2 ± 0.6 | 2.5 ± 0.4 | 2576 (2/3) | 72 ± 0 |

## Pairwise tests (hard score, lower is better)

Mann–Whitney U, Holm-corrected. A12 = chance the first method beats the second.

| first | second | A12 | p (Holm) |
|---|---|---|---|
| adam | cmaes_smooth | 0.33 | 0.7 |

## Race reports (each method's best run)

- **adam** (seed 0): T 45.22 s · D 938.1 m · v̄ 74.7 km/h · max G 2.88 · median turn radius 21.6 m (tightest point 19.3 m) · turns averaging 18–22 m: 57% · P_DNF 1.8% (risk from load 100%)
- **cmaes_smooth** (seed 2): T 45.26 s · D 938.1 m · v̄ 74.6 km/h · max G 3.08 · median turn radius 21.7 m (tightest point 18.2 m) · turns averaging 18–22 m: 57% · P_DNF 2.2% (risk from load 100%)

## Warnings

- ⚠ adam: mean speed 74.7 km/h outside (55.0, 70.0)
- ⚠ cmaes_smooth: mean speed 74.6 km/h outside (55.0, 70.0)
