# Summary: final_trap_margin

Course: `{"kind": "library", "name": "deceptive"}` · policy `arc` · best known E[score] 52.025 s

## Headline

Mean ± std over seeds, finished runs only (no missed gate, no hard DNF).

| method | seeds | finish rate | E[score] (s) | T (s) | D (m) | v̄ (km/h) | P_DNF (%) | evals to 1% of best | wall clock (s) |
|---|---|---|---|---|---|---|---|---|---|
| adam | 3 | 3/3 | 52.07 ± 0.01 | 51.23 ± 0.02 | 975.4 ± 0.2 | 68.5 ± 0.0 | 2.8 ± 0.1 | 131211 (3/3) | 878 ± 16 |
| hybrid | 3 | 3/3 | 52.09 ± 0.07 | 51.31 ± 0.06 | 975.0 ± 0.8 | 68.4 ± 0.1 | 2.6 ± 0.2 | 14592 (3/3) | 585 ± 21 |

## Pairwise tests (hard score, lower is better)

Mann–Whitney U, Holm-corrected. A12 = chance the first method beats the second.

| first | second | A12 | p (Holm) |
|---|---|---|---|
| adam | hybrid | 0.67 | 0.7 |

## Race reports (each method's best run)

- **adam** (seed 1): T 51.24 s · D 975.7 m · v̄ 68.5 km/h · max G 3.33 · median turn radius 21.1 m (tightest point 16.3 m) · turns averaging 18–22 m: 42% · P_DNF 2.7% (risk from load 100%)
- **hybrid** (seed 2): T 51.33 s · D 975.8 m · v̄ 68.4 km/h · max G 3.21 · median turn radius 21.1 m (tightest point 16.2 m) · turns averaging 18–22 m: 47% · P_DNF 2.3% (risk from load 100%)

## Warnings

- ⚠ adam: only 42% of turns average 18–22 m
- ⚠ hybrid: only 47% of turns average 18–22 m
