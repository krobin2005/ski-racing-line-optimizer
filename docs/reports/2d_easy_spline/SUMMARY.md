# Summary: baseline_spline

Course: `{"kind": "library", "name": "easy"}` · policy `spline` · best known E[score] 62.905 s

## Headline

Mean ± std over seeds, finished runs only (no missed gate, no hard DNF).

| method | seeds | finish rate | E[score] (s) | T (s) | D (m) | v̄ (km/h) | P_DNF (%) | evals to 1% of best | wall clock (s) |
|---|---|---|---|---|---|---|---|---|---|
| adam | 3 | 3/3 | 64.42 ± 0.15 | 63.39 ± 0.16 | 936.9 ± 0.3 | 53.2 ± 0.1 | 6.6 ± 0.0 | never | 1021 ± 61 |
| cmaes | 3 | 2/3 | 66.89 ± 0.47 | 65.39 ± 0.57 | 936.8 ± 0.3 | 51.6 ± 0.4 | 11.1 ± 0.3 | never | 99 ± 13 |
| cmaes_smooth | 3 | 3/3 | 62.94 ± 0.03 | 61.91 ± 0.02 | 935.1 ± 0.5 | 54.4 ± 0.0 | 6.0 ± 0.1 | 127008 (3/3) | 99 ± 13 |
| fall_line | 3 | 0/3 | DQ | – | – | – | – | never | 1 ± 2 |
| ga | 3 | 2/3 | 75.91 ± 1.00 | 75.30 ± 1.16 | 942.2 ± 1.9 | 45.1 ± 0.6 | 16.9 ± 0.9 | never | 72 ± 5 |

## Pairwise tests (hard score, lower is better)

Mann–Whitney U, Holm-corrected. A12 = chance the first method beats the second.

| first | second | A12 | p (Holm) |
|---|---|---|---|
| adam | cmaes | 1.00 | 0.636 |
| adam | cmaes_smooth | 0.00 | 0.636 |
| adam | fall_line | 1.00 | 0.636 |
| adam | ga | 1.00 | 0.636 |
| cmaes | cmaes_smooth | 0.00 | 0.636 |
| cmaes | fall_line | 1.00 | 0.636 |
| cmaes | ga | 1.00 | 0.636 |
| cmaes_smooth | fall_line | 1.00 | 0.636 |
| cmaes_smooth | ga | 1.00 | 0.636 |
| fall_line | ga | 0.00 | 0.636 |

## Race reports (each method's best run)

- **adam** (seed 1): T 63.22 s · D 936.8 m · v̄ 53.3 km/h · max G 2.71 · median turn radius 28.4 m (tightest point 12.0 m) · turns averaging 18–22 m: 0% · P_DNF 6.7% (risk from tight 94%, slow 5%)
- **cmaes** (seed 2): T 63.79 s · D 936.3 m · v̄ 52.8 km/h · max G 2.80 · median turn radius 28.2 m (tightest point 12.0 m) · turns averaging 18–22 m: 8% · P_DNF 9.3% (risk from load 1%, tight 77%, slow 23%) · **1 gates missed**
- **cmaes_smooth** (seed 2): T 61.90 s · D 935.5 m · v̄ 54.4 km/h · max G 2.73 · median turn radius 28.6 m (tightest point 12.0 m) · turns averaging 18–22 m: 3% · P_DNF 5.9% (risk from load 1%, tight 93%, slow 6%)
- **fall_line** (seed 0): T 29.12 s · D 877.0 m · v̄ 108.4 km/h · max G 0.94 · median turn radius nan m (tightest point nan m) · turns averaging 18–22 m: 0% · P_DNF 77.8% (risk from late 100%) · **27 gates missed**
- **ga** (seed 0): T 74.47 s · D 940.9 m · v̄ 45.5 km/h · max G 2.46 · median turn radius 26.7 m (tightest point 12.0 m) · turns averaging 18–22 m: 11% · P_DNF 16.3% (risk from tight 80%, slow 20%)

## Warnings

- ⚠ adam: only 0% of turns average 18–22 m
- ⚠ adam: mean speed 53.3 km/h outside (55.0, 70.0)
- ⚠ cmaes_smooth: only 3% of turns average 18–22 m
- ⚠ cmaes_smooth: mean speed 54.4 km/h outside (55.0, 70.0)
- ⚠ ga: only 11% of turns average 18–22 m
- ⚠ ga: over a quarter of turns touch the 12 m curvature clip
- ⚠ ga: mean speed 45.5 km/h outside (55.0, 70.0)
