# Sensitivity to `sim3d.k_turn`

Default `sim3d.k_turn` = 0.18. One run per value and method; T = course time, E = expected score (time with the DNF risk priced in), v̄ = mean speed. ✗ marks a line that misses a gate or hits a hard limit.

| method | 0.14 | 0.16 | 0.18 | 0.2 | 0.22 |
|---|---|---|---|---|---|
| coach_line | T 88.57 s · E 90.26 · 54.2 km/h · P_DNF 11.6% | T 91.80 s · E 93.09 · 52.3 km/h · P_DNF 11.4% | T 94.89 s · E 95.88 · 50.6 km/h · P_DNF 12.0% | T 97.87 s · E 98.55 · 49.1 km/h · P_DNF 12.9% | T 100.72 s · E 101.06 · 47.7 km/h · P_DNF 14.0% |
| dp | T 69.53 s · E 70.64 · 68.1 km/h · P_DNF 3.3% | T 72.91 s · E 73.40 · 64.9 km/h · P_DNF 1.6% | T 75.94 s · E 76.21 · 62.3 km/h · P_DNF 1.0% | T 78.96 s · E 79.16 · 59.9 km/h · P_DNF 0.8% | T 81.90 s · E 82.10 · 57.8 km/h · P_DNF 0.9% |
