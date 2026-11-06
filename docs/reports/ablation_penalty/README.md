# Smooth-loss bias ablation (REPORT.md §6)

Summaries of the ablation runs (raw runs live in `results/`, which is not in git; re-run with
`python -m skiopt.batch <configs>`). Each folder's SUMMARY.md is the output of
`python -m skiopt.analysis.summarize results/<experiment>`. 3 seeds per method (DP → Adam: 1, deterministic).

| folder | configs | what changes vs. the final runs |
|---|---|---|
| `gran_risa_sharp`, `final_easy_sharp` | `*_sharp*.yaml` | gate penalty sharp (beta_miss 200): no bias inside gates, but Adam clips gates by mm–cm |
| `gran_risa_margin`, `final_easy_margin`, `final_trap_margin` | `*_margin*.yaml` | sharp penalty measured to a window shrunk 5 cm per side (objective.miss_margin 0.05): all runs clean |
| `gran_risa_edgewall` | `gran_risa_edgewall*.yaml` | margin penalty + sharp 3D edge-angle wall (sim3d.risk.wall_beta_edge 1146 /rad) |

Compare against the 10-seed final runs in `docs/reports/final_*`.
