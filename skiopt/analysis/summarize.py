"""Experiment summary (PLAN.md §9-10): reads saved runs only, never re-runs anything.

    python -m skiopt.analysis.summarize results/<experiment> [results/<experiment2> ...]

Writes SUMMARY.md and summary.csv next to the runs:
- headline table, one row per method, across seeds: finish rate, T, D, v_mean, P_DNF, E[score],
  evaluations to reach within 1% of the best known score, wall clock;
- pairwise Mann-Whitney U tests on the hard score (Holm-corrected) with the Vargha-Delaney A12;
- a race report for each method's champion: time, D, v_mean, max G, tightest radius, share of
  turns in the 18-22 m band, risk by cause;
- warnings when results look physically wrong.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

TYPICAL_BAND = (18.0, 22.0)
REALISM = {"T": (60.0, 80.0), "v_mean_kmh": (55.0, 70.0)}  # PLAN.md §4, for standard generated courses
P_DNF_SENSIBLE = (0.005, 0.10)


def load_runs(exp_dir: Path) -> list[dict]:
    runs = []
    for manifest in sorted(exp_dir.glob("*/seed_*/manifest.json")):
        m = json.loads(manifest.read_text())
        if not m.get("complete"):
            continue
        d = manifest.parent
        m["_dir"] = d
        with open(d / "progress.csv") as f:
            m["_progress"] = list(csv.DictReader(f))
        runs.append(m)
    return runs


def method_label(m: dict) -> str:
    if isinstance(m["method"], dict):
        return m["method"].get("label", m["method"]["name"])
    return str(m["method"])


def turn_radii(telemetry_csv: Path, min_turn_m: float = 8.0) -> np.ndarray:
    """Tightest radius of each turn: the line is split where curvature changes sign, and each
    segment longer than `min_turn_m` counts as one turn."""
    with open(telemetry_csv) as f:
        k = np.array([float(r["kappa"]) for r in csv.DictReader(f)])
    if len(k) == 0:
        return np.zeros(0)
    sign = np.sign(k)
    edges = np.flatnonzero(np.diff(sign) != 0) + 1
    radii = []
    for seg in np.split(k, edges):
        peak = np.max(np.abs(seg))
        if len(seg) >= min_turn_m and peak > 1e-3:
            radii.append(1.0 / peak)
    return np.array(radii)


def a12(x, y) -> float:
    """Vargha-Delaney A12: probability that a value from x is smaller (better) than one from y."""
    x, y = np.asarray(x), np.asarray(y)
    wins = (x[:, None] < y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()
    return float(wins / (len(x) * len(y)))


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()


def evals_to_within(progress: list[dict], target: float) -> float | None:
    for row in progress:
        if float(row["best"]) <= target:
            return float(row["evals"])
    return None


def fmt(values, scale=1.0, digits=2) -> str:
    v = np.asarray(values, dtype=float) * scale
    if len(v) == 0:
        return "–"
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


def summarize(exp_dir: Path) -> str:
    runs = load_runs(exp_dir)
    if not runs:
        raise SystemExit(f"no complete runs under {exp_dir}")
    by_method: dict[str, list[dict]] = {}
    for m in runs:
        by_method.setdefault(method_label(m), []).append(m)

    finished = [m for m in runs if m.get("gates_missed", 0) == 0 and not m.get("hard_dnf")]
    best_known = min((m["best_score"] for m in finished), default=min(m["best_score"] for m in runs))

    lines = [f"# Summary: {exp_dir.name}", ""]
    course = runs[0]["config"].get("course", {})
    lines += [f"Course: `{json.dumps(course)}` · policy `{runs[0]['config'].get('policy')}` · "
              f"best known E[score] {best_known:.3f} s", ""]

    header = ["method", "seeds", "finish rate", "E[score] (s)", "T (s)", "D (m)", "v̄ (km/h)", "P_DNF (%)",
              "evals to 1% of best", "wall clock (s)"]
    rows, csv_rows = [], []
    for name, ms in by_method.items():
        ok = [m for m in ms if m.get("gates_missed", 0) == 0 and not m.get("hard_dnf")]
        reach = [evals_to_within(m["_progress"], 1.01 * best_known) for m in ms]
        reached = [r for r in reach if r is not None]
        row = [name, str(len(ms)), f"{len(ok)}/{len(ms)}",
               fmt([m["best_score"] for m in ok]) if ok else "DQ",
               fmt([m["T"] for m in ok]), fmt([m["D"] for m in ok], digits=1),
               fmt([m["v_mean"] for m in ok], 3.6, 1), fmt([m["P_DNF"] for m in ok], 100, 1),
               f"{np.median(reached):.0f} ({len(reached)}/{len(ms)})" if reached else "never",
               fmt([m["wall_clock_s"] for m in ms], digits=0)]
        rows.append(row)
        for m in ms:
            csv_rows.append({"method": name, "seed": m["seed"], "best_score": m["best_score"], "T": m["T"],
                             "D": m["D"], "v_mean_kmh": m["v_mean"] * 3.6, "P_DNF": m.get("P_DNF"),
                             "gates_missed": m.get("gates_missed"), "hard_dnf": m.get("hard_dnf"),
                             "evals": m.get("evals"), "wall_clock_s": m["wall_clock_s"]})

    lines += ["## Headline", "", "Mean ± std over seeds, finished runs only (no missed gate, no hard DNF).", ""]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]

    names = [n for n, ms in by_method.items() if len(ms) >= 2]
    if len(names) >= 2:
        pairs = list(itertools.combinations(names, 2))
        raw, a12s = [], []
        for a, b in pairs:
            xa = [m["best_score"] for m in by_method[a]]
            xb = [m["best_score"] for m in by_method[b]]
            raw.append(float(mannwhitneyu(xa, xb, alternative="two-sided").pvalue))
            a12s.append(a12(xa, xb))
        adj = holm(raw)
        lines += ["", "## Pairwise tests (hard score, lower is better)", "",
                  "Mann–Whitney U, Holm-corrected. A12 = chance the first method beats the second.", "",
                  "| first | second | A12 | p (Holm) |", "|---|---|---|---|"]
        lines += [f"| {a} | {b} | {x:.2f} | {p:.3g} |" for (a, b), x, p in zip(pairs, a12s, adj)]

    lines += ["", "## Race reports (each method's best run)", ""]
    warnings = []
    for name, ms in by_method.items():
        champ = min(ms, key=lambda m: m["best_score"])
        radii = turn_radii(champ["_dir"] / "telemetry.csv")
        in_band = float(np.mean((radii >= TYPICAL_BAND[0]) & (radii <= TYPICAL_BAND[1]))) if len(radii) else 0.0
        causes = champ.get("H_by_cause", {})
        total_h = sum(causes.values()) or 1.0
        cause_txt = ", ".join(f"{k} {100 * v / total_h:.0f}%" for k, v in causes.items() if v / total_h > 0.005)
        lines.append(
            f"- **{name}** (seed {champ['seed']}): T {champ['T']:.2f} s · D {champ['D']:.1f} m · "
            f"v̄ {champ['v_mean'] * 3.6:.1f} km/h · max G {champ['max_G']:.2f} · "
            f"tightest turn {radii.min() if len(radii) else float('nan'):.1f} m · "
            f"turns in 18–22 m: {100 * in_band:.0f}% · P_DNF {100 * champ.get('P_DNF', 0):.1f}%"
            + (f" (risk from {cause_txt})" if cause_txt else "")
            + (f" · **{champ['gates_missed']} gates missed**" if champ.get("gates_missed") else "")
        )
        if champ.get("gates_missed", 0) == 0:
            if in_band < 0.5:
                warnings.append(f"{name}: only {100 * in_band:.0f}% of turns in the 18–22 m band")
            if len(radii) and radii.min() <= 12.05:
                warnings.append(f"{name}: champion sits at the 12 m curvature clip")
            if champ["max_G"] >= 3.95:
                warnings.append(f"{name}: champion sits at G_max")
            v = champ["v_mean"] * 3.6
            if not REALISM["v_mean_kmh"][0] <= v <= REALISM["v_mean_kmh"][1]:
                warnings.append(f"{name}: mean speed {v:.1f} km/h outside {REALISM['v_mean_kmh']}")
            if not P_DNF_SENSIBLE[0] <= champ.get("P_DNF", 0) <= 0.30:
                warnings.append(f"{name}: P_DNF {100 * champ.get('P_DNF', 0):.1f}% outside the 0.5–30% range")
    if warnings:
        lines += ["", "## Warnings", ""] + [f"- ⚠ {w}" for w in warnings]

    out = "\n".join(lines) + "\n"
    (exp_dir / "SUMMARY.md").write_text(out)
    with open(exp_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="+", type=Path)
    args = parser.parse_args(argv)
    for exp in args.experiments:
        print(summarize(exp))


if __name__ == "__main__":
    main()
