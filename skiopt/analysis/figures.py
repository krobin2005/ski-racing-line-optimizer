"""Report figures from saved runs (PLAN.md §3, §10). Nothing is re-simulated.

    python -m skiopt.analysis.figures results/baseline --out figures/baseline

Writes:
- convergence.png: best finished E[score] vs. rollout-equivalents; thin lines per seed, bold median
  where every seed has a clean line (no DNF). Curves start at a seed's first clean line.
- dv_plane.png: each finished champion as a point in the distance / mean-speed plane over
  iso-time curves T = D / v (the signature figure), labelled with its DNF probability.
- turn_radii.png: average radius of every turn in each method's finished champions, 18-22 m shaded.
- course_<method>.png: the best champion line of each method on the course.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from skiopt.analysis.plots import INK, INK_2, MUTED, SURFACE, _style, plot_course, read_run  # noqa: E402
from skiopt.analysis.summarize import load_runs, method_label, turn_radii  # noqa: E402

# Colour follows the method, never its rank: fixed slots from the reference categorical palette,
# plus a marker shape per method so identity never rests on colour alone.
METHOD_STYLE = {
    "adam": ("#2a78d6", "o"),
    "cmaes": ("#eb6834", "s"),
    "ga": ("#1baf7a", "^"),
    "hybrid": ("#eda100", "D"),
    "cmaes_smooth": ("#e87ba4", "v"),
    "adam_restarts": ("#008300", "P"),
    "coach_line": ("#4a3aa7", "*"),
}
NEUTRAL = ("#8a8984", "x")
DQ = 1000.0


def style(name: str):
    return METHOD_STYLE.get(name, NEUTRAL)


def finished(m: dict) -> bool:
    return m.get("gates_missed", 0) == 0 and not m.get("hard_dnf")


def by_method(runs):
    out: dict[str, list[dict]] = {}
    for m in runs:
        out.setdefault(method_label(m), []).append(m)
    return out


def convergence(runs, ax):
    for name, ms in by_method(runs).items():
        if name == "fall_line":
            continue
        grid = np.linspace(0, max(float(m["_progress"][-1]["evals"]) for m in ms), 200)
        curves = []
        for m in ms:
            ev = np.array([float(r["evals"]) for r in m["_progress"]])
            best = np.array([float(r["best"]) for r in m["_progress"]])
            best = np.where(best < DQ, best, np.nan)
            idx = np.searchsorted(ev, grid, side="right") - 1
            curves.append(np.where(idx >= 0, best[np.clip(idx, 0, None)], np.nan))
        curves = np.array(curves)
        colour, marker = style(name)
        if np.all(np.isnan(curves)):
            ax.plot([], [], color=colour, marker=marker, lw=2, label=f"{name} (never clean)")
            continue
        for c in curves:  # each seed, thin
            ax.plot(grid, c, color=colour, lw=0.8, alpha=0.45)
        # The median only where every seed has a clean line; otherwise a seed finishing late would
        # shift the median and make a best-so-far curve appear to rise.
        all_clean = ~np.any(np.isnan(curves), axis=0)
        med = np.where(all_clean, np.median(np.where(np.isnan(curves), 0, curves), axis=0), np.nan)
        ax.plot(grid, med, color=colour, lw=2.2, label=f"{name} (n={len(ms)})")
        last = np.flatnonzero(~np.isnan(med))
        if len(last):
            ax.plot(grid[last[-1]], med[last[-1]], marker=marker, color=colour, ms=7, mec=SURFACE, mew=1.5)
    ax.set_xlabel("rollout-equivalents")
    ax.set_ylabel("best E[score] of a clean line (s)")
    ax.set_title("Convergence · thin: each seed · bold: median once every seed is clean",
                 loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    _style(ax)


def dv_plane(runs, ax):
    pts = [m for m in runs if finished(m)]
    if not pts:
        ax.text(0.5, 0.5, "no finished runs", transform=ax.transAxes, ha="center", color=INK_2)
        return
    D = np.array([m["D"] for m in pts])
    V = np.array([m["v_mean"] * 3.6 for m in pts])
    d_lo, d_hi = D.min() - 0.15 * np.ptp(D) - 3, D.max() + 0.15 * np.ptp(D) + 3
    v_lo, v_hi = V.min() - 0.15 * np.ptp(V) - 2, V.max() + 0.15 * np.ptp(V) + 2
    dd = np.linspace(d_lo, d_hi, 100)
    T_lo, T_hi = (d_lo / (v_hi / 3.6)), (d_hi / (v_lo / 3.6))
    for T in np.arange(np.floor(T_lo), np.ceil(T_hi) + 1, max(1, round((T_hi - T_lo) / 8))):
        ax.plot(dd, dd / T * 3.6, color="#e4e3df", lw=1, zorder=0)
        ax.annotate(f"{T:.0f} s", (dd[-1], dd[-1] / T * 3.6), fontsize=7, color=MUTED, va="center",
                    xytext=(2, 0), textcoords="offset points", annotation_clip=True)
    seen = set()
    for m in pts:
        name = method_label(m)
        colour, marker = style(name)
        ax.scatter(m["D"], m["v_mean"] * 3.6, color=colour, marker=marker, s=60, edgecolors=SURFACE,
                   linewidths=1, zorder=3, label=None if name in seen else name)
        seen.add(name)
        ax.annotate(f"{100 * m.get('P_DNF', 0):.0f}%", (m["D"], m["v_mean"] * 3.6), fontsize=7, color=INK_2,
                    xytext=(5, -3), textcoords="offset points")
    ax.set_xlim(d_lo, d_hi)
    ax.set_ylim(v_lo, v_hi)
    ax.set_xlabel("distance travelled D (m)")
    ax.set_ylabel("mean speed v̄ (km/h)")
    ax.set_title("Distance vs. mean speed · grey lines: equal time T = D / v̄ · labels: P_DNF",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    _style(ax)


def radius_hist(runs, fig):
    groups = {n: ms for n, ms in by_method(runs).items() if any(finished(m) for m in ms) and n != "fall_line"}
    if not groups:
        return
    axes = fig.subplots(1, len(groups), sharey=True, squeeze=False)[0]
    bins = np.arange(12, 41, 1.0)
    for ax, (name, ms) in zip(axes, groups.items()):
        radii = np.concatenate([turn_radii(m["_dir"] / "telemetry.csv") for m in ms if finished(m)])
        colour, _ = style(name)
        ax.axvspan(18, 22, color="#f0efec", zorder=0)
        ax.hist(np.clip(radii, 12, 40), bins=bins, color=colour, edgecolor=SURFACE, linewidth=1)
        share = np.mean((radii >= 18) & (radii <= 22)) if len(radii) else 0
        ax.set_title(f"{name} · {100 * share:.0f}% average 18–22 m", loc="left", fontsize=9, color=INK)
        ax.set_xlabel("turn radius (m)")
        _style(ax)
    axes[0].set_ylabel("turns")
    fig.suptitle("Average radius of each turn (arc length ÷ heading change) in finished champion lines; "
                 "40+ m drawn at 40", 
                 x=0.07, ha="left", fontsize=10, color=INK)


def make_figures(exp_dir: Path, out: Path) -> list[Path]:
    runs = load_runs(exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    convergence(runs, ax)
    written.append(out / "convergence.png")
    fig.savefig(written[-1], dpi=150, bbox_inches="tight", facecolor=SURFACE)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    dv_plane(runs, ax)
    written.append(out / "dv_plane.png")
    fig.savefig(written[-1], dpi=150, bbox_inches="tight", facecolor=SURFACE)

    fig = plt.figure(figsize=(16, 3.6), facecolor=SURFACE)
    radius_hist(runs, fig)
    written.append(out / "turn_radii.png")
    fig.savefig(written[-1], dpi=150, bbox_inches="tight", facecolor=SURFACE)

    for name, ms in by_method(runs).items():
        champ = min(ms, key=lambda m: m["best_score"])
        spec, line = read_run(champ["_dir"])
        title = (f"{name} on {spec.name} · seed {champ['seed']} · T {champ['T']:.2f} s · "
                 f"v̄ {champ['v_mean'] * 3.6:.1f} km/h · P_DNF {100 * champ.get('P_DNF', 0):.1f}%"
                 + (f" · {champ['gates_missed']} gates missed" if champ.get("gates_missed") else ""))
        fig = plot_course(spec, line=line, title=title)
        written.append(out / f"course_{name}.png")
        fig.savefig(written[-1], dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close("all")
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in make_figures(args.experiment, args.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
