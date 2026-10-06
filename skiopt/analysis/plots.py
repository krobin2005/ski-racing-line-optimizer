"""Course plot (PLAN.md §7): the slope plane from above, with red and blue GS gates, piste edges,
start and finish, an optional racing line coloured by speed, and the pitch profile underneath.

The course runs left to right (y, down the fall line) and the across-slope axis (x) is drawn
exaggerated so 40 m of piste stays readable next to ~1 km of course.

    python -m skiopt.analysis.plots --library easy --out figures/easy.png
    python -m skiopt.analysis.plots --generated 3 --out figures/gen_3.png
    python -m skiopt.analysis.plots --run results/smoke/fall_line/seed_0 --out figures/run.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from skiopt import courses as course_lib  # noqa: E402
from skiopt.courses import CourseSpec  # noqa: E402

# FIS gate colours are fixed by the rules (ICR 901.2.2), so they are domain colours, not a palette.
GATE_COLOURS = {"red": "#d1342f", "blue": "#2a78d6"}
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#a3a29c"
SURFACE = "#fcfcfb"
# Speed is a magnitude on top of red and blue gates, so it gets one neutral ramp (light = slow).
SPEED_CMAP = LinearSegmentedColormap.from_list("speed", ["#c9c8c2", "#0b0b0b"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def plot_course(spec: CourseSpec, line: dict | None = None, missed: np.ndarray | None = None,
                title: str | None = None):
    """Draw a course. `line` is optional telemetry: dict with arrays 'y', 'x' and 'v' (m/s).
    `missed` is an optional boolean array marking missed gates. Returns the figure."""
    fig, (ax, ax_p) = plt.subplots(
        2, 1, figsize=(14, 5.2), sharex=True, gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08},
        facecolor=SURFACE,
    )
    half = spec.piste_width / 2

    # Piste, start and finish.
    ax.axhspan(-half, half, color="#f0efec", zorder=0)
    for edge in (-half, half):
        ax.axhline(edge, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    for y0, label in ((0.0, "start"), (spec.length, "finish")):
        ax.axvline(y0, color=INK_2, lw=1.5, zorder=1)
        ax.text(y0, half + 1.5, label, ha="center", va="bottom", fontsize=8, color=INK_2)

    # Gates: panel between turning and outside pole, turning pole as a solid dot.
    for i in range(spec.n_gates):
        c = GATE_COLOURS.get(spec.gate_colour[i], INK)
        y, x_in, x_out = spec.gate_y[i], spec.gate_x[i], spec.outside_x[i]
        ax.plot([y, y], [x_in, x_out], color=c, lw=3, alpha=0.35, solid_capstyle="round", zorder=2)
        ax.plot([y], [x_out], "o", ms=3.5, color=c, mfc=SURFACE, mew=1.2, zorder=3)
        ax.plot([y], [x_in], "o", ms=4.5, color=c, zorder=3)
        if i > 0 and spec.gate_side[i] == spec.gate_side[i - 1]:
            ax.annotate("delay", (y, x_out), xytext=(0, 6 * np.sign(x_out)), textcoords="offset points",
                        ha="center", va="center", fontsize=7, color=INK_2)

    if line is not None:
        pts = np.column_stack([line["y"], line["x"]])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        speed = np.asarray(line["v"]) * 3.6
        lc = LineCollection(segs, cmap=SPEED_CMAP, linewidths=2, zorder=4, capstyle="round")
        lc.set_array(0.5 * (speed[:-1] + speed[1:]))
        ax.add_collection(lc)
        cb = fig.colorbar(lc, ax=[ax, ax_p], fraction=0.015, pad=0.01)
        cb.set_label("speed (km/h)", color=INK_2, fontsize=8)
        cb.ax.tick_params(labelsize=8, colors=INK_2)
        cb.outline.set_visible(False)

    if missed is not None and np.any(missed):
        ax.plot(spec.gate_y[missed], spec.gate_x[missed], "x", ms=9, mew=2, color=INK, zorder=5)

    ax.set_ylim(-half - 4, half + 6)
    ax.set_ylabel("across slope x (m)")
    _style(ax)
    n_delay = spec.n_gates - spec.direction_changes
    ax.set_title(
        title or f"{spec.name}  ·  {spec.length:.0f} m  ·  vertical drop {spec.vertical_drop:.0f} m  ·  "
                 f"{spec.n_gates} gates ({spec.direction_changes} direction changes, {n_delay} delay)",
        loc="left", fontsize=10, color=INK, pad=14,
    )

    handles = [
        Line2D([], [], color=GATE_COLOURS["red"], marker="o", lw=3, alpha=0.6, label="red gate"),
        Line2D([], [], color=GATE_COLOURS["blue"], marker="o", lw=3, alpha=0.6, label="blue gate"),
        Line2D([], [], color=MUTED, ls=(0, (4, 3)), label="piste edge"),
    ]
    if line is not None:
        handles.append(Line2D([], [], color=INK, lw=2, label="racing line"))
    if missed is not None and np.any(missed):
        handles.append(Line2D([], [], color=INK, marker="x", lw=0, mew=2, label="missed gate"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(handles),
              frameon=False, fontsize=8, labelcolor=INK_2, borderaxespad=0.2)

    # Pitch profile.
    ys = np.linspace(0, spec.length, 400)
    ax_p.plot(ys, np.degrees(spec.pitch_at(ys)), color=INK, lw=1.5)
    ax_p.set_ylabel("pitch (°)")
    ax_p.set_xlabel("down the fall line y (m)")
    lo, hi = np.degrees(spec.theta_knots).min(), np.degrees(spec.theta_knots).max()
    ax_p.set_ylim(max(0, lo - 4), hi + 4)
    ax_p.grid(axis="y", color="#e4e3df", lw=0.8)
    _style(ax_p)
    ax_p.set_xlim(-10, spec.length + 10)
    return fig


def read_run(run_dir: Path) -> tuple[CourseSpec, dict]:
    """Course spec and champion telemetry from a saved run directory."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    spec = course_lib.spec_from_config(manifest["config"]["course"])
    with open(run_dir / "telemetry.csv") as f:
        rows = list(csv.DictReader(f))
    line = {k: np.array([float(r[k]) for r in rows]) for k in ("x", "y", "v")}
    return spec, line


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--library", help="hand-built course name, e.g. easy")
    src.add_argument("--generated", type=int, help="generator seed")
    src.add_argument("--run", type=Path, help="saved run directory (plots the champion line)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    line = None
    if args.library:
        spec = course_lib.LIBRARY[args.library]()
    elif args.generated is not None:
        spec = course_lib.generate(args.generated)
    else:
        spec, line = read_run(args.run)
    fig = plot_course(spec, line=line)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
