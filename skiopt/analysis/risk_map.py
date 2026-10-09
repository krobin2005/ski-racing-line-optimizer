"""Risk map (PLAN.md §5): DNF probability of a single 90° turn across the fall line, for every
turn radius and speed, computed from configs/risk.yaml alone. Use it to tune the risk settings
before running any optimizer.

    python -m skiopt.analysis.risk_map --out figures/risk_map.png [--pitch 20] [--config my.yaml]

Panels: total, then each term (load, tight, slow; the late-line term depends on gates, so it is
not shown). Contours at 1%, 5% and 20%; the typical 18-22 m x 60-80 km/h band is outlined;
hatching marks a hard DNF (G above G_max somewhere in the turn).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from skiopt.analysis.plots import INK, INK_2, MUTED, SURFACE, _style  # noqa: E402
from skiopt.config import deep_merge, load_config, load_yaml  # noqa: E402
from skiopt.hazard import turn_risk  # noqa: E402
from skiopt.types import RiskParams  # noqa: E402

# One-hue sequential ramp (light = low risk), from the reference palette's blue steps.
RAMP = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LEVELS = [0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.2, 1.0]
LEVEL_LABELS = ["0", "0.1%", "0.25%", "0.5%", "1%", "2.5%", "5%", "20%", "100%"]


def risk_grid(r: RiskParams, pitch_deg: float, radii: np.ndarray, speeds_kmh: np.ndarray, g: float = 9.81):
    R, V = np.meshgrid(radii, speeds_kmh / 3.6, indexing="xy")
    total, per_term = turn_risk(R, V, math.radians(pitch_deg), r, g=g)
    hard_r = r._replace(load_on=0.0, tight_on=0.0, slow_on=0.0)
    hard, _ = turn_risk(R, V, math.radians(pitch_deg), hard_r, g=g)
    return np.asarray(total), np.asarray(per_term), np.asarray(hard) >= 1.0


def plot_risk_map(r: RiskParams, pitch_deg: float = 20.0, cfg_skier: dict | None = None):
    radii = np.linspace(12, 30, 145)  # the simulator clips anything tighter than 12 m
    speeds = np.linspace(25, 90, 131)
    total, per_term, hard = risk_grid(r, pitch_deg, radii, speeds)

    cmap = ListedColormap(RAMP)
    norm = BoundaryNorm(LEVELS, cmap.N)
    panels = [("total", total), ("load (G)", per_term[..., 0]), ("tight radius", per_term[..., 1]),
              ("too slow", per_term[..., 3])]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), sharey=True, facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.08})
    lo_hi = (cfg_skier or {}).get("r_typical_lo", 18.0), (cfg_skier or {}).get("r_typical_hi", 22.0)
    for ax, (name, z) in zip(axes, panels):
        mesh = ax.pcolormesh(radii, speeds, z, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
        cs = ax.contour(radii, speeds, z, levels=[0.01, 0.05, 0.2], colors=[INK_2], linewidths=0.8)
        ax.clabel(cs, fmt={0.01: "1%", 0.05: "5%", 0.2: "20%"}, fontsize=7, colors=INK_2)
        if name == "total" and hard.any():
            ax.contourf(radii, speeds, hard.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
            ax.contour(radii, speeds, hard.astype(float), levels=[0.5], colors=[INK], linewidths=1.2)
        ax.add_patch(Rectangle((lo_hi[0], 60), lo_hi[1] - lo_hi[0], 20, fill=False, ec=INK, lw=1.5, ls="--"))
        ax.set_title(name, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("turn radius (m)")
        _style(ax)
    axes[0].set_ylabel("speed (km/h)")
    axes[0].text(lo_hi[0] + 0.2, 80.8, "typical band", fontsize=7, color=INK, va="bottom")
    cb = fig.colorbar(mesh, ax=axes, fraction=0.02, pad=0.01, ticks=LEVELS)
    cb.ax.set_yticklabels(LEVEL_LABELS, fontsize=7, color=INK_2)
    cb.set_label("DNF probability for one 90° turn", fontsize=8, color=INK_2)
    cb.outline.set_visible(False)
    fig.suptitle(f"Risk map · one 90° turn across the fall line on a {pitch_deg:g}° pitch · "
                 f"hatched = hard DNF (G > {r.G_max:g} g)", x=0.125, ha="left", fontsize=11, color=INK)
    return fig


def summary_table(r: RiskParams, pitch_deg: float = 20.0) -> str:
    """Per-turn DNF probability at a few reference points, for the console."""
    points = [(22, 65), (20, 70), (18, 75), (18, 80), (16, 75), (14, 70), (13, 70), (12, 65), (20, 30)]
    lines = [f"{'radius':>7} {'speed':>7}   P(DNF) per 90° turn"]
    for rad, kmh in points:
        total, _, hard = risk_grid(r, pitch_deg, np.array([rad]), np.array([float(kmh)]))
        tag = "  (hard DNF)" if hard[0, 0] else ""
        lines.append(f"{rad:>5} m {kmh:>4} km/h   {100 * total[0, 0]:7.3f}%{tag}")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pitch", type=float, default=20.0, help="reference pitch (degrees)")
    parser.add_argument("--config", type=Path, help="YAML with `risk:` overrides (e.g. an experiment file)")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.config:
        cfg = deep_merge(cfg, load_yaml(args.config))
    r = RiskParams.from_config(cfg["risk"])
    print(summary_table(r, args.pitch))
    fig = plot_risk_map(r, args.pitch, cfg["skier"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
