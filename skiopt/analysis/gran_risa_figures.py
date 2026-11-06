"""Gran Risa figures from saved 3D runs (docs/ROADMAP_3D.md). Nothing is re-simulated.

    python -m skiopt.analysis.gran_risa_figures results/gran_risa --out docs/reports/gran_risa

- line_on_map_<method>.png: each method's best line on the real terrain (hillshade + contours), with
  the piste outline and the red/blue gates, coloured by speed.
- run_trace.png: along the course for every method's best line: speed, G, edge angle and normal load,
  with the 3D risk thresholds (boot-out ~70 deg, edge DNF 78 deg, G_max) marked.
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
from matplotlib.colors import LightSource  # noqa: E402

from skiopt import terrain3d as T  # noqa: E402
from skiopt.analysis.figures import style as method_style  # noqa: E402
from skiopt.analysis.plots import GATE_COLOURS, INK, INK_2, MUTED, SPEED_CMAP, SURFACE, _style  # noqa: E402
from skiopt.analysis.summarize import load_runs, method_label  # noqa: E402
from skiopt.course3d import build_gran_risa  # noqa: E402


def _telemetry(run_dir: Path) -> dict:
    with open(run_dir / "telemetry.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def line_on_map(build, tel, title):
    hf = T.HeightField.load(T.DATA / "gran_risa" / "terrain.npz")
    frame = build.frame
    fig, ax = plt.subplots(figsize=(7.5, 10.5), facecolor=SURFACE)
    extent = (hf.x0, hf.x0 + hf.z.shape[1] * hf.cell, hf.y0, hf.y0 + hf.z.shape[0] * hf.cell)
    shade = LightSource(azdeg=315, altdeg=45).hillshade(hf.z, vert_exag=1.5, dx=hf.cell, dy=hf.cell)
    ax.imshow(shade, extent=extent, origin="lower", cmap="Greys_r", alpha=0.45)
    xs = np.linspace(extent[0], extent[1], hf.z.shape[1])
    ys = np.linspace(extent[2], extent[3], hf.z.shape[0])
    cs = ax.contour(xs, ys, hf.z, levels=np.arange(1400, 1901, 25), colors=INK_2, linewidths=0.35, alpha=0.5)
    ax.clabel(cs, levels=np.arange(1400, 1901, 100), fontsize=7, fmt="%d m")
    outline = T.osm_outline(T.DATA / "gran_risa" / "osm_pistes.json")
    if outline is not None:
        ax.plot(*outline, color=MUTED, lw=1.0)
    spec = build.spec
    for i in range(spec.n_gates):
        c = GATE_COLOURS.get(spec.gate_colour[i], INK)
        e0, n0 = frame.to_map(spec.gate_y[i], spec.gate_x[i])
        e1, n1 = frame.to_map(spec.gate_y[i], spec.outside_x[i])
        ax.plot([e0, e1], [n0, n1], color=c, lw=2.2, alpha=0.6, solid_capstyle="round", zorder=3)
        ax.plot([e0], [n0], "o", color=c, ms=3, zorder=4)
    E, N = frame.to_map(tel["y"], tel["x"])
    pts = np.column_stack([E, N])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    speed = tel["v"] * 3.6
    lc = LineCollection(segs, cmap=SPEED_CMAP, linewidths=2.0, zorder=5)
    lc.set_array(0.5 * (speed[:-1] + speed[1:]))
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("speed (km/h)", fontsize=8, color=INK_2)
    cb.outline.set_visible(False)
    for (s, label) in ((0.0, "start 1,868 m"), (float(spec.length), "finish 1,420 m")):
        e, n = frame.to_map(s, float(np.interp(s, tel["y"], tel["x"])))
        ax.plot(e, n, "s", color=INK, ms=6, zorder=6)
        ax.annotate(label, (e, n), xytext=(8, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
    pad = 60
    ax.set_xlim(frame.E.min() - pad, frame.E.max() + pad)
    ax.set_ylim(frame.N.min() - pad, frame.N.max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("easting (m, UTM 32N)")
    ax.set_ylabel("northing (m)")
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    _style(ax)
    return fig


def run_trace(champions: dict, p3_edge_safe=66.0, edge_dnf=78.0, g_max=4.0):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, facecolor=SURFACE, gridspec_kw={"hspace": 0.12})
    for name, tel in champions.items():
        colour, _ = method_style(name)
        s = tel["y"]
        axes[0].plot(s, tel["v"] * 3.6, color=colour, lw=1.4, label=name)
        axes[1].plot(s, tel["G"], color=colour, lw=1.0)
        axes[2].plot(s, np.degrees(tel["edge"]), color=colour, lw=1.0)
        axes[3].plot(s, tel["N_load"] / 9.81, color=colour, lw=1.0)
    axes[0].set_ylabel("speed (km/h)")
    axes[1].set_ylabel("total load (g)")
    axes[1].axhline(g_max, color=INK, lw=1, ls="--")
    axes[1].text(0, g_max, " G_max (DNF)", fontsize=7, color=INK_2, va="bottom")
    axes[2].set_ylabel("edge angle (°)")
    for y, label in ((p3_edge_safe, " boot-out risk starts"), (70.0, " boot-out ≈ 70° (49 mm stack)"), (edge_dnf, " DNF")):
        axes[2].axhline(y, color=MUTED if y != edge_dnf else INK, lw=1, ls="--")
        axes[2].text(0, y, label, fontsize=7, color=INK_2, va="bottom")
    axes[3].set_ylabel("normal load (g)\n< 0: airborne")
    axes[3].axhline(0, color=INK, lw=1, ls="--")
    axes[3].set_xlabel("distance along the course s (m, horizontal)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, ncol=4, loc="lower center")
    axes[0].set_title("Gran Risa · best line of each method along the course", loc="left", fontsize=10, color=INK)
    for ax in axes:
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        _style(ax)
    return fig


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.experiment)
    cfg = runs[0]["config"]
    build = build_gran_risa(seed=int(cfg["course"].get("seed", 0)),
                            smooth_m=float(cfg.get("sim3d", {}).get("terrain_smooth_m", 8.0)))
    champions = {}
    for name in sorted({method_label(m) for m in runs}):
        ms = [m for m in runs if method_label(m) == name]
        champ = min(ms, key=lambda m: m["best_score"])
        tel = _telemetry(champ["_dir"])
        champions[name] = tel
        title = (f"{name} · seed {champ['seed']} · T {champ['T']:.2f} s (real fastest run 1: 76.00 s) · "
                 f"v̄ {champ['v_mean'] * 3.6:.1f} km/h · P_DNF {100 * champ.get('P_DNF', 0):.1f}%"
                 + (" · DNF" if champ.get("gates_missed") or champ.get("hard_dnf") else ""))
        fig = line_on_map(build, tel, title)
        fig.savefig(args.out / f"line_on_map_{name}.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        print(f"wrote {args.out / f'line_on_map_{name}.png'}")
    fig = run_trace(champions)
    fig.savefig(args.out / "run_trace.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {args.out / 'run_trace.png'}")
    (args.out / "course_set.json").write_text(json.dumps(build.meta, indent=1, default=str))


if __name__ == "__main__":
    main()
