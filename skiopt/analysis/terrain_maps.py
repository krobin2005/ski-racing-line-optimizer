"""Terrain figures for the 3D stage (docs/ROADMAP_3D.md):

    python -m skiopt.analysis.terrain_maps --out docs/reports/gran_risa_terrain

- map.png: hillshaded elevation with contours, the OSM piste outline and line, the smoothed course
  centreline, and the race start and finish.
- profile.png: along the course, elevation, fall-line gradient (%), and the cross slope at the
  centreline (camber), with the published Gran Risa figures marked.
- course_grid.png: the course-aligned grid, cross slope (camber / fall-away) and along-course
  curvature of the snow surface (positive = compression, negative = roll / crest).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LightSource, TwoSlopeNorm  # noqa: E402

from skiopt import terrain3d as T  # noqa: E402
from skiopt.analysis.plots import INK, INK_2, MUTED, SURFACE, _style  # noqa: E402

DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list(
    "camber", ["#2a78d6", "#9ec5f4", "#f0efec", "#f2a8a7", "#e34948"])  # blue <-> red, grey midpoint


def plot_map(hf, frame, outline, osm_line, start_finish):
    fig, ax = plt.subplots(figsize=(7, 10), facecolor=SURFACE)
    extent = (hf.x0, hf.x0 + hf.z.shape[1] * hf.cell, hf.y0, hf.y0 + hf.z.shape[0] * hf.cell)
    shade = LightSource(azdeg=315, altdeg=45).hillshade(hf.z, vert_exag=1.5, dx=hf.cell, dy=hf.cell)
    ax.imshow(shade, extent=extent, origin="lower", cmap="Greys_r", alpha=0.55)
    cs = ax.contour(np.linspace(extent[0], extent[1], hf.z.shape[1]), np.linspace(extent[2], extent[3], hf.z.shape[0]),
                    hf.z, levels=np.arange(1400, 2101, 25), colors=INK_2, linewidths=0.4, alpha=0.6)
    ax.clabel(cs, levels=np.arange(1400, 2101, 100), fontsize=7, fmt="%d m")
    if outline is not None:
        ax.plot(*outline, color="#2a78d6", lw=1.2, label="OSM piste outline")
    ax.plot(*osm_line, color=MUTED, lw=1, ls="--", label="OSM piste line")
    ax.plot(frame.E, frame.N, color="#e34948", lw=2.2, label="course centreline (smoothed)")
    for (E, N), name in start_finish:
        ax.plot(E, N, "o", color=INK, ms=7)
        ax.annotate(name, (E, N), xytext=(8, 0), textcoords="offset points", fontsize=9, color=INK, va="center")
    pad = 120
    ax.set_xlim(frame.E.min() - pad, frame.E.max() + pad)
    ax.set_ylim(frame.N.min() - pad, frame.N.max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("easting (m, UTM 32N)")
    ax.set_ylabel("northing (m)")
    ax.set_title("Gran Risa · terrain (South Tyrol DTM 2.5 m), 25 m contours", loc="left", fontsize=10, color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_2)
    _style(ax)
    return fig


def plot_profile(ct, start_s, finish_s):
    s = ct.frame.s
    j0 = int(np.argmin(np.abs(ct.n)))
    z = ct.Z[:, j0]
    grad = 100 * ct.fall_line_gradient()
    cross = np.degrees(np.arctan(ct.cross_slope()[:, j0]))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.12, "height_ratios": [1.3, 1, 1]})
    axes[0].plot(s, z, color=INK, lw=2)
    axes[0].set_ylabel("elevation (m)")
    for x, label in ((start_s, "start ≈ 1,870 m"), (finish_s, "finish ≈ 1,420 m")):
        for ax in axes:
            ax.axvline(x, color=MUTED, lw=1, ls="--")
        axes[0].annotate(label, (x, np.interp(x, s, z)), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK_2)
    axes[0].set_title(f"Gran Risa along the course · drop {np.interp(start_s, s, z) - np.interp(finish_s, s, z):.0f} m "
                      "(published 448 m)", loc="left", fontsize=10, color=INK)
    axes[1].plot(s, grad, color="#2a78d6", lw=1.6)
    axes[1].axhline(36, color=MUTED, lw=1, ls=":")
    axes[1].axhline(69, color=MUTED, lw=1, ls=":")
    axes[1].text(s[-1], 36, " published mean 36%", fontsize=7, color=INK_2, va="bottom", ha="right")
    axes[1].text(s[-1], 69, " published max 69%", fontsize=7, color=INK_2, va="bottom", ha="right")
    axes[1].set_ylabel("fall-line gradient (%)")
    axes[2].axhline(0, color=MUTED, lw=1)
    axes[2].plot(s, cross, color="#e34948", lw=1.6)
    axes[2].set_ylabel("cross slope at centreline (°)\n+ rises to the left")
    axes[2].set_xlabel("distance along the course s (m, horizontal)")
    for ax in axes:
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        _style(ax)
    return fig


def plot_course_grid(ct):
    s, n = ct.frame.s, ct.n
    cross = np.degrees(np.arctan(ct.cross_slope()))
    # Along-course curvature of the surface: d2z/ds2 (1/m); + = compression (concave), - = roll (convex).
    curv = np.gradient(np.gradient(ct.Z, ct.ds, axis=0), ct.ds, axis=0)
    from scipy.ndimage import uniform_filter1d
    curv = uniform_filter1d(curv, size=9, axis=0)  # 9 m window: GS-scale rolls, not grid noise
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True, facecolor=SURFACE, gridspec_kw={"hspace": 0.18})
    lim = np.nanpercentile(np.abs(cross), 98)
    m0 = axes[0].pcolormesh(s, n, cross.T, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), shading="nearest", rasterized=True)
    fig.colorbar(m0, ax=axes[0], fraction=0.02, pad=0.01).set_label("cross slope (°)\n+ rises to the left", fontsize=8, color=INK_2)
    axes[0].set_title("Cross slope in course coordinates: a turn whose outside is lower than its inside is "
                      "on fall-away", loc="left", fontsize=10, color=INK)
    clim = np.nanpercentile(np.abs(curv), 98)
    m1 = axes[1].pcolormesh(s, n, curv.T, cmap=DIVERGING, norm=TwoSlopeNorm(0, -clim, clim), shading="nearest", rasterized=True)
    fig.colorbar(m1, ax=axes[1], fraction=0.02, pad=0.01).set_label("d²z/ds² (1/m)\n+ compression, − roll", fontsize=8, color=INK_2)
    axes[1].set_title("Along-course surface curvature: rolls unload the skier, compressions add G",
                      loc="left", fontsize=10, color=INK)
    axes[1].set_xlabel("distance along the course s (m)")
    for ax in axes:
        ax.set_ylabel("across n (m)")
        ax.axhline(20, color=INK_2, lw=0.6, ls="--")
        ax.axhline(-20, color=INK_2, lw=0.6, ls="--")
        _style(ax)
    return fig


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    hf, frame, report = T.gran_risa()
    ct = T.resample(hf, frame)
    start_s, finish_s = report["start_offset_m"], frame.length - report["finish_offset_m"]
    outline = T.osm_outline(T.DATA / "gran_risa" / "osm_pistes.json")
    osm_line = T.osm_centreline(T.DATA / "gran_risa" / "osm_pistes.json")
    sf = [(frame.to_map(start_s, 0.0), "start"), (frame.to_map(finish_s, 0.0), "finish")]
    for name, fig in (("map.png", plot_map(hf, frame, outline, osm_line, sf)),
                      ("profile.png", plot_profile(ct, start_s, finish_s)),
                      ("course_grid.png", plot_course_grid(ct))):
        fig.savefig(args.out / name, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        print(f"wrote {args.out / name}")
    print({k: round(v, 1) for k, v in report.items()})


if __name__ == "__main__":
    main()
