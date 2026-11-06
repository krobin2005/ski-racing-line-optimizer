"""Real terrain for the 3D stage (docs/ROADMAP_3D.md): a height field from a digital terrain
model, a smoothed course centreline, and the terrain resampled into course coordinates.

Course coordinates follow the piste: s is horizontal distance along the smoothed centreline from the
race start, n is horizontal distance across it (positive to the left when facing downhill). A
point (s, n) maps to  c(s) + n * l(s),  where c is the centreline and l its unit left normal.
The mapping is valid while |n| * kappa_c(s) < 1. The centreline is smoothed until its radius stays
well above the piste half-width.

Everything here is NumPy/SciPy preprocessing that runs once per course; the simulator then reads
the resampled grid Z[s, n] (with the frame's curvature) through JAX interpolation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.ndimage import map_coordinates

from skiopt.geo import latlon_to_utm

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass
class HeightField:
    """Elevation grid; z[i, j] is the cell centred at (x0 + (j + 0.5) cell, y0 + (i + 0.5) cell)."""

    z: np.ndarray
    x0: float
    y0: float
    cell: float

    @classmethod
    def load(cls, path: str | Path) -> "HeightField":
        d = np.load(path)
        return cls(z=d["z"].astype(np.float64), x0=float(d["x0"]), y0=float(d["y0"]), cell=float(d["cell"]))

    def height(self, E, N, order: int = 3):
        E, N = np.asarray(E, float), np.asarray(N, float)
        rows = (np.atleast_1d(N) - self.y0) / self.cell - 0.5
        cols = (np.atleast_1d(E) - self.x0) / self.cell - 0.5
        z = map_coordinates(self.z, [rows.ravel(), cols.ravel()], order=order, mode="nearest")
        return z.reshape(np.broadcast_shapes(E.shape, N.shape)) if (E.ndim or N.ndim) else float(z[0])


@dataclass
class CourseFrame:
    """A smoothed centreline sampled every `ds` metres of horizontal distance, s = 0 at the start."""

    s: np.ndarray
    E: np.ndarray
    N: np.ndarray
    tx: np.ndarray       # unit tangent (downhill along the course), map plane
    ty: np.ndarray
    kappa: np.ndarray    # signed horizontal curvature of the centreline (1/m), positive = turning left

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def to_map(self, s, n):
        """(s, n) -> (E, N); n is positive to the left of the downhill direction."""
        E = np.interp(s, self.s, self.E) - n * np.interp(s, self.s, self.ty)
        N = np.interp(s, self.s, self.N) + n * np.interp(s, self.s, self.tx)
        return E, N


def osm_centreline(path: str | Path, n_points: int | None = None):
    """(E, N) of the longest OSM piste line (not the closed outline) in the file."""
    elements = json.loads(Path(path).read_text())["elements"]
    lines = [e for e in elements if e["geometry"][0] != e["geometry"][-1]]
    line = max(lines, key=lambda e: len(e["geometry"]))
    lat = [p["lat"] for p in line["geometry"]]
    lon = [p["lon"] for p in line["geometry"]]
    return latlon_to_utm(lat, lon)


def osm_outline(path: str | Path):
    """(E, N) of the closed piste outline, if the file has one."""
    elements = json.loads(Path(path).read_text())["elements"]
    closed = [e for e in elements if e["geometry"][0] == e["geometry"][-1]]
    if not closed:
        return None
    g = max(closed, key=lambda e: len(e["geometry"]))["geometry"]
    return latlon_to_utm([p["lat"] for p in g], [p["lon"] for p in g])


def build_frame(E, N, hf: HeightField, z_start: float, z_finish: float, ds: float = 1.0,
                lam: float = 1e5, before: float = 30.0, after: float = 40.0) -> tuple[CourseFrame, dict]:
    """Smooth the raw centreline (smoothing splines in x(s), y(s) with penalty `lam`), orient it
    downhill, and cut it to the race section: from where it crosses `z_start` (minus `before` m)
    to where it crosses `z_finish` (plus `after` m). Returns the frame and a report dict."""
    E, N = np.asarray(E, float), np.asarray(N, float)
    if hf.height(E[0], N[0]) < hf.height(E[-1], N[-1]):
        E, N = E[::-1], N[::-1]
    s_raw = np.r_[0.0, np.cumsum(np.hypot(np.diff(E), np.diff(N)))]
    sx, sy = make_smoothing_spline(s_raw, E, lam=lam), make_smoothing_spline(s_raw, N, lam=lam)
    s_fine = np.arange(0.0, s_raw[-1], ds)
    Ef, Nf = sx(s_fine), sy(s_fine)
    # Re-parameterize by the smoothed curve's own arc length.
    s_arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(Ef), np.diff(Nf)))]
    z = hf.height(Ef, Nf)
    i0 = int(np.argmax(z <= z_start))
    i1 = int(np.argmax(z <= z_finish))
    a, b = s_arc[i0] - before, s_arc[i1] + after
    s = np.arange(0.0, b - a, ds)
    E_s, N_s = np.interp(a + s, s_arc, Ef), np.interp(a + s, s_arc, Nf)
    dE, dN = np.gradient(E_s, ds), np.gradient(N_s, ds)
    norm = np.hypot(dE, dN)
    tx, ty = dE / norm, dN / norm
    heading = np.unwrap(np.arctan2(ty, tx))
    kappa = np.gradient(heading, ds)
    frame = CourseFrame(s=s, E=E_s, N=N_s, tx=tx, ty=ty, kappa=kappa)
    report = {
        "raw_length_m": float(s_raw[-1]), "section_start_s_raw": float(a), "section_m": float(s[-1]),
        "start_offset_m": before, "finish_offset_m": after,
        "min_centreline_radius_m": float(1.0 / np.max(np.abs(kappa))),
        "max_deviation_from_osm_m": float(_max_deviation(E, N, Ef, Nf)),
    }
    return frame, report


def _max_deviation(E, N, Ef, Nf) -> float:
    d = np.hypot(E[:, None] - Ef[None, :], N[:, None] - Nf[None, :])
    return float(d.min(axis=1).max())


@dataclass
class CourseTerrain:
    """Terrain resampled on a course-aligned grid: Z[i, j] is the height at (s_i, n_j)."""

    frame: CourseFrame
    n: np.ndarray
    Z: np.ndarray

    @property
    def ds(self) -> float:
        return float(self.frame.s[1] - self.frame.s[0])

    @property
    def dn(self) -> float:
        return float(self.n[1] - self.n[0])

    def fall_line_gradient(self) -> np.ndarray:
        """Along-course gradient on the centreline (positive downhill), as a fraction."""
        j = int(np.argmin(np.abs(self.n)))
        return -np.gradient(self.Z[:, j], self.ds)

    def cross_slope(self) -> np.ndarray:
        """dZ/dn on the course grid (positive = terrain rises to the left)."""
        return np.gradient(self.Z, self.dn, axis=1)


def resample(hf: HeightField, frame: CourseFrame, half_width: float = 30.0, dn: float = 0.5) -> CourseTerrain:
    n = np.arange(-half_width, half_width + dn / 2, dn)
    S, Nn = np.meshgrid(frame.s, n, indexing="ij")
    E, N = frame.to_map(S, Nn)
    return CourseTerrain(frame=frame, n=n, Z=hf.height(E, N))


def gran_risa(z_start: float = 1870.0, z_finish: float = 1420.0, **kwargs):
    """The Gran Risa race section from the bundled data (scripts/fetch_gran_risa.py)."""
    hf = HeightField.load(DATA / "gran_risa" / "terrain.npz")
    E, N = osm_centreline(DATA / "gran_risa" / "osm_pistes.json")
    frame, report = build_frame(E, N, hf, z_start, z_finish, **kwargs)
    return hf, frame, report
