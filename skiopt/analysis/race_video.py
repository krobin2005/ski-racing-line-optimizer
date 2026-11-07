"""Broadcast-style race video on the real 3D Gran Risa: every method's best line raced head to head.

    python -m skiopt.analysis.race_video results/gran_risa --out docs/media/gran_risa_race.mp4
    python -m skiopt.analysis.race_video results/gran_risa --out /tmp/still.png --still 30   # one frame at t = 30 s

Nothing is re-simulated. Each racer replays the saved telemetry of its method's best run (position along
the course s, offset n, speed, G, edge angle at every step), placed on the real terrain (South Tyrol DTM,
2.5 m). All racers start together, so the distance between them on screen is their time gap.

What is on screen:
- the terrain, piste (snow), forest, safety nets, and the FIS gate set (red/blue poles and panels);
- one skier per method, leaning into each turn by its edge angle, with a short trail;
- a "Schwarz pace" ghost: the fastest line run at Marco Schwarz's average pace (76.00 s, the real fastest
  run 1 on 21 Dec 2025). His real line is not public, so this shows his time, not his line;
- TV graphics: race clock, live standings with gaps, the leader's speed / G / edge angle, a course map,
  Schwarz's official split times as they come up, and a final results board with the real top 5.

Cameras (--camera): `broadcast` (default) cuts between a chase camera, a side camera, a drone shot and a
finish camera, like a TV feed; `chase` and `overhead` hold one camera for the whole run.
Rendering uses PyVista (VTK) off screen; video encoding uses imageio-ffmpeg. Both are optional extras:
    pip install pyvista imageio-ffmpeg
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from matplotlib.path import Path as MplPath
from PIL import Image, ImageDraw, ImageFont

from skiopt import terrain3d as T
from skiopt.analysis.summarize import load_runs, method_label
from skiopt.course3d import build_gran_risa

# Display names and suit colours (bright, distinct on snow). Order = default lineup.
RACERS = {
    "cmaes": ("CMA-ES", "#ff3b30"),
    "hybrid": ("CMA-ES > Adam", "#ff9500"),
    "ga": ("Genetic alg.", "#34c759"),
    "dp": ("DP grid", "#af52de"),
    "dp_adam": ("DP > Adam", "#5856d6"),
    "adam": ("Adam", "#007aff"),
    "cmaes_smooth": ("CMA-ES smooth", "#ff2d55"),
    "coach_line": ("Coach line", "#8e8e93"),
}
PACE_COLOUR = "#f2f2f2"
RACE = json.loads((T.DATA / "gran_risa" / "race_2025_run1.json").read_text())
SCHWARZ_T = float(RACE["fastest_run1"]["finish_s"])
SPLITS = [("INT 1", RACE["fastest_run1"]["int1_s"]), ("INT 2", RACE["fastest_run1"]["int2_s"]),
          ("INT 3", RACE["fastest_run1"]["int3_s"])]
BOOT_OUT_DEG = 75.7
INTRO_S, OUTRO_S = 4.0, 8.0


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


@dataclass
class Racer:
    key: str
    name: str
    colour: str
    t: np.ndarray        # time at each step (s)
    s: np.ndarray        # distance along the course (m)
    E: np.ndarray        # map position (m, UTM 32N), on the course frame
    N: np.ndarray
    Z: np.ndarray        # ground height under the skier (m)
    v: np.ndarray
    G: np.ndarray
    edge: np.ndarray     # edge angle (rad)
    kappa: np.ndarray    # signed curvature, > 0 = turning left
    T: float             # finish time

    def at(self, t: float) -> dict:
        """State at race time t (clamped to the finish)."""
        tc = min(max(t, 0.0), self.T)
        g = lambda a: float(np.interp(tc, self.t, a))  # noqa: E731
        # heading from a short look-ahead along the line
        dt = 0.15
        E1, N1 = float(np.interp(min(tc + dt, self.T), self.t, self.E)), float(np.interp(min(tc + dt, self.T), self.t, self.N))
        E0, N0 = float(np.interp(max(tc - dt, 0), self.t, self.E)), float(np.interp(max(tc - dt, 0), self.t, self.N))
        Z1, Z0 = float(np.interp(min(tc + dt, self.T), self.t, self.Z)), float(np.interp(max(tc - dt, 0), self.t, self.Z))
        h = np.array([E1 - E0, N1 - N0, Z1 - Z0])
        return {"E": g(self.E), "N": g(self.N), "Z": g(self.Z), "s": g(self.s), "v": g(self.v), "G": g(self.G),
                "edge": g(self.edge), "kappa": g(self.kappa), "heading": h / (np.linalg.norm(h) + 1e-9),
                "finished": t >= self.T}

    def time_at_s(self, s: float) -> float:
        return float(np.interp(s, self.s, self.t))


def _telemetry(run_dir: Path) -> dict:
    with open(run_dir / "telemetry.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def load_racers(results: Path, methods: list[str] | None, build, hf) -> list[Racer]:
    runs = load_runs(results)
    racers = []
    names = methods or [m for m in RACERS if any(method_label(r) == m for r in runs)]
    for key in names:
        ms = [r for r in runs if method_label(r) == key]
        if not ms:
            print(f"no runs for {key!r} in {results}, skipped")
            continue
        champ = min(ms, key=lambda r: r["best_score"])
        tel = _telemetry(champ["_dir"])
        E, N = build.frame.to_map(tel["y"], tel["x"])
        t = tel["t"] + tel["dt"]  # time at the end of each step
        t = np.concatenate([[0.0], t])
        pad = lambda a: np.concatenate([[a[0]], a])  # noqa: E731
        name, colour = RACERS.get(key, (key, "#cccccc"))
        racers.append(Racer(key, name, colour, t, pad(tel["y"]), pad(E), pad(N), hf.height(pad(E), pad(N)),
                            pad(tel["v"]), pad(tel["G"]), pad(tel["edge"]), pad(tel["kappa"]), float(champ["T"])))
    racers.sort(key=lambda r: r.T)
    return racers


def pace_ghost(leader: Racer) -> Racer:
    """The leader's line, run at Schwarz's average pace (time scaled so it finishes in 76.00 s)."""
    k = SCHWARZ_T / leader.T
    return Racer("schwarz_pace", "Schwarz pace", PACE_COLOUR, leader.t * k, leader.s, leader.E, leader.N, leader.Z,
                 leader.v / k, leader.G, leader.edge, leader.kappa, SCHWARZ_T)


# ----------------------------------------------------------------------------------------------- scene


class Scene:
    def __init__(self, build, hf, racers: list[Racer], size=(1920, 1080)):
        import pyvista as pv

        self.pv = pv
        self.build, self.hf, self.racers = build, hf, racers
        f = build.frame
        self.origin = np.array([f.E[0], f.N[0], hf.height(f.E[0], f.N[0])])
        self.p = pv.Plotter(off_screen=True, window_size=size)
        self.p.set_background("#c9dff2", top="#4a86c5")
        self._terrain()
        self._trees()
        self._nets()
        self._gates()
        self._finish_arch()
        self.actors = {r.key: self._skier(r) for r in racers}
        self.trails = {}
        self.p.camera.view_angle = 42.0
        self.p.camera.clipping_range = (0.5, 6000)

    def local(self, E, N, Z):
        return np.stack([np.asarray(E) - self.origin[0], np.asarray(N) - self.origin[1],
                         np.asarray(Z) - self.origin[2]], axis=-1)

    def _terrain(self):
        hf, f = self.hf, self.build.frame
        pad = 260.0
        e0, e1 = f.E.min() - pad, f.E.max() + pad
        n0, n1 = f.N.min() - pad, f.N.max() + pad
        j0, j1 = int(max((e0 - hf.x0) / hf.cell, 0)), int(min((e1 - hf.x0) / hf.cell, hf.z.shape[1]))
        i0, i1 = int(max((n0 - hf.y0) / hf.cell, 0)), int(min((n1 - hf.y0) / hf.cell, hf.z.shape[0]))
        xs = hf.x0 + (np.arange(j0, j1) + 0.5) * hf.cell
        ys = hf.y0 + (np.arange(i0, i1) + 0.5) * hf.cell
        X, Y = np.meshgrid(xs, ys)
        Z = hf.z[i0:i1, j0:j1]
        L = self.local(X, Y, Z)
        grid = self.pv.StructuredGrid(L[..., 0], L[..., 1], L[..., 2])
        outline = T.osm_outline(T.DATA / "gran_risa" / "osm_pistes.json")
        self.outline = outline
        on_piste = MplPath(np.column_stack(outline)).contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
        # Soft edge: blur the mask a little so the piste border is not a staircase.
        from scipy.ndimage import gaussian_filter

        m = gaussian_filter(on_piste.astype(float), 1.0)
        rng = np.random.default_rng(0)
        grain = rng.normal(0, 1, X.shape)
        snow = np.stack([0.95 + 0.025 * grain, 0.965 + 0.025 * grain, 1.0 + 0.0 * grain], -1)
        forest_snow = np.stack([0.80 + 0.05 * grain, 0.83 + 0.05 * grain, 0.86 + 0.05 * grain], -1)
        rgb = m[..., None] * snow + (1 - m[..., None]) * forest_snow
        self.forest_mask = (X, Y, ~on_piste)
        grid.point_data["rgb"] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8).reshape(-1, 3, order="F")
        self.p.add_mesh(grid, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.35, diffuse=0.75, specular=0.05)

    def _trees(self):
        X, Y, off = self.forest_mask
        Eo, No = self.outline
        from scipy.spatial import cKDTree

        tree = cKDTree(np.column_stack([Eo, No]))
        rng = np.random.default_rng(1)
        cand = np.column_stack([X[off], Y[off]])
        cand = cand[rng.random(len(cand)) < 0.22]
        cand = cand + rng.uniform(-1.2, 1.2, cand.shape)
        d, _ = tree.query(cand)
        keep = (d > 6) & (d < 140) & (rng.random(len(cand)) < np.clip(1.1 - d / 160, 0.2, 1.0))
        # the finish area is an open stadium, not forest
        fe, fn = self.build.frame.to_map(self.build.frame.length, 0.0)
        keep &= np.hypot(cand[:, 0] - fe, cand[:, 1] - fn) > 90
        cand = cand[keep]
        if not len(cand):
            return
        Z = self.hf.height(cand[:, 0], cand[:, 1])
        heights = rng.uniform(9, 18, len(cand))
        pts = self.local(cand[:, 0], cand[:, 1], Z + heights * 0.5)
        cloud = self.pv.PolyData(pts)
        cloud["h"] = heights
        cone = self.pv.Cone(direction=(0, 0, 1), height=1.0, radius=0.28, resolution=8)
        glyphs = cloud.glyph(geom=cone, scale="h", orient=False)
        shade = rng.uniform(0.75, 1.0, glyphs.n_points)
        base = np.array([0.13, 0.25, 0.17])
        glyphs.point_data["rgb"] = (np.clip(base[None, :] * shade[:, None] * 1.4, 0, 1) * 255).astype(np.uint8)
        self.p.add_mesh(glyphs, scalars="rgb", rgb=True, ambient=0.3, diffuse=0.8)

    def _nets(self):
        """Red B-nets along the piste outline, 2 m tall, a few metres outside the edge."""
        f = self.build.frame
        for side_lo in (True, False):
            edge = self.build.piste_lo if side_lo else self.build.piste_hi
            s = f.s[(f.s >= -5) & (f.s <= f.length + 5)]
            n = np.interp(s, f.s, edge) + (-2.0 if side_lo else 2.0)
            E, N = f.to_map(s, n)
            Z = self.hf.height(E, N)
            bottom = self.local(E, N, Z)
            top = self.local(E, N, Z + 2.0)
            pts = np.vstack([bottom, top])
            k = len(s)
            faces = np.hstack([[4, i, i + 1, k + i + 1, k + i] for i in range(k - 1)])
            self.p.add_mesh(self.pv.PolyData(pts, faces=faces), color="#d62d20", opacity=0.75, ambient=0.5)

    def _gates(self):
        spec, f = self.build.spec, self.build.frame
        meshes = {"red": [], "blue": []}
        for i in range(spec.n_gates):
            colour = spec.gate_colour[i]
            s = float(spec.gate_y[i])
            inside, outside = float(spec.gate_x[i]), float(spec.outside_x[i])
            direction = np.sign(outside - inside) or 1.0
            for n_pole, toward in ((inside, direction), (outside, -direction)):
                for dn in (0.0, 0.75 * toward):
                    E, N = f.to_map(s, n_pole + dn)
                    base = self.local(E, N, self.hf.height(E, N))
                    meshes[colour].append(self.pv.Cylinder(center=base + [0, 0, 0.95], direction=(0, 0, 1),
                                                           radius=0.035, height=1.9, resolution=8))
                Ea, Na = f.to_map(s, n_pole)
                Eb, Nb = f.to_map(s, n_pole + 0.75 * toward)
                a = self.local(Ea, Na, self.hf.height(Ea, Na) + 1.2)
                b = self.local(Eb, Nb, self.hf.height(Eb, Nb) + 1.2)
                panel = self.pv.Plane(center=(a + b) / 2 + [0, 0, 0.3], direction=np.cross(b - a, [0, 0, 1]),
                                      i_size=0.6, j_size=np.linalg.norm(b - a), i_resolution=1, j_resolution=1)
                meshes[colour].append(panel)
        for colour, ms in meshes.items():
            if ms:
                hexc = {"red": "#e0201b", "blue": "#1a5fd6"}[colour]
                self.p.add_mesh(self.pv.merge(ms), color=hexc, ambient=0.45, diffuse=0.7)

    def _finish_arch(self):
        f, b = self.build.frame, self.build
        for s, colour in ((0.0, "#1c1c1e"), (f.length, "#d62d20")):
            lo, hi = float(np.interp(s, f.s, b.piste_lo)), float(np.interp(s, f.s, b.piste_hi))
            posts = []
            for n in (lo, hi):
                E, N = f.to_map(s, n)
                base = self.local(E, N, self.hf.height(E, N))
                posts.append(base)
                self.p.add_mesh(self.pv.Cylinder(center=base + [0, 0, 3.5], direction=(0, 0, 1), radius=0.35, height=7),
                                color=colour)
            a, c = posts
            top = max(a[2], c[2]) + 6.0
            banner = self.pv.Plane(center=[(a[0] + c[0]) / 2, (a[1] + c[1]) / 2, top],
                                   direction=np.cross(c - a, [0, 0, 1]), i_size=1.6, j_size=np.linalg.norm(c - a))
            self.p.add_mesh(banner, color=colour, ambient=0.6)

    def _skier(self, r: Racer):
        pv = self.pv
        parts = [
            pv.Box(bounds=(-0.95, 0.98, 0.10, 0.17, 0.0, 0.04)),     # skis, 193 cm
            pv.Box(bounds=(-0.95, 0.98, -0.17, -0.10, 0.0, 0.04)),
            pv.Cylinder(center=(0.05, 0.13, 0.35), direction=(0.25, 0, 1), radius=0.08, height=0.7),  # legs
            pv.Cylinder(center=(0.05, -0.13, 0.35), direction=(0.25, 0, 1), radius=0.08, height=0.7),
            pv.Cylinder(center=(0.22, 0.0, 0.92), direction=(0.9, 0, 1), radius=0.19, height=0.75),   # tucked torso
            pv.Sphere(center=(0.52, 0.0, 1.25), radius=0.15),                                          # helmet
        ]
        body = pv.merge(parts)
        body.points *= 1.35  # a touch larger than life so a 2 m racer reads from a TV camera distance
        opacity = 0.55 if r.key == "schwarz_pace" else 1.0
        return self.p.add_mesh(body, color=r.colour, smooth_shading=True, ambient=0.4, opacity=opacity)

    def place(self, r: Racer, st: dict):
        h = st["heading"]
        hz = np.array([h[0], h[1], 0.0])
        hz /= np.linalg.norm(hz) + 1e-9
        centre_side = np.array([-hz[1], hz[0], 0.0]) * np.sign(st["kappa"] or 1.0)
        # a racer angulates: the body leans less than the skis are edged
        e = 0.65 * min(abs(st["edge"]), np.radians(80))
        up = np.cos(e) * np.array([0, 0, 1.0]) + np.sin(e) * centre_side
        fwd0 = h
        left = np.cross(up, fwd0)
        left /= np.linalg.norm(left) + 1e-9
        fwd = np.cross(left, up)
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2] = fwd, left, up
        M[:3, 3] = self.local(st["E"], st["N"], st["Z"] + 0.03)
        self.actors[r.key].user_matrix = M
        # hide a skier the camera is about to pass through
        cam = np.asarray(self.p.camera.position)
        self.actors[r.key].SetVisibility(bool(np.linalg.norm(M[:3, 3] - cam) > 7.0))

    def trail(self, r: Racer, t: float, seconds: float = 2.5):
        if r.key in self.trails:
            self.p.remove_actor(self.trails[r.key])
        tt = np.linspace(max(t - seconds, 0), min(t, r.T), 40)
        if tt[-1] - tt[0] < 0.05:
            return
        pts = self.local(np.interp(tt, r.t, r.E), np.interp(tt, r.t, r.N), np.interp(tt, r.t, r.Z) + 0.05)
        line = self.pv.lines_from_points(pts)
        self.trails[r.key] = self.p.add_mesh(line, color=r.colour, line_width=4, opacity=0.8)

    def to_screen(self, xyz) -> tuple[float, float] | None:
        ren = self.p.renderer
        ren.SetWorldPoint(*xyz, 1.0)
        ren.WorldToDisplay()
        x, y, z = ren.GetDisplayPoint()
        if not (0 < z < 1):
            return None
        return x, self.p.window_size[1] - y


# --------------------------------------------------------------------------------------------- cameras


def _course_point(build, hf, s, n=0.0):
    E, N = build.frame.to_map(s, n)
    return float(E), float(N), float(hf.height(E, N))


def camera(mode: str, t: float, lead: dict, scene: Scene, t_end: float, total_T: float):
    """(position, focal point, up) in scene coordinates for race time t."""
    b, hf, f = scene.build, scene.hf, scene.build.frame
    s, n_lead = lead["s"], float(np.interp(lead["s"], f.s, f.s * 0))  # n of centreline is 0
    L = scene.local
    if t < 0:  # intro: drone flies down to the start
        a = (t + INTRO_S) / INTRO_S
        a = a * a * (3 - 2 * a)
        far = np.array(L(*_course_point(b, hf, 220.0))) + [0, 0, 260]
        near = np.array(L(*_course_point(b, hf, -30.0))) + [0, 0, 16]
        pos = (1 - a) * far + a * near
        foc = (1 - a) * np.array(L(*_course_point(b, hf, 500.0))) + a * np.array(L(*_course_point(b, hf, 15.0)))
        return pos, foc, (0, 0, 1)
    if mode == "overhead":
        p = np.array(L(lead["E"], lead["N"], lead["Z"]))
        tx, ty = np.interp(s, f.s, f.tx), np.interp(s, f.s, f.ty)
        return p + [-tx * 10, -ty * 10, 95], p + [tx * 8, ty * 8, 0], (tx, ty, 0)
    if mode == "broadcast":
        if t > total_T - 4.5:
            mode = "finish"
        else:
            shots = ["chase", "chase", "side", "chase", "drone", "chase", "side"]
            mode = shots[int(t // 7.5) % len(shots)]
    skier = np.array(L(lead["E"], lead["N"], lead["Z"]))
    if mode == "finish":
        # on the open piste just past the finish line, off to one side, looking back up at the racers
        n_cam = 0.55 * float(np.interp(f.length, f.s, b.piste_hi))
        pos = np.array(L(*_course_point(b, hf, f.length + 6, n_cam))) + [0, 0, 5]
        return pos, skier + [0, 0, 1.0], (0, 0, 1)
    if mode == "side":
        # just inside the safety net, ahead of the racer, alternating sides between shots
        edge = b.piste_hi if (int(t // 7.5) % 2) else b.piste_lo
        n_edge = float(np.interp(s + 16, f.s, edge))
        cam = np.array(L(*_course_point(b, hf, s + 16, n_edge - np.sign(n_edge) * 3.0))) + [0, 0, 4.5]
        return cam, skier + [0, 0, 1.0], (0, 0, 1)
    if mode == "drone":
        cam = np.array(L(*_course_point(b, hf, s - 45, 25.0))) + [0, 0, 55]
        return cam, np.array(L(*_course_point(b, hf, s + 25))) + [0, 0, 0], (0, 0, 1)
    # chase: behind and above the leader, along the (smooth) course direction
    tx, ty = float(np.interp(s, f.s, f.tx)), float(np.interp(s, f.s, f.ty))
    ce, cn = lead["E"] - 15 * tx, lead["N"] - 15 * ty
    cam = np.array(L(ce, cn, hf.height(ce, cn) + 6.5))
    fe, fn = lead["E"] + 16 * tx, lead["N"] + 16 * ty
    foc = 0.65 * skier + 0.35 * np.array(L(fe, fn, hf.height(fe, fn)))
    return cam, foc + [0, 0, 1.0], (0, 0, 1)


class SmoothCam:
    """Critically damped camera so cuts are cuts but tracking is smooth."""

    def __init__(self):
        self.pos = self.foc = None
        self.mode = None

    def __call__(self, pos, foc, mode_key, alpha=0.25):
        pos, foc = np.asarray(pos, float), np.asarray(foc, float)
        if self.pos is None or mode_key != self.mode:
            self.pos, self.foc, self.mode = pos, foc, mode_key
        else:
            self.pos = self.pos + alpha * (pos - self.pos)
            self.foc = self.foc + alpha * (foc - self.foc)
        return self.pos, self.foc


# -------------------------------------------------------------------------------------------- graphics


def _font(size: int, bold: bool = True):
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def _surname(fis_name: str) -> str:
    """FIS lists surnames in capitals first: 'MCGRATH Atle Lie' -> 'McGrath'."""
    words = [w for w in fis_name.split() if w.isupper()]
    out = " ".join(w.title() for w in words)
    return out.replace("Mcg", "McG")


def _clock(t: float) -> str:
    t = max(t, 0.0)
    return f"{int(t // 60)}:{t % 60:05.2f}" if t >= 60 else f"{t:.2f}"


class Graphics:
    def __init__(self, build, hf, racers, size):
        self.W, self.H = size
        k = self.H / 1080
        self.k = k
        self.f_big, self.f_mid, self.f_small, self.f_tiny = (_font(int(64 * k)), _font(int(30 * k)), _font(int(24 * k)),
                                                             _font(int(18 * k), bold=False))
        self.racers = racers
        self.build = build
        self._map(build, hf)

    def _map(self, build, hf):
        f = build.frame
        k = self.k
        w, h = int(250 * k), int(330 * k)
        pad = 25
        self.map_box = (self.W - w - int(30 * k), self.H - h - int(30 * k), w, h)
        e0, e1, n0, n1 = f.E.min() - pad, f.E.max() + pad, f.N.min() - pad, f.N.max() + pad
        scale = min(w / (e1 - e0), h / (n1 - n0))
        ox = (w - (e1 - e0) * scale) / 2
        oy = (h - (n1 - n0) * scale) / 2
        self.map_xy = lambda E, N: (self.map_box[0] + ox + (np.asarray(E) - e0) * scale,
                                    self.map_box[1] + h - oy - (np.asarray(N) - n0) * scale)
        img = Image.new("RGBA", (w, h), (10, 18, 32, 190))
        d = ImageDraw.Draw(img)
        Eo, No = T.osm_outline(T.DATA / "gran_risa" / "osm_pistes.json")
        x, y = self.map_xy(Eo, No)
        d.polygon([(a - self.map_box[0], b - self.map_box[1]) for a, b in zip(x, y)], fill=(235, 240, 248, 235))
        self.map_img = img

    def draw(self, frame: np.ndarray, t: float, states: dict, scene: Scene, title_alpha: float = 0.0,
             results: float = 0.0) -> np.ndarray:
        img = Image.fromarray(frame).convert("RGBA")
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        k = self.k
        racers = [r for r in self.racers]
        leader = min(racers, key=lambda r: r.T)

        # name tags above each skier; tags that would overlap are stacked upwards, leader's lowest
        placed = []
        for r in racers:  # sorted fastest first
            st = states[r.key]
            p = scene.to_screen(scene.local(st["E"], st["N"], st["Z"] + 3.2))
            if p is None or not (0 < p[0] < self.W and 0 < p[1] < self.H):
                continue
            label = r.name
            tw = d.textlength(label, font=self.f_small)
            x, y = p[0] - tw / 2, p[1] - 30 * k
            for _ in range(len(placed)):
                hit = [b for b in placed if x < b[2] and x + tw + 16 * k > b[0] and y < b[3] and y + 34 * k > b[1]]
                if not hit:
                    break
                y = min(b[1] for b in hit) - 34 * k
            placed.append((x - 8 * k, y - 4 * k, x + tw + 8 * k, y + 28 * k))
            d.rounded_rectangle((x - 8 * k, y - 4 * k, x + tw + 8 * k, y + 28 * k), 6 * k, fill=(*_rgb(r.colour), 215))
            d.text((x, y), label, font=self.f_small, fill=(255, 255, 255, 255) if r.key != "schwarz_pace" else (20, 20, 20, 255))

        if title_alpha > 0:
            self._title(d, title_alpha)
            img.alpha_composite(over)
            return np.asarray(img.convert("RGB"))

        # top-left: event bug
        d.rectangle((30 * k, 30 * k, 520 * k, 112 * k), fill=(10, 18, 32, 220))
        d.rectangle((30 * k, 30 * k, 40 * k, 112 * k), fill=(214, 45, 32, 255))
        d.text((56 * k, 38 * k), "GRAN RISA  ·  ALTA BADIA", font=self.f_mid, fill="white")
        d.text((56 * k, 76 * k), "Men's Giant Slalom · run 1 · simulated lines", font=self.f_tiny, fill=(190, 200, 215))

        # top-right: clock
        race_t = max(t, 0.0)
        cw = 300 * k
        d.rectangle((self.W - cw - 30 * k, 30 * k, self.W - 30 * k, 128 * k), fill=(10, 18, 32, 220))
        d.text((self.W - cw - 10 * k, 36 * k), _clock(min(race_t, max(r.T for r in racers))), font=self.f_big, fill="white")
        # Schwarz split pop-up for 4 s after each split time
        for label, ts in SPLITS + [("FINISH", SCHWARZ_T)]:
            if ts <= race_t < ts + 4:
                d.rectangle((self.W - cw - 30 * k, 132 * k, self.W - 30 * k, 180 * k), fill=(214, 45, 32, 230))
                d.text((self.W - cw - 14 * k, 140 * k), f"SCHWARZ {label}  {ts:.2f}", font=self.f_small, fill="white")

        # left: standings (by progress), gap = time behind the leader at the same point
        rows = sorted(racers, key=lambda r: (-(states[r.key]["s"]), r.T) if race_t < r.T else (-1e9, r.T))
        rows = sorted(racers, key=lambda r: (0, r.T) if states[r.key]["finished"] else (1, -states[r.key]["s"]))
        front = rows[0]
        y = 160 * k
        d.rectangle((30 * k, y, 470 * k, y + 40 * k), fill=(214, 45, 32, 235))
        d.text((44 * k, y + 6 * k), "LIVE STANDINGS", font=self.f_small, fill="white")
        y += 44 * k
        for i, r in enumerate(rows):
            st = states[r.key]
            if st["finished"]:
                gap = f"{r.T:.2f}"
            elif r is front or i == 0:
                gap = "--"
            else:
                ref = front if not states[front.key]["finished"] else leader
                gap = f"+{max(race_t - ref.time_at_s(st['s']), 0):.2f}"
            d.rectangle((30 * k, y, 470 * k, y + 40 * k), fill=(10, 18, 32, 205))
            d.rectangle((30 * k, y, 40 * k, y + 40 * k), fill=(*_rgb(r.colour), 255))
            d.text((52 * k, y + 7 * k), f"{i + 1}", font=self.f_small, fill=(190, 200, 215))
            d.text((86 * k, y + 7 * k), r.name, font=self.f_small, fill="white")
            gw = d.textlength(gap, font=self.f_small)
            d.text((458 * k - gw, y + 7 * k), gap, font=self.f_small, fill=(255, 214, 10) if st["finished"] else "white")
            y += 42 * k

        # bottom-left: leader telemetry
        st = states[front.key] if not states[front.key]["finished"] else states[leader.key]
        who = front if not states[front.key]["finished"] else leader
        bx, by = 30 * k, self.H - 170 * k
        d.rectangle((bx, by, bx + 640 * k, by + 140 * k), fill=(10, 18, 32, 220))
        d.rectangle((bx, by, bx + 640 * k, by + 8 * k), fill=(*_rgb(who.colour), 255))
        d.text((bx + 16 * k, by + 16 * k), who.name.upper(), font=self.f_small, fill=(190, 200, 215))
        for j, (lab, val) in enumerate((("km/h", f"{st['v'] * 3.6:.0f}"), ("G", f"{st['G']:.1f}"),
                                        ("edge °", f"{np.degrees(abs(st['edge'])):.0f}"))):
            x = bx + 16 * k + j * 205 * k
            d.text((x, by + 48 * k), val, font=self.f_big, fill="white")
            d.text((x + d.textlength(val, font=self.f_big) + 8 * k, by + 82 * k), lab, font=self.f_small, fill=(190, 200, 215))
        # edge bar with boot-out mark
        frac = min(np.degrees(abs(st["edge"])) / 85.0, 1.0)
        d.rectangle((bx + 16 * k, by + 124 * k, bx + 624 * k, by + 132 * k), fill=(60, 70, 90, 255))
        d.rectangle((bx + 16 * k, by + 124 * k, bx + 16 * k + frac * 608 * k, by + 132 * k), fill=(255, 214, 10, 255))
        xb = bx + 16 * k + BOOT_OUT_DEG / 85 * 608 * k
        d.line((xb, by + 118 * k, xb, by + 138 * k), fill=(214, 45, 32, 255), width=max(int(3 * k), 1))

        # bottom-right: course map
        mx, my, mw, mh = self.map_box
        over.alpha_composite(self.map_img, (int(mx), int(my)))
        for r in sorted(racers, key=lambda r: -r.T):
            s2 = states[r.key]
            x, y2 = self.map_xy(s2["E"], s2["N"])
            rr = 6 * k
            d.ellipse((x - rr, y2 - rr, x + rr, y2 + rr), fill=(*_rgb(r.colour), 255), outline=(10, 18, 32, 255))

        d.rectangle((0, self.H - 26 * k, self.W, self.H), fill=(10, 18, 32, 170))
        d.text((30 * k, self.H - 23 * k), "Schwarz pace = the fastest simulated line at Marco Schwarz's average pace "
               "(76.00 s, real fastest run 1, 21 Dec 2025); his real line is not public.  Terrain: South Tyrol DTM 2.5 m.",
               font=self.f_tiny, fill=(200, 210, 225, 255))

        if results > 0:
            self._results(d, results)
        img.alpha_composite(over)
        return np.asarray(img.convert("RGB"))

    def _title(self, d, title_alpha):
        if True:
            a = int(230 * title_alpha)
            d.rectangle((0, self.H * 0.36, self.W, self.H * 0.62), fill=(10, 18, 32, a))
            for txt, font, yy in (("RACING-LINE OPTIMIZER", self.f_big, 0.40),
                                  ("Evolution vs. gradient descent on the real Gran Risa, Alta Badia", self.f_mid, 0.50),
                                  ("Every racer is one optimizer's best line · DCS 340 · Kyle Robin & Liam Baron", self.f_small, 0.56)):
                tw = d.textlength(txt, font=font)
                d.text(((self.W - tw) / 2, self.H * yy), txt, font=font, fill=(255, 255, 255, int(255 * title_alpha)))

    def _results(self, d, alpha):
        k = self.k
        sim = [(r.name, r.T, r.colour, True) for r in self.racers if r.key != "schwarz_pace"]
        real = [(f"{_surname(name)} ({nat})", t, "#ffffff", False) for name, nat, t in RACE["run1_top5"]]
        rows = sorted(sim + real, key=lambda x: x[1])
        w, rh = 760 * k, 44 * k
        x0 = (self.W - w) / 2
        y0 = (self.H - (len(rows) + 1) * rh) / 2
        a = int(235 * alpha)
        d.rectangle((x0, y0, x0 + w, y0 + rh), fill=(214, 45, 32, a))
        d.text((x0 + 16 * k, y0 + 8 * k), "RESULTS · simulated lines vs. the real run-1 top 5", font=self.f_small,
               fill=(255, 255, 255, int(255 * alpha)))
        best = rows[0][1]
        for i, (name, t, colour, is_sim) in enumerate(rows):
            y = y0 + (i + 1) * rh
            d.rectangle((x0, y, x0 + w, y + rh - 2 * k), fill=(10, 18, 32, a))
            d.rectangle((x0, y, x0 + 10 * k, y + rh - 2 * k), fill=(*_rgb(colour), a))
            fill = (255, 255, 255, int(255 * alpha))
            d.text((x0 + 24 * k, y + 8 * k), f"{i + 1}", font=self.f_small, fill=fill)
            d.text((x0 + 70 * k, y + 8 * k), name, font=self.f_small, fill=fill)
            tag = "simulated" if is_sim else "real race"
            d.text((x0 + 400 * k, y + 11 * k), tag, font=self.f_tiny, fill=(190, 200, 215, int(255 * alpha)))
            txt = f"{t:.2f}" + ("" if i == 0 else f"   +{t - best:.2f}")
            tw = d.textlength(txt, font=self.f_small)
            d.text((x0 + w - 16 * k - tw, y + 8 * k), txt, font=self.f_small, fill=fill)


# ------------------------------------------------------------------------------------------------ main


def render(results: Path, out: Path, methods=None, camera_mode="broadcast", fps=30, size=(1920, 1080),
           still: float | None = None, pace=True, max_race_s: float | None = None) -> Path:
    hf = T.HeightField.load(T.DATA / "gran_risa" / "terrain.npz")
    cfg_seed = 0
    build = build_gran_risa(seed=cfg_seed)
    racers = load_racers(results, methods, build, hf)
    if not racers:
        raise SystemExit(f"no runs found in {results}")
    if pace:
        racers.append(pace_ghost(racers[0]))
    scene = Scene(build, hf, racers, size)
    gfx = Graphics(build, hf, racers, size)
    smooth = SmoothCam()
    # The race ends a little after the last racer within 10 s of the winner (the coach line can be ~20 s back).
    contenders = [r.T for r in racers if r.T < racers[0].T + 10]
    race_end = max(contenders) + 2.0
    if max_race_s is not None:
        race_end = min(race_end, max_race_s)
    times = [still] if still is not None else list(np.arange(-INTRO_S, race_end + OUTRO_S, 1.0 / fps))
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if still is None:
        import imageio_ffmpeg

        writer = imageio_ffmpeg.write_frames(str(out), size, fps=fps, codec="libx264", quality=None,
                                             output_params=["-crf", "28", "-pix_fmt", "yuv420p", "-preset", "medium"],
                                             macro_block_size=8)
        writer.send(None)
    for i, t in enumerate(times):
        race_t = min(max(t, 0.0), race_end)
        states = {r.key: r.at(race_t) for r in racers}
        lead_r = next((r for r in racers if not states[r.key]["finished"] and r.key != "schwarz_pace"), racers[0])
        lead = states[lead_r.key]
        mode_key = "intro" if t < 0 else (camera_mode if camera_mode != "broadcast" else
                                           ("finish" if t > racers[0].T - 4.5 else int(t // 7.5)))
        pos, foc, up = camera(camera_mode, t if t < 0 else min(t, racers[0].T), lead, scene, race_end, racers[0].T)
        pos, foc = smooth(pos, foc, mode_key, alpha=0.18 if t >= 0 else 1.0)
        scene.p.camera.position, scene.p.camera.focal_point, scene.p.camera.up = tuple(pos), tuple(foc), tuple(up)
        scene.p.camera.clipping_range = (0.5, 6000)
        for r in racers:
            scene.place(r, states[r.key])
            scene.trail(r, race_t)
        scene.p.render()
        frame = scene.p.screenshot(return_img=True)
        title = 0.0 if t >= 0 else float(np.clip(min(t + INTRO_S, -t) / 0.6, 0, 1))
        res = float(np.clip((t - race_end) / 0.8, 0, 1))
        frame = gfx.draw(frame, race_t if t >= 0 else 0.0, states, scene, title_alpha=title, results=res)
        if still is not None:
            Image.fromarray(frame).save(out)
        else:
            writer.send(np.ascontiguousarray(frame))
        if i % (fps * 10) == 0:
            print(f"  t = {t:6.1f} s  ({i}/{len(times)} frames)", flush=True)
    if writer is not None:
        writer.close()
    print(f"wrote {out}")
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="experiment results directory, e.g. results/gran_risa")
    parser.add_argument("--out", type=Path, required=True, help=".mp4 for a video, .png with --still")
    parser.add_argument("--methods", nargs="*", help="which methods race (default: every one found)")
    parser.add_argument("--camera", choices=["broadcast", "chase", "overhead"], default="broadcast")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--size", type=int, nargs=2, default=(1920, 1080), metavar=("W", "H"))
    parser.add_argument("--still", type=float, help="render one frame at this race time (s) instead of a video")
    parser.add_argument("--no-pace", action="store_true", help="leave out the Schwarz pace ghost")
    parser.add_argument("--max-race-s", type=float, help="cut the race at this time (for quick previews)")
    a = parser.parse_args(argv)
    render(a.results, a.out, a.methods, a.camera, a.fps, tuple(a.size), a.still, not a.no_pace, a.max_race_s)


if __name__ == "__main__":
    main()
