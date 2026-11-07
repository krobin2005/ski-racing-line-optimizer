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
FINISH_BOARDS_FROM = 45.0   # m before the line: the red nets give way to the padded boards of the finish area
RUNOUT = 75.0               # m of finish area past the line
DYE_BLUE, DYE_RED = "#1f5fd0", "#d9251d"
START_HALF_W, START_DEPTH = 2.6, 6.0   # start house (m)
START_SHOT_S = 1.6                     # seconds of the start camera before the broadcast cuts begin
START_CAM = (11.0, 6.0, 3.5)           # start camera in start coordinates (downhill, left, up)


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
        tc = min(max(t, 0.0), self.t[-1])  # the samples run past T: the racer coasts into the finish area
        g = lambda a: float(np.interp(tc, self.t, a))  # noqa: E731
        # heading from a short look-ahead along the line
        dt = 0.15
        end = self.t[-1]
        E1, N1 = float(np.interp(min(tc + dt, end), self.t, self.E)), float(np.interp(min(tc + dt, end), self.t, self.N))
        E0, N0 = float(np.interp(max(tc - dt, 0), self.t, self.E)), float(np.interp(max(tc - dt, 0), self.t, self.N))
        Z1, Z0 = float(np.interp(min(tc + dt, end), self.t, self.Z)), float(np.interp(max(tc - dt, 0), self.t, self.Z))
        h = np.array([E1 - E0, N1 - N0, Z1 - Z0])
        return {"E": g(self.E), "N": g(self.N), "Z": g(self.Z), "s": g(self.s), "v": g(self.v), "G": g(self.G),
                "edge": g(self.edge), "kappa": g(self.kappa), "heading": h / (np.linalg.norm(h) + 1e-9),
                "finished": t >= self.T}

    def time_at_s(self, s: float) -> float:
        return float(np.interp(s, self.s, self.t))


def map_ext(frame, s, n):
    """CourseFrame.to_map, continued straight past the finish line along the last tangent (the finish area)."""
    s, n = np.asarray(s, float), np.asarray(n, float)
    E, N = frame.to_map(np.clip(s, 0.0, frame.length), n)
    over = np.maximum(s - frame.length, 0.0)
    tx, ty = float(np.interp(frame.length, frame.s, frame.tx)), float(np.interp(frame.length, frame.s, frame.ty))
    under = np.minimum(s, 0.0)  # and straight back uphill before the start
    return E + over * tx + under * float(frame.tx[0]), N + over * ty + under * float(frame.ty[0])


RUNOUT_DECEL = 4.0   # m/s², a hard skid to a stop after the finish line


def coast(t, s, n, v, T, frame, hf):
    """Samples after the finish: the racer carries on straight, braking at RUNOUT_DECEL until stopped."""
    v0 = float(v[-1])
    stop = v0 / RUNOUT_DECEL
    tau = np.linspace(0, stop, 40)[1:]
    d = v0 * tau - 0.5 * RUNOUT_DECEL * tau ** 2
    s2, n2 = s[-1] + d, np.full_like(d, n[-1])
    E, N = map_ext(frame, s2, n2)
    return T + tau, s2, E, N, hf.height(E, N), v0 - RUNOUT_DECEL * tau


def flag_spans(inside: float, outside: float, width: float = 0.75):
    """(pole, far pole) of the two flags of a GS gate, as offsets n. The racer passes between the turning
    pole and the outside pole, so each flag extends away from that gap: the inside flag on the far side of
    the turning pole, the outside flag beyond the outside pole."""
    d = np.sign(outside - inside) or 1.0
    return [(inside, inside - d * width), (outside, outside + d * width)]


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
        T = float(champ["T"])
        s, n, v = pad(tel["y"]), pad(tel["x"]), pad(tel["v"])
        ct, cs, cE, cN, cZ, cv = coast(t, s, n, v, t[-1], build.frame, hf)
        zeros = np.zeros_like(ct)
        racers.append(Racer(key, name, colour, np.concatenate([t, ct]), np.concatenate([s, cs]),
                            np.concatenate([pad(E), cE]), np.concatenate([pad(N), cN]),
                            np.concatenate([hf.height(pad(E), pad(N)), cZ]), np.concatenate([v, cv]),
                            np.concatenate([pad(tel["G"]), zeros + 1.0]), np.concatenate([pad(tel["edge"]), zeros]),
                            np.concatenate([pad(tel["kappa"]), zeros]), T))
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
        self._dye()
        self._finish_area()
        self._finish_arch()
        self._start_house()
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
        Z = hf.z[i0:i1, j0:j1].copy()
        # the start platform is groomed flat: level the snow under and around the start house
        self.start_axes = self._start_frame()
        o, a, b, _ = self.start_axes
        z0 = o[2] + self.origin[2]
        dx, dy = X - (o[0] + self.origin[0]), Y - (o[1] + self.origin[1])
        pa, pb = dx * a[0] + dy * a[1], dx * b[0] + dy * b[1]
        out = np.maximum(np.maximum(-START_DEPTH - 3 - pa, 0), np.maximum(np.abs(pb) - START_HALF_W - 3, 0))
        # cut only behind the gate: downhill of it the racers ski on the real slope
        w = np.clip(1 - out / 5.0, 0, 1) * np.clip(1.0 - pa / 1.5, 0, 1)
        Z = w * z0 + (1 - w) * Z
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
        fe, fn = self.build.frame.to_map(self.build.spec.length, 0.0)
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
        f, b_len = self.build.frame, float(self.build.spec.length)
        for side_lo in (True, False):
            edge = self.build.piste_lo if side_lo else self.build.piste_hi
            s = f.s[(f.s >= -5) & (f.s <= b_len - FINISH_BOARDS_FROM)]
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
            for n_pole, n_far in flag_spans(inside, outside):
                for n_p in (n_pole, n_far):
                    E, N = f.to_map(s, n_p)
                    base = self.local(E, N, self.hf.height(E, N))
                    meshes[colour].append(self.pv.Cylinder(center=base + [0, 0, 0.95], direction=(0, 0, 1),
                                                           radius=0.035, height=1.9, resolution=8))
                Ea, Na = f.to_map(s, n_pole)
                Eb, Nb = f.to_map(s, n_far)
                a = self.local(Ea, Na, self.hf.height(Ea, Na) + 1.2)
                b = self.local(Eb, Nb, self.hf.height(Eb, Nb) + 1.2)
                panel = self.pv.Plane(center=(a + b) / 2 + [0, 0, 0.3], direction=np.cross(b - a, [0, 0, 1]),
                                      i_size=0.6, j_size=np.linalg.norm(b - a), i_resolution=1, j_resolution=1)
                meshes[colour].append(panel)
        for colour, ms in meshes.items():
            if ms:
                hexc = {"red": "#e0201b", "blue": "#1a5fd6"}[colour]
                self.p.add_mesh(self.pv.merge(ms), color=hexc, ambient=0.45, diffuse=0.7)

    # ------------------------------------------------------------------ dye, finish area

    def _ribbon(self, E, N, width, lift=0.08):
        """A flat strip of the given width draped on the snow along the map polyline (E, N)."""
        E, N = np.asarray(E, float), np.asarray(N, float)
        dE, dN = np.gradient(E), np.gradient(N)
        norm = np.hypot(dE, dN) + 1e-9
        pe, pn = -dN / norm * width / 2, dE / norm * width / 2
        Ea, Na, Eb, Nb = E + pe, N + pn, E - pe, N - pn
        a = self.local(Ea, Na, self.hf.height(Ea, Na) + lift)
        b = self.local(Eb, Nb, self.hf.height(Eb, Nb) + lift)
        k = len(E)
        faces = np.hstack([[4, i, i + 1, k + i + 1, k + i] for i in range(k - 1)])
        return self.pv.PolyData(np.vstack([a, b]), faces=faces)

    def _course_ribbon(self, s, n, width, lift=0.08):
        E, N = map_ext(self.build.frame, s, n)
        return self._ribbon(E, N, width, lift)

    def _paint(self, ribbons, colour, opacity=0.9):
        if ribbons:
            self.p.add_mesh(self.pv.merge(ribbons), color=colour, opacity=opacity, ambient=0.75, diffuse=0.35,
                            specular=0.0)

    def finish_width(self, s):
        """(n_lo, n_hi) of the course at s: the piste, opening out into the finish area after the line."""
        f, b = self.build.frame, self.build
        s = np.asarray(s, float)
        lo = np.interp(np.minimum(s, self.build.spec.length), f.s, b.piste_lo)
        hi = np.interp(np.minimum(s, self.build.spec.length), f.s, b.piste_hi)
        a = np.clip((s - self.build.spec.length) / 18.0, 0, 1)
        a = a * a * (3 - 2 * a)
        return (1 - a) * lo + a * -26.0, (1 - a) * hi + a * 26.0

    def _dye(self):
        """Blue dye on the race piste, as sprayed at the Gran Risa so racers can read the snow in flat light:
        a stroke from every turning pole toward the next gate, and lines across the slope every few gates."""
        spec, f = self.build.spec, self.build.frame
        blue = []
        for i in range(spec.n_gates - 1):
            s0, n0 = float(spec.gate_y[i]), float(spec.gate_x[i])
            s1, n1 = float(spec.gate_y[i + 1]), float(spec.gate_x[i + 1])
            u = np.linspace(0.06, 0.06 + 10.0 / max(np.hypot(s1 - s0, n1 - n0), 1.0), 16)
            blue.append(self._course_ribbon(s0 + u * (s1 - s0), n0 + u * (n1 - n0), 0.45))
            if i % 3 == 1:  # a line across the slope between gates, slightly diagonal
                sm = 0.5 * (s0 + s1)
                lo, hi = self.finish_width(sm)
                n = np.linspace(float(lo) + 1.0, float(hi) - 1.0, 30)
                blue.append(self._course_ribbon(sm + 0.12 * (n - n.mean()), n, 0.4))
        self._paint(blue, DYE_BLUE, 0.85)

    def _finish_area(self):
        """The finish at La Villa: padded boards, the dyed finish line, blue stripes in the run-out,
        floodlights and a grandstand. Generic (no sponsor logos), laid out after photos of the race."""
        pv, f, hf = self.pv, self.build.frame, self.hf
        L = float(self.build.spec.length)  # the finish line (the map frame runs on 40 m past it)
        s = np.linspace(L - FINISH_BOARDS_FROM, L + RUNOUT, 120)
        lo, hi = self.finish_width(s)
        # padded boards, 1.2 m tall, alternating blue and white panels
        for edge, out in ((lo, -1.0), (hi, 1.0)):
            E, N = map_ext(f, s, edge + out * 1.0)
            Z = hf.height(E, N)
            bottom, top = self.local(E, N, Z), self.local(E, N, Z + 1.2)
            k = len(s)
            faces = np.hstack([[4, i, i + 1, k + i + 1, k + i] for i in range(k - 1)])
            mesh = pv.PolyData(np.vstack([bottom, top]), faces=faces)
            cols = np.array([[22, 70, 170] if (i // 3) % 2 == 0 else [240, 240, 240] for i in range(k - 1)], np.uint8)
            mesh.cell_data["rgb"] = cols
            self.p.add_mesh(mesh, scalars="rgb", rgb=True, ambient=0.55, diffuse=0.6)
        # the finish line: a broad red dye band right under the arch
        lo0, hi0 = self.finish_width(L)
        n = np.linspace(float(lo0), float(hi0), 30)
        self._paint([self._course_ribbon(np.full_like(n, L), n, 1.0, lift=0.15)], DYE_RED, 0.95)
        # diagonal blue stripes and a red border band in the run-out
        stripes = []
        for k0 in np.arange(L + 10, L + RUNOUT - 8, 6.5):
            n = np.linspace(-22, 22, 90)
            ss = k0 + 0.55 * n
            ok = (ss > L + 4) & (ss < L + RUNOUT - 4)
            if ok.sum() > 3:
                stripes.append(self._course_ribbon(ss[ok], n[ok], 1.4, lift=0.18))
        self._paint(stripes, DYE_BLUE, 0.8)
        band = [self._course_ribbon(np.full(60, L + d), np.linspace(-24, 24, 60), 0.5, lift=0.18) for d in (4.0, RUNOUT - 4)]
        self._paint(band, DYE_RED, 0.9)
        # floodlight towers behind the boards
        lights = []
        for d, side in ((-40, -1), (-25, 1), (25, -1), (60, -1)):
            lo_d, hi_d = self.finish_width(L + d)
            nn = (float(hi_d) + 6) if side > 0 else (float(lo_d) - 6)
            E, N = map_ext(f, L + d, nn)
            base = self.local(E, N, hf.height(E, N))
            lights.append(pv.Cylinder(center=base + [0, 0, 9], direction=(0, 0, 1), radius=0.25, height=18))
            lights.append(pv.Box(bounds=(base[0] - 1.4, base[0] + 1.4, base[1] - 0.4, base[1] + 0.4,
                                         base[2] + 17.5, base[2] + 19.3)))
        self.p.add_mesh(pv.merge(lights), color="#9a9fa6", ambient=0.4)
        # a grandstand on the outside of the run-out, filled with a speckled crowd
        rng = np.random.default_rng(3)
        rows = []
        cols = []
        for r in range(6):
            for d in np.linspace(12, RUNOUT - 8, 26):
                n0 = 29.0 + r * 1.6
                E, N = map_ext(f, L + d, n0)
                z = hf.height(*map_ext(f, L + d, 27.0)) + 0.6 + r * 0.9
                c = self.local(E, N, z)
                rows.append(pv.Box(bounds=(c[0] - 1.3, c[0] + 1.3, c[1] - 1.3, c[1] + 1.3, c[2] - 0.6 - r * 0.9, c[2])))
                cols.append(rng.choice([[200, 40, 40], [30, 30, 40], [230, 230, 235], [40, 80, 170], [220, 170, 40]]))
        stand = pv.merge(rows)
        per = stand.n_cells // len(rows)
        stand.cell_data["rgb"] = np.repeat(np.array(cols, np.uint8), per, axis=0)
        self.p.add_mesh(stand, scalars="rgb", rgb=True, ambient=0.5, diffuse=0.6)

    # ------------------------------------------------------------------ start

    def _start_frame(self):
        """Origin and axes at the start: a = downhill along the course, b = left, c = up. The origin is
        where the racers stand (s = 0) at the height of the start platform."""
        f = self.build.frame
        n0 = float(self.build.spec.x_start)
        E, N = map_ext(f, 0.0, n0)
        tx, ty = float(f.tx[0]), float(f.ty[0])
        a = np.array([tx, ty, 0.0])
        b = np.array([-ty, tx, 0.0])
        o = self.local(E, N, self.hf.height(E, N))
        return o, a, b, np.array([0, 0, 1.0])

    def _box(self, lo, hi):
        """An axis-aligned box in start coordinates (a, b, c), as a mesh in the scene."""
        o, a, b, c = self.start_axes
        m = self.pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        p = m.points.copy()
        m.points = o + p[:, :1] * a + p[:, 1:2] * b + p[:, 2:3] * c
        return m

    def _text(self, txt, centre, height, facing=+1, depth=0.06):
        """3D lettering on a wall facing downhill (facing=+1) or sideways to the left/right (facing=+-2)."""
        o, a, b, c = self.start_axes
        m = self.pv.Text3D(txt, depth=depth)
        m.points *= height / 1.14  # Text3D capitals are ~1.14 units tall
        x, y, z = m.points[:, 0], m.points[:, 1], m.points[:, 2]
        if abs(facing) == 1:   # facing downhill (+1, read from below) or uphill (-1, read by the racer)
            right, out = b * facing, a * facing
        else:                   # on a side wall facing +b (left) or -b (right)
            s = np.sign(facing)
            right, out = -a * s, b * s
        ca, cb, cc = centre
        m.points = o + ca * a + cb * b + cc * c + x[:, None] * right + y[:, None] * c + z[:, None] * out
        return m

    def _start_house(self):
        """The start, after photos of World Cup starts: a roofed start house on a platform, open downhill,
        with padded side panels, the timing wand, banner boards on every wall, a start clock and coaches.
        Sponsor boards are blank colour blocks: no real logos."""
        self.start_axes = self._start_frame()
        box = self._box
        red, white, dark, wood = [], [], [], []
        W, D, H = START_HALF_W, START_DEPTH, 3.4   # half-width, depth and wall height of the house (m)
        floor = 0.0
        wand_a = 1.0                      # the timing wand, just in front of the racers' boots
        # platform under the house and a short groomed ramp in front of it (the slope falls away fast)
        wood.append(box((-D - 1.0, -W - 1.5, -3.0), (wand_a + 0.4, W + 1.5, floor)))
        # walls: back and two sides, red with a white banner band; roof slab with a slight overhang
        red.append(box((-D, -W, floor), (-D + 0.25, W, H)))
        for side in (-1, 1):
            red.append(box((-D, side * W - 0.125, floor), (wand_a - 0.3, side * W + 0.125, H)))
            white.append(box((-D + 0.6, side * (W + 0.13) - 0.02, 1.5), (wand_a - 0.8, side * (W + 0.13) + 0.02, 2.4)))
        dark.append(box((-D - 0.4, -W - 0.5, H), (wand_a + 0.2, W + 0.5, H + 0.35)))
        # fascia over the opening, white, with START on it
        white.append(box((wand_a - 0.45, -W - 0.5, H - 0.9), (wand_a - 0.3, W + 0.5, H)))
        # the start gate: two padded panels either side of the racer, the wand between them
        for side in (-1, 1):
            red.append(box((wand_a - 0.35, side * 0.75 - 0.12, floor), (wand_a + 0.05, side * 0.75 + 0.12, 1.25)))
            dark.append(box((wand_a - 0.1, side * 0.62 - 0.07, 0.35), (wand_a + 0.05, side * 0.62 + 0.07, 0.55)))
        wand = self.pv.Cylinder(radius=0.02, height=1.2, direction=(0, 1, 0))
        o, a, b, c = self.start_axes
        wand.points = o + wand.points[:, :1] * a + wand.points[:, 1:2] * b + (wand.points[:, 2:3] + 0.45) * c \
            + wand_a * a
        # sponsor-style banner boards along the platform edge and beside the gate (blank colour blocks)
        boards = []
        cols = ["#1b3f8f", "#f2f2f2", "#c8102e", "#111111", "#f2c200", "#1b3f8f"]
        for i, col in enumerate(cols):
            a0 = -D + i * (D + wand_a) / len(cols)
            for side in (-1, 1):
                boards.append((box((a0 + 0.05, side * (W + 1.4) - 0.04, floor),
                                   (a0 + (D + wand_a) / len(cols) - 0.05, side * (W + 1.4) + 0.04, 0.9)), col))
        # start clock on a post to the racers' left, facing the gate
        dark.append(box((wand_a + 0.6, W + 0.6, floor), (wand_a + 0.75, W + 0.75, 2.2)))
        dark.append(box((wand_a + 0.2, W + 0.2, 2.2), (wand_a + 1.2, W + 1.4, 2.9)))
        clock = self._text("0:10", (wand_a + 0.19, W + 0.8, 2.55), 0.45, facing=-1, depth=0.03)
        # coaches beside the gate
        people = []
        rng = np.random.default_rng(5)
        for (pa, pb) in ((wand_a + 2.0, W + 1.0), (wand_a + 2.6, -W - 0.8), (wand_a + 1.4, -W - 1.6)):
            z0 = float(self.hf.height(*[q for q in self._start_map(pa, pb)])) - float(o[2] + self.origin[2])
            people.append(box((pa - 0.2, pb - 0.25, z0), (pa + 0.2, pb + 0.25, z0 + 1.45)))
            people.append(box((pa - 0.13, pb - 0.13, z0 + 1.45), (pa + 0.13, pb + 0.13, z0 + 1.75)))
        txt = [self._text("START", (wand_a - 0.29, 0.0, H - 0.65), 0.55),
               self._text("GRAN RISA", (-D / 2 + 0.3, W + 0.16, 1.95), 0.5, facing=2),
               self._text("GRAN RISA", (-D / 2 + 0.3, -W - 0.16, 1.95), 0.5, facing=-2)]
        add = self.p.add_mesh
        add(self.pv.merge(red), color="#c8102e", ambient=0.45, diffuse=0.7)
        add(self.pv.merge(white), color="#f4f4f4", ambient=0.5, diffuse=0.6)
        add(self.pv.merge(dark), color="#22252b", ambient=0.4, diffuse=0.6)
        add(self.pv.merge(wood), color="#e9edf2", ambient=0.5, diffuse=0.6)   # snow-covered platform
        add(wand, color="#d9dde3", ambient=0.5, specular=0.6)
        for m, col in boards:
            add(m, color=col, ambient=0.5, diffuse=0.6)
        add(clock, color="#f2c200", ambient=0.9)
        add(self.pv.merge(txt[:1]), color="#c8102e", ambient=0.8)
        add(self.pv.merge(txt[1:]), color="white", ambient=0.8)
        add(self.pv.merge(people), color="#1d2c4a", ambient=0.4)

    def start_point(self, abc):
        o, a, b, c = self.start_axes
        return o + abc[0] * a + abc[1] * b + abc[2] * c

    def _start_map(self, pa, pb):
        o, a, b, c = self.start_axes
        p = o + pa * a + pb * b
        return p[0] + self.origin[0], p[1] + self.origin[1]

    def _finish_arch(self):
        f, b = self.build.frame, self.build
        for s, colour in ((float(b.spec.length), "#d62d20"),):
            lo, hi = float(np.interp(s, f.s, b.piste_lo)), float(np.interp(s, f.s, b.piste_hi))
            if s > 0:  # the finish arch stands a little outside the dyed line, over the whole width
                lo, hi = lo - 1.5, hi + 1.5
            posts = []
            for n in (lo, hi):
                E, N = f.to_map(s, n)
                base = self.local(E, N, self.hf.height(E, N))
                posts.append(base)
                self.p.add_mesh(self.pv.Cube(center=base + [0, 0, 3.6], x_length=1.0, y_length=1.0, z_length=7.2),
                                color=colour, ambient=0.45)
            a, c = posts
            top = max(a[2], c[2]) + 6.4
            span = np.linalg.norm(c - a)
            beam = self.pv.Cube(center=[(a[0] + c[0]) / 2, (a[1] + c[1]) / 2, top], x_length=span + 1.0,
                                y_length=0.8, z_length=1.6)
            ang = np.degrees(np.arctan2(c[1] - a[1], c[0] - a[0]))
            beam.rotate_z(ang, point=beam.center, inplace=True)
            self.p.add_mesh(beam, color=colour, ambient=0.55)
            if s > 0:  # photocell timing posts at each end of the line
                for n in (lo + 1.0, hi - 1.0):
                    E, N = f.to_map(s, n)
                    base = self.local(E, N, self.hf.height(E, N))
                    self.p.add_mesh(self.pv.Cylinder(center=base + [0, 0, 0.6], direction=(0, 0, 1), radius=0.07,
                                                     height=1.2), color="#f2c200")

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
        near = scene.start_point(START_CAM)
        pos = (1 - a) * far + a * near
        af = min(1.0, (t + INTRO_S) / (0.65 * INTRO_S))  # the look settles on the start house first
        af = af * af * (3 - 2 * af)
        foc = (1 - af) * np.array(L(*_course_point(b, hf, 500.0))) + af * scene.start_point((-1.0, 0.0, 1.5))
        return pos, foc, (0, 0, 1)
    if t < START_SHOT_S and mode == "broadcast":  # out of the gate, seen from beside the start ramp
        skier = np.array(L(lead["E"], lead["N"], lead["Z"]))
        return scene.start_point(START_CAM), skier + [0, 0, 1.0], (0, 0, 1)
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
        # in the finish area, off to the side of the run-out, looking back up through the arch
        E, N = map_ext(f, float(b.spec.length) + 58, 12.0)
        pos = np.array(L(E, N, hf.height(E, N) + 9.0))
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
        for r in (racers if title_alpha <= 0 else []):  # sorted fastest first; none under the title card
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
                txt = f"SCHWARZ {label}  {ts:.2f}"
                bw = max(cw, d.textlength(txt, font=self.f_small) + 32 * k)
                d.rectangle((self.W - bw - 30 * k, 132 * k, self.W - 30 * k, 180 * k), fill=(214, 45, 32, 230))
                d.text((self.W - bw - 14 * k, 140 * k), txt, font=self.f_small, fill="white")

        # left: standings (by progress), gap = time behind the leader at the same point
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
                                           ("finish" if t > racers[0].T - 4.5 else
                                            "start" if t < START_SHOT_S else int(t // 7.5)))
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
