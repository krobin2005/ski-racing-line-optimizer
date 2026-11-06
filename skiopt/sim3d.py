"""3D point mass on real terrain (docs/ROADMAP_3D.md §2).

Course coordinates: s along the smoothed centreline (horizontal metres, fixed step ds), n across it
(positive left), terrain height Z(s, n) on a course grid. A point maps to c(s) + n l(s); along s
the horizontal scale factor is h = 1 - n kappa_c(s).

State (n, psi, v): psi is the horizontal heading relative to the centreline tangent; the control is
the horizontal curvature kappa_h of the path (1/m), so the line representations of 2D carry over.
Per step:

    d_sigma = h ds / cos(psi)                       horizontal path length
    n'      = n + tan(psi) h ds
    psi'    = psi + kappa_h d_sigma - kappa_c ds     the frame itself turns by kappa_c ds

Physics, from the surface geometry along the path (r(sigma) = (x, z(x)) with horizontal unit
tangent hh, left normal m, slope gradient g_z and Hessian H of z):

    r'  = (hh, g_z.hh),   q = |r'|,   T = r'/q                 3D unit tangent, d_ell = q d_sigma
    r'' = (kappa_h m, kappa_h g_z.m + hh^T H hh)
    k   = (r'' - (r''.T) T) / q^2                               3D curvature vector
    u   = (-g_z, 1)/|.|  (upward surface normal),  b = u x T    (left, in the surface)
    kappa_n = k.u   (+ compression, - roll),   kappa_g = k.b    (the carved turn)
    N     = v^2 kappa_n + g u_z                                 normal load per unit mass
    F_lat = v^2 kappa_g + g b_z                                 lateral load the edge must hold
    v dv/d_ell = -g T_z - mu max(N, 0) - c_d v^2 - turn losses(kappa_g, F_lat)

On a plane z = -sin(theta) y this reduces to the 2D model: N = g cos(theta) and F_lat = a_net.
G = sqrt(N^2 + F_lat^2)/g and the edge angle is atan(|F_lat| / N).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from skiopt import hazard
from skiopt.dynamics import clip_kappa, limit_heading, softplus_b
from skiopt.types import Gates, SimParams, StepOutput, Summary, Telemetry


class Grid3D(NamedTuple):
    """Terrain on the course grid: Z[i, j] at (s_i = i ds, n_j = n0 + j dn), plus precomputed
    derivative fields F = (z, z_s, z_n, z_ss, z_sn, z_nn) that the simulator interpolates bilinearly
    (much cheaper than differentiating a bicubic patch twice at every step)."""

    Z: jnp.ndarray
    ds: jnp.ndarray
    n0: jnp.ndarray
    dn: jnp.ndarray
    kappa_c: jnp.ndarray   # centreline curvature per s row (1/m)
    dkappa_c: jnp.ndarray  # its derivative d kappa_c / ds
    F: jnp.ndarray = None  # (6, rows, cols)


def make_grid(Z, ds: float, n0: float, dn: float, kappa_c=None) -> Grid3D:
    """Grid3D with derivative fields from second-order finite differences (exact for quadratics)."""
    Z = np.asarray(Z, float)
    zs, zn = np.gradient(Z, ds, axis=0), np.gradient(Z, dn, axis=1)
    zss, zsn = np.gradient(zs, ds, axis=0), np.gradient(zs, dn, axis=1)
    znn = np.gradient(zn, dn, axis=1)
    kappa_c = np.zeros(Z.shape[0]) if kappa_c is None else np.asarray(kappa_c, float)
    return Grid3D(Z=jnp.asarray(Z), ds=jnp.asarray(ds), n0=jnp.asarray(n0), dn=jnp.asarray(dn),
                  kappa_c=jnp.asarray(kappa_c), dkappa_c=jnp.asarray(np.gradient(kappa_c, ds)),
                  F=jnp.asarray(np.stack([Z, zs, zn, zss, zsn, znn])))


class Course3D(NamedTuple):
    """Duck-types the 2D Course for objectives and hazards: y -> s, x -> n."""

    length: jnp.ndarray
    x_start: jnp.ndarray
    piste_half_width: jnp.ndarray
    terrain: Grid3D
    gates: Gates
    piste_lo: jnp.ndarray = None   # right piste edge n per s row (real piste outline); None: +-half width
    piste_hi: jnp.ndarray = None   # left piste edge


class Params3D(NamedTuple):
    """3D-only risk constants (configs/base.yaml, `sim3d.risk`); the shared ones stay in SimParams.

    Boot-out: the boot touches the snow at about atan(stack / (boot half-width - ski half-waist)):
    49 mm stack, ~100 mm boot sole (assumed) and a 65 mm GS waist give atan(49 / 17.5) = 70 deg.
    Racers reach the low 70s by angulating, so this is a steep hazard around boot-out, not an
    instant DNF; only `edge_max` (well beyond it) ends the run outright."""

    edge_safe: float = np.radians(66.0)  # edge-angle hazard starts here
    c_edge: float = 0.0                  # hazard per metre per rad of edge angle above edge_safe
    beta_edge: float = 20.0
    edge_max: float = np.radians(78.0)   # hard DNF: no ski holds beyond this
    c_air: float = 0.0                   # hazard per metre per g of unloading below n_min
    n_min: float = 0.2                   # normal load (g) below which the skis lose grip
    beta_air: float = 10.0
    airborne_dnf: float = 1.0            # 1.0: a negative normal load (airborne) is a hard DNF
    edge_dnf: float = 1.0                # 1.0: edge angle above edge_max is a hard DNF
    # Sharpness (1/rad) of the smooth-objective wall at edge_max; 0 = the shared risk.wall.beta. The
    # shared 20 /m-or-g is soft for an excess in radians: a line 11 deg below edge_max still pays
    # ~0.1 s per metre of high edge in the smooth loss (REPORT.md §6, ablation 3).
    wall_beta_edge: float = 0.0

    @classmethod
    def from_config(cls, cfg: dict, equipment: dict | None = None) -> "Params3D":
        r = cfg.get("sim3d", {}).get("risk")
        if not r:
            return cls()
        if "edge_safe_below_bootout_deg" in r:
            from skiopt.config import load_equipment

            boot = boot_out_angle(equipment or load_equipment())
            edge_safe = boot - np.radians(r["edge_safe_below_bootout_deg"])
            edge_max = boot + np.radians(r["edge_max_above_bootout_deg"])
        else:
            edge_safe, edge_max = np.radians(r["edge_safe_deg"]), np.radians(r["edge_max_deg"])
        return cls(edge_safe=float(edge_safe), c_edge=float(r["c_edge"]), beta_edge=float(r["beta_edge"]),
                   edge_max=float(edge_max), c_air=float(r["c_air"]), n_min=float(r["n_min_g"]),
                   beta_air=float(r["beta_air"]), airborne_dnf=float(r["airborne_dnf"]), edge_dnf=float(r["edge_dnf"]),
                   wall_beta_edge=float(r.get("wall_beta_edge", 0.0)))


def boot_out_angle(equipment: dict) -> float:
    """Edge angle (rad) at which the boot sole edge touches the snow: atan(stack / overhang), where
    the overhang is how far the boot sole sticks out past the ski edge, (sole width - waist) / 2."""
    stack = float(equipment["stack_height_mm"]["heel"])
    overhang = (float(equipment["boot_sole_width_mm"]) - float(equipment["ski_waist_mm"])) / 2.0
    return float(np.arctan2(stack, overhang))


# --- terrain interpolation (Catmull-Rom bicubic, C1) ---------------------------------------------


def _cr_weights(t):
    t2, t3 = t * t, t * t * t
    return jnp.stack([(-t3 + 2 * t2 - t) / 2, (3 * t3 - 5 * t2 + 2) / 2, (-3 * t3 + 4 * t2 + t) / 2, (t3 - t2) / 2])


def _extrapolated(Z, idx, axis):
    """Z gathered at integer indices along `axis`, linearly extrapolated past either edge, so a
    plane is reproduced exactly up to and beyond the grid boundary (clamping would halve slopes
    at the edge rows)."""
    size = Z.shape[axis]
    inside = jnp.clip(idx, 0, size - 1)
    vals = jnp.take(Z, inside, axis=axis)
    first, second = jnp.take(Z, jnp.array([0]), axis=axis), jnp.take(Z, jnp.array([1]), axis=axis)
    last, before_last = jnp.take(Z, jnp.array([size - 1]), axis=axis), jnp.take(Z, jnp.array([size - 2]), axis=axis)
    shape = [1] * Z.ndim
    shape[axis] = -1
    below = jnp.minimum(idx, 0).reshape(shape).astype(Z.dtype)
    above = jnp.maximum(idx - (size - 1), 0).reshape(shape).astype(Z.dtype)
    return vals + below * (second - first) + above * (last - before_last)


def height(grid: Grid3D, s, n):
    """Z at (s, n): bicubic (Catmull-Rom) on the course grid, linearly extrapolated at the edges."""
    fi = s / grid.ds
    fj = (n - grid.n0) / grid.dn
    i0, j0 = jnp.floor(fi), jnp.floor(fj)
    ti, tj = fi - i0, fj - j0
    rows = i0.astype(jnp.int32) + jnp.arange(-1, 3)
    cols = j0.astype(jnp.int32) + jnp.arange(-1, 3)
    i_lo = jnp.clip(i0.astype(jnp.int32) - 1, 0, grid.Z.shape[0] - 4)
    j_lo = jnp.clip(j0.astype(jnp.int32) - 1, 0, grid.Z.shape[1] - 4)
    # Work on a 4-wide band first (cheap), then extrapolate within it if the stencil leaves the grid.
    band = jax.lax.dynamic_slice(grid.Z, (i_lo, jnp.zeros_like(i_lo)), (4, grid.Z.shape[1]))
    band = _extrapolated(band, rows - i_lo, 0)
    patch = jax.lax.dynamic_slice(band, (jnp.zeros_like(j_lo), j_lo), (4, 4))
    patch = _extrapolated(patch, cols - j_lo, 1)
    return _cr_weights(ti) @ patch @ _cr_weights(tj)


def surface(grid: Grid3D, s, n):
    """z, z_s, z_n, z_ss, z_sn, z_nn at (s, n): bilinear in the precomputed fields. Outside the grid
    the derivatives hold their edge values and z continues linearly, so a plane stays exact."""
    rows, cols = grid.F.shape[1], grid.F.shape[2]
    fi = s / grid.ds
    fj = (n - grid.n0) / grid.dn
    ci, cj = jnp.clip(fi, 0.0, rows - 1.0), jnp.clip(fj, 0.0, cols - 1.0)
    i0 = jnp.clip(jnp.floor(ci).astype(jnp.int32), 0, rows - 2)
    j0 = jnp.clip(jnp.floor(cj).astype(jnp.int32), 0, cols - 2)
    ti, tj = ci - i0, cj - j0
    # Smoothstep blend (3t^2 - 2t^3): C1 across grid lines, so gradients have no kinks where a line
    # crosses a node row or column (plain bilinear weights would put a kink at every grid line).
    ti, tj = ti * ti * (3 - 2 * ti), tj * tj * (3 - 2 * tj)
    patch = jax.lax.dynamic_slice(grid.F, (jnp.zeros_like(i0), i0, j0), (6, 2, 2))
    vals = ((1 - ti) * (1 - tj) * patch[:, 0, 0] + ti * (1 - tj) * patch[:, 1, 0]
            + (1 - ti) * tj * patch[:, 0, 1] + ti * tj * patch[:, 1, 1])
    z, zs, zn, zss, zsn, znn = vals
    z = z + zs * (fi - ci) * grid.ds + zn * (fj - cj) * grid.dn  # linear continuation past the edge
    return z, zs, zn, zss, zsn, znn


def local_geometry(grid: Grid3D, s, n):
    """Horizontal gradient and Hessian of z in the orthonormal (t, l) frame at (s, n), including the
    curvilinear terms from the frame's own curvature (scale factor h = 1 - n kappa_c)."""
    z, zs, zn, zss, zsn, znn = surface(grid, s, n)
    k = jnp.interp(s, jnp.arange(grid.kappa_c.shape[0]) * grid.ds, grid.kappa_c)
    dk = jnp.interp(s, jnp.arange(grid.dkappa_c.shape[0]) * grid.ds, grid.dkappa_c)
    # Floor on the scale factor: a line far off the piste (already a DNF) must not drive the frame
    # degenerate and produce NaNs that would poison gradients.
    h = jnp.maximum(1.0 - n * k, 0.2)
    gt, gl = zs / h, zn
    d_s_of_zs_over_h = zss / h + zs * n * dk / h**2
    Htt = (d_s_of_zs_over_h - k * zn) / h
    Htl = zsn / h + zs * k / h**2
    Hll = znn
    return z, h, k, jnp.stack([gt, gl]), jnp.array([[Htt, Htl], [Htl, Hll]])


def path_loads(grid: Grid3D, s, n, psi, kappa_h, v, g):
    """Normal and lateral load, geodesic and normal curvature, along-track gravity and the 3D
    stretch factor q for a path at (s, n) with heading psi and horizontal curvature kappa_h."""
    z, h, kc, grad, H = local_geometry(grid, s, n)
    hh = jnp.stack([jnp.cos(psi), jnp.sin(psi)])        # horizontal tangent in (t, l)
    m = jnp.stack([-jnp.sin(psi), jnp.cos(psi)])        # horizontal left normal
    slope_t = grad @ hh
    r1 = jnp.concatenate([hh, slope_t[None]])
    q = jnp.sqrt(1.0 + slope_t**2)
    T = r1 / q
    r2 = jnp.concatenate([kappa_h * m, (kappa_h * (grad @ m) + hh @ H @ hh)[None]])
    kvec = (r2 - (r2 @ T) * T) / q**2
    u = jnp.concatenate([-grad, jnp.ones(1)]) / jnp.sqrt(1.0 + grad @ grad)
    b = jnp.cross(u, T)
    kappa_n, kappa_g = kvec @ u, kvec @ b
    N = v**2 * kappa_n + g * u[2]
    F_lat = v**2 * kappa_g + g * b[2]
    return dict(z=z, h=h, kappa_c=kc, q=q, T_z=T[2], kappa_n=kappa_n, kappa_g=kappa_g, N=N, F_lat=F_lat,
                pitch=jnp.arctan(-slope_t))


# --- step and rollout ----------------------------------------------------------------------------


class State3D(NamedTuple):
    n: jnp.ndarray
    psi: jnp.ndarray
    v: jnp.ndarray


SOFT_ABS_EPS = 0.5  # m/s^2: smooths |F_lat| through zero at each turn transition (loads are 15-30 m/s^2)


def soft_abs(x, eps=SOFT_ABS_EPS):
    """sqrt(x^2 + eps^2) - eps: |x| without the corner at 0, so gradients see no kink each time a
    turn changes direction."""
    return jnp.sqrt(x * x + eps * eps) - eps


def turn_losses(kappa_g, F_lat, v, p: SimParams):
    """The 2D turning bands, applied to the carved (geodesic) curvature and the lateral load."""
    abs_k = jnp.abs(kappa_g)
    tight = softplus_b(abs_k - 1.0 / p.r_typical_lo, p.beta_kappa) * v**2
    skid = softplus_b(abs_k - 1.0 / p.r_skid, p.beta_kappa) * v**2
    return p.w_turn * p.k_turn * soft_abs(F_lat) + p.w_tight * p.k_tight * tight + p.w_skid * p.k_skid * skid


def step(state: State3D, kappa_cmd, s, grid: Grid3D, p: SimParams):
    n, psi, v = state
    kappa_h = clip_kappa(kappa_cmd, p)
    L = path_loads(grid, s, n, psi, kappa_h, v, p.g)
    ds = grid.ds
    d_sigma = L["h"] * ds / jnp.cos(psi)
    d_ell = L["q"] * d_sigma
    n_new = n + jnp.tan(psi) * L["h"] * ds
    psi_new = limit_heading(psi + kappa_h * d_sigma - L["kappa_c"] * ds, p)
    force = (
        -p.w_gravity * p.g * L["T_z"]
        - p.w_friction * p.mu * jnp.maximum(L["N"], 0.0)
        - p.w_drag * p.c_d * v**2
        - turn_losses(L["kappa_g"], L["F_lat"], v, p)
    )
    v_new = jnp.sqrt(jnp.maximum(v**2 + 2.0 * force * d_ell, p.v_floor**2))
    dt = 2.0 * d_ell / (v + v_new)
    G = jnp.sqrt(L["N"] ** 2 + L["F_lat"] ** 2) / p.g
    edge = jnp.arctan2(soft_abs(L["F_lat"]) + SOFT_ABS_EPS, jnp.maximum(L["N"], 1e-6))
    return State3D(n_new, psi_new, v_new), dict(L, dt=dt, d_ell=d_ell, G=G, edge=edge, kappa_h=kappa_h)


def advance(state: State3D, kappa_cmd, k, course: Course3D, p: SimParams, p3: Params3D = Params3D()):
    """One course step k with full bookkeeping: dynamics, hazards, hard limits, walls, telemetry.
    Shared by the full rollout and by segment simulations (optim/dp.py), so both use identical physics."""
    grid = course.terrain
    dtype = jnp.result_type(float)
    r = p.risk
    s = k * grid.ds
    new_state, o = step(state, kappa_cmd, s, grid, p)
    active = (s < course.length).astype(dtype)
    new_state = jax.tree.map(lambda a, b: jnp.where(active > 0, a, b), new_state, state)
    k_req = hazard.kappa_required(state.n, s, state.psi, course, r.clearance, r.kappa_req_cap)
    h4 = hazard.hazard_terms(o["G"], o["kappa_g"], k_req, state.v, s, r)
    # 3D-only hazards fold into the "load" column (edge/unloading are load failures).
    h_edge = p3.c_edge * softplus_b(o["edge"] - p3.edge_safe, p3.beta_edge)
    h_air = p3.c_air * softplus_b(p3.n_min - o["N"] / p.g, p3.beta_air)
    h4 = h4.at[0].add(h_edge + h_air)
    # G and speed limits from the shared hazard code; the piste check uses the real edges.
    broken, wall = hazard.hard_limits(o["G"], state.v, jnp.zeros_like(state.n), jnp.inf, r)
    if course.piste_lo is None:
        out_by = jnp.abs(state.n) - course.piste_half_width
    else:
        s_rows = jnp.arange(course.piste_lo.shape[0]) * grid.ds
        out_by = jnp.maximum(jnp.interp(s, s_rows, course.piste_lo) - state.n,
                             state.n - jnp.interp(s, s_rows, course.piste_hi))
    broken = jnp.maximum(broken, (r.piste_on * (out_by > 0)).astype(dtype))
    wall = wall + r.wall_weight * r.piste_on * softplus_b(out_by, r.wall_beta)
    beta_edge_wall = jnp.where(p3.wall_beta_edge > 0, p3.wall_beta_edge, r.wall_beta)
    air = p3.airborne_dnf * (o["N"] < 0)
    boot = p3.edge_dnf * (o["edge"] > p3.edge_max)
    broken = jnp.maximum(broken, jnp.maximum(air, boot).astype(dtype))
    wall = wall + r.wall_weight * (p3.airborne_dnf * softplus_b(-o["N"] / p.g, r.wall_beta)
                                   + p3.edge_dnf * softplus_b(o["edge"] - p3.edge_max, beta_edge_wall))
    d_ell = o["d_ell"] * active
    tel = Telemetry(x=state.n, y=s, psi=state.psi, v=state.v, kappa=o["kappa_g"], a_net=o["F_lat"],
                    G=o["G"], theta=o["pitch"], z=o["z"], N_load=o["N"], edge=o["edge"])
    out = StepOutput(dt=o["dt"] * active, ds=d_ell, active=active, telemetry=tel, hazard=h4 * d_ell,
                     hard_dnf=broken * active, wall=wall * d_ell, kappa_req=k_req)
    return new_state, out


def rollout3d(policy_act, params, course: Course3D, p: SimParams, n_steps: int, p3: Params3D = Params3D()):
    """Fixed-length scan over s; returns (Summary, StepOutput) exactly like the 2D rollout, so the
    objectives, evaluator, optimizers and logger are unchanged."""
    grid = course.terrain
    dtype = jnp.result_type(float)
    state0 = State3D(jnp.asarray(course.x_start, dtype), jnp.zeros((), dtype), jnp.asarray(p.v0, dtype))

    def body(state, k):
        return advance(state, policy_act(params, None, k), k, course, p, p3)

    final, traj = jax.lax.scan(body, state0, jnp.arange(n_steps))
    T = jnp.sum(traj.dt)
    D = jnp.sum(traj.ds)
    G = jnp.where(traj.active > 0, traj.telemetry.G, 0.0)
    gate_idx = jnp.clip(jnp.round(course.gates.y / grid.ds).astype(jnp.int32), 0, n_steps - 1)
    H_by_cause = jnp.sum(traj.hazard, axis=0)
    summary = Summary(T=T, D=D, v_mean=D / T, v_end=final.v, max_G=jnp.max(G), x_gates=traj.telemetry.x[gate_idx],
                      H=jnp.sum(H_by_cause), H_by_cause=H_by_cause, hard_dnf=jnp.max(traj.hard_dnf),
                      wall=jnp.sum(traj.wall))
    return summary, traj


# --- building a Course3D -------------------------------------------------------------------------


def grid_from(course_terrain, smooth_m: float = 0.0) -> Grid3D:
    """Grid3D from terrain3d.CourseTerrain. `smooth_m` is a Gaussian smoothing scale for the snow
    surface (the DTM resolves metre-scale bumps that groomed snow does not have)."""
    from scipy.ndimage import gaussian_filter

    Z = np.asarray(course_terrain.Z, float)
    # Heights relative to the start keep float32 values small: the curvature terms difference Z twice,
    # and absolute heights (~1,800 m) would cost about 0.1 m/s^2 of rounding noise in the normal load.
    Z = Z - Z[0, Z.shape[1] // 2]
    if smooth_m > 0:
        Z = gaussian_filter(Z, sigma=(smooth_m / course_terrain.ds, smooth_m / course_terrain.dn), mode="nearest")
    return make_grid(Z, course_terrain.ds, float(course_terrain.n[0]), course_terrain.dn, course_terrain.frame.kappa)


def plane_course(theta_deg: float, length: float, half_width: float = 30.0, ds: float = 1.0, dn: float = 0.5,
                 cross_deg: float = 0.0, gates: Gates | None = None) -> Course3D:
    """A straight course on an inclined plane: z = -tan(theta) s + tan(cross) n. For tests."""
    s = np.arange(0.0, length + 4 * ds, ds)
    n = np.arange(-half_width, half_width + dn / 2, dn)
    Z = -np.tan(np.radians(theta_deg)) * s[:, None] + np.tan(np.radians(cross_deg)) * n[None, :]
    grid = make_grid(Z, ds, float(n[0]), dn)
    from skiopt.courses import no_gates

    return Course3D(length=jnp.asarray(float(length)), x_start=jnp.asarray(0.0), piste_half_width=jnp.asarray(20.0),
                    terrain=grid, gates=gates if gates is not None else no_gates())
