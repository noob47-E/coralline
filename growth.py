"""Differential-growth engine.

Grows ONE closed curve that never crosses itself. The simulation works in
"spacing units": the repulsion radius is 1.0, which becomes `spacing_mm`
millimetres on export. Every step:

  1. smoothing  - each node is pulled toward the midpoint of its neighbours
  2. springs    - neighbouring nodes keep a rest distance along the curve
  3. repulsion  - nodes from different parts of the curve push apart (radius 1)
  4. walls      - outer boundary, centre hole and radial channels push nodes away
  5. growth     - new nodes are inserted, preferring free space and curved tips
"""
from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass, fields

import numpy as np
from scipy.spatial import cKDTree

from geometry import closed_length, count_self_intersections, resample_closed

BOUNDARIES = ["circle", "ellipse", "triangle", "square", "hexagon", "octagon"]
START_SHAPES = ["ring", "circle", "star"]
_SIDES = {"triangle": 3, "square": 4, "hexagon": 6, "octagon": 8}

# Simulation constants (spacing units)
REST = 0.42        # rest length of an edge
MAX_EDGE = 0.7     # longer edges are split
MIN_EDGE = 0.14    # shorter edges are collapsed
MAX_STEP = 0.12    # max movement of a node per step
WALL_CLEAR = 0.35  # hard minimum distance from any wall
LINE_GAP = 0.92    # typical distance between neighbouring strands when packed


@dataclass
class GrowthParams:
    # size
    diameter_mm: float = 200.0
    spacing_mm: float = 4.5
    boundary: str = "circle"
    aspect: float = 1.0
    rotation_deg: float = 0.0
    # growth
    fill: float = 0.8
    growth_speed: float = 0.5
    branchiness: float = 0.6
    smoothness: float = 0.5
    wiggle: float = 0.3
    max_iterations: int = 5000
    # start shape and structure
    start_shape: str = "circle"
    start_size: float = 0.12
    star_arms: int = 5
    hole: float = 0.0
    channels: int = 0
    channel_width: float = 1.5
    channel_length: float = 1.0
    channel_rotation_deg: float = 90.0
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "GrowthParams":
        names = {f.name for f in fields(cls)}
        out = cls()
        for k, v in d.items():
            if k in names:
                cur = getattr(out, k)
                setattr(out, k, type(cur)(v))
        return out

    def to_dict(self) -> dict:
        return asdict(self)


class DifferentialGrowth:
    def __init__(self, params: GrowthParams):
        self.p = params
        self.rng = np.random.default_rng(params.seed)
        self.iteration = 0
        self.done = False
        self.stop_reason = ""
        self.Rb = max(4.0, 0.5 * params.diameter_mm / params.spacing_mm)

        self._build_boundary()
        self.hole_r = params.hole * self._rb_min if params.hole > 0 else 0.0
        self._build_seed()
        self._build_channels()
        self.P = self._seed_pts
        self._estimate_target()
        self._length_history: list[float] = []

        s = params.smoothness
        self.k_smooth = 0.04 + 0.36 * s
        self.k_spring = 0.25
        self.k_rep = 0.35
        self.k_wall = 0.5
        self.noise = 0.004 + 0.03 * params.wiggle
        self.grow_prob = 0.002 + 0.028 * params.growth_speed ** 1.5

    # ------------------------------------------------------------------ setup
    def _polar_boundary_r(self, theta: np.ndarray) -> np.ndarray:
        return np.interp(np.mod(theta, 2 * np.pi), self._b_theta, self._b_r, period=2 * np.pi)

    def _build_boundary(self):
        p = self.p
        rot = math.radians(p.rotation_deg)
        t = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
        if p.boundary in _SIDES:
            n = _SIDES[p.boundary]
            apothem = self.Rb * math.cos(math.pi / n)
            r = apothem / np.cos(np.mod(t, 2 * np.pi / n) - math.pi / n)
        else:
            r = np.full_like(t, self.Rb)
        pts = np.column_stack([r * np.cos(t), r * np.sin(t)])
        aspect = p.aspect if p.boundary != "circle" else 1.0
        pts[:, 1] *= aspect
        c, s = math.cos(rot), math.sin(rot)
        pts = pts @ np.array([[c, s], [-s, c]])
        pts = resample_closed(pts, 0.1)
        th = np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2 * np.pi)
        rr = np.hypot(pts[:, 0], pts[:, 1])
        order = np.argsort(th)
        self._b_theta, self._b_r = th[order], rr[order]
        self._b_pts = pts
        self._b_tree = cKDTree(pts)
        self._rb_min = float(rr.min())

    def _random_wobble(self, theta: np.ndarray, amp: float) -> np.ndarray:
        w = np.zeros_like(theta)
        for k in range(2, 8):
            w += self.rng.normal() / k * np.sin(k * theta + self.rng.uniform(0, 2 * np.pi))
        return amp * w / max(1e-9, np.abs(w).max())

    def _seed_radius(self, theta: np.ndarray) -> np.ndarray:
        p = self.p
        base = max(1.6, self.hole_r + 1.6)
        if p.start_shape == "ring":
            return self._polar_boundary_r(theta) - 1.3
        if p.start_shape == "star":
            amp = max(2.0, p.start_size * self.Rb)
            k = max(2, int(p.star_arms))
            lobes = 0.5 + 0.5 * np.cos(k * (theta - math.radians(p.channel_rotation_deg)))
            return base + amp * lobes ** 2
        return np.full_like(theta, max(base, p.start_size * self.Rb))

    def _build_seed(self):
        t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        r = self._seed_radius(t)
        if self.p.start_shape == "ring":
            r = r - np.abs(self._random_wobble(t, 0.3))
        else:
            r = r + self._random_wobble(t, 0.3)
        pts = np.column_stack([r * np.cos(t), r * np.sin(t)])
        self._seed_pts = resample_closed(pts, REST)

    def _build_channels(self):
        p = self.p
        self.channels = []
        if p.channels <= 0:
            return
        half = 0.5 * p.channel_width
        rot = math.radians(p.channel_rotation_deg)
        for k in range(int(p.channels)):
            th = rot + 2 * math.pi * k / p.channels
            rb = float(self._polar_boundary_r(np.array([th]))[0])
            rs = float(self._seed_radius(np.array([th]))[0])
            if p.start_shape == "ring":
                r_in = self.hole_r
                r_out = min(p.channel_length * rb, rs - half - 1.4)
            else:
                r_in = rs + 0.3 + half + 1.4
                r_out = p.channel_length * rb + (half + 1.0 if p.channel_length >= 0.999 else 0.0)
            if r_out > r_in + 0.5:
                u = np.array([math.cos(th), math.sin(th)])
                self.channels.append((u, r_in, r_out, half))

    def _allowed_mask(self, pts: np.ndarray) -> np.ndarray:
        th = np.arctan2(pts[:, 1], pts[:, 0])
        r = np.hypot(pts[:, 0], pts[:, 1])
        ok = r < self._polar_boundary_r(th)
        if self.hole_r > 0:
            ok &= r > self.hole_r
        for u, r_in, r_out, half in self.channels:
            t = np.clip(pts @ u, r_in, r_out)
            ok &= np.hypot(pts[:, 0] - t * u[0], pts[:, 1] - t * u[1]) > half
        return ok

    def _estimate_target(self):
        h = 0.5
        R = float(self._b_r.max())
        g = np.arange(-R, R + h, h)
        X, Y = np.meshgrid(g, g)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        area = self._allowed_mask(pts).sum() * h * h
        self.full_length = area / LINE_GAP
        self.target_length = max(closed_length(self.P) * 1.05, self.p.fill * self.full_length)

    # ------------------------------------------------------------ simulation
    def step(self, grow: bool = True):
        P = self.P
        N = len(P)
        F = np.zeros_like(P)
        prev = np.roll(P, 1, axis=0)
        nxt = np.roll(P, -1, axis=0)

        # 1. smoothing
        F += self.k_smooth * (0.5 * (prev + nxt) - P)

        # 2. springs along the curve
        e = nxt - P
        L = np.maximum(np.linalg.norm(e, axis=1), 1e-9)
        fs = (self.k_spring * (L - REST) / L)[:, None] * e
        F += fs
        F -= np.roll(fs, 1, axis=0)

        # 3. repulsion between non-adjacent nodes
        pressure = np.zeros(N)
        pairs = cKDTree(P).query_pairs(1.0, output_type="ndarray")
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            hop = np.abs(i - j)
            keep = np.minimum(hop, N - hop) > 2
            i, j = i[keep], j[keep]
            d = P[i] - P[j]
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
            mag = 1.0 - dist
            f = (self.k_rep * mag / dist)[:, None] * d
            F[:, 0] += np.bincount(i, f[:, 0], N) - np.bincount(j, f[:, 0], N)
            F[:, 1] += np.bincount(i, f[:, 1], N) - np.bincount(j, f[:, 1], N)
            pressure += np.bincount(i, mag, N) + np.bincount(j, mag, N)

        # 4. walls (soft force)
        pressure += self._wall_forces(P, F)

        # noise
        F += self.rng.normal(scale=self.noise, size=P.shape)

        # integrate with a step limit
        step_len = np.linalg.norm(F, axis=1)
        scale = np.minimum(1.0, MAX_STEP / np.maximum(step_len, 1e-12))
        P = P + F * scale[:, None]
        P = self._clamp_walls(P)

        # 5. topology: grow, collapse tiny edges, split long edges
        if grow:
            P = self._grow(P, pressure)
        P = self._collapse_short(P)
        P = self._split_long(P)

        self.P = P
        self.iteration += 1

    def _wall_forces(self, P: np.ndarray, F: np.ndarray) -> np.ndarray:
        pressure = np.zeros(len(P))
        # outer boundary
        d, idx = self._b_tree.query(P, distance_upper_bound=1.0)
        near = np.isfinite(d)
        if near.any():
            v = P[near] - self._b_pts[idx[near]]
            dn = np.maximum(d[near], 1e-6)
            mag = 1.0 - dn
            F[near] += (self.k_wall * mag / dn)[:, None] * v
            pressure[near] += mag
        r = np.maximum(np.hypot(P[:, 0], P[:, 1]), 1e-9)
        # centre hole
        if self.hole_r > 0:
            dw = r - self.hole_r
            m = dw < 1.0
            if m.any():
                mag = 1.0 - np.maximum(dw[m], 0.0)
                F[m] += (self.k_wall * mag / r[m])[:, None] * P[m]
                pressure[m] += mag
        # channels
        for u, r_in, r_out, half in self.channels:
            t = np.clip(P @ u, r_in, r_out)
            v = P - t[:, None] * u
            dist = np.maximum(np.linalg.norm(v, axis=1), 1e-9)
            dw = dist - half
            m = dw < 1.0
            if m.any():
                mag = 1.0 - np.maximum(dw[m], 0.0)
                F[m] += (self.k_wall * mag / dist[m])[:, None] * v[m]
                pressure[m] += mag
        return pressure

    def _clamp_walls(self, P: np.ndarray) -> np.ndarray:
        th = np.arctan2(P[:, 1], P[:, 0])
        r = np.maximum(np.hypot(P[:, 0], P[:, 1]), 1e-9)
        rmax = self._polar_boundary_r(th) - WALL_CLEAR
        rmin = self.hole_r + WALL_CLEAR if self.hole_r > 0 else 0.0
        r_new = np.clip(r, rmin, rmax)
        P = P * (r_new / r)[:, None]
        for u, r_in, r_out, half in self.channels:
            t = np.clip(P @ u, r_in, r_out)
            v = P - t[:, None] * u
            dist = np.linalg.norm(v, axis=1)
            m = dist < half + WALL_CLEAR
            if m.any():
                perp = np.array([-u[1], u[0]])
                side = np.sign(v[m] @ perp)
                side[side == 0] = 1.0
                dirs = np.where(dist[m, None] > 1e-6, v[m] / np.maximum(dist[m, None], 1e-6), side[:, None] * perp)
                P[m] = t[m, None] * u + dirs * (half + WALL_CLEAR)
        return P

    def _collapse_short(self, P: np.ndarray) -> np.ndarray:
        L = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
        short = L < MIN_EDGE
        if not short.any() or len(P) < 10:
            return P
        # remove node i+1 for short edge i, never two neighbours in one pass
        rm = np.roll(short, 1)
        rm &= ~np.roll(rm, 1)
        return P[~rm]

    def _curvature(self, P: np.ndarray) -> np.ndarray:
        a = P - np.roll(P, 1, axis=0)
        b = np.roll(P, -1, axis=0) - P
        ang = np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(axis=1)))
        k = np.ones(7) / 7.0
        ext = np.concatenate([ang[-3:], ang, ang[:3]])
        return np.convolve(ext, k, mode="valid")

    def _grow(self, P: np.ndarray, pressure) -> np.ndarray:
        if closed_length(P) >= self.target_length:
            return P
        N = len(P)
        freedom = np.exp(-pressure / 0.35)
        b = self.p.branchiness
        curv = np.clip(self._curvature(P) / 0.35, 0.0, 1.0)
        w = freedom * ((1.0 - b) + b * curv)
        w_edge = 0.5 * (w + np.roll(w, -1))
        L = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
        pick = (self.rng.random(N) < self.grow_prob * w_edge) & (L > 2.2 * MIN_EDGE)
        return self._insert_midpoints(P, pick)

    def _split_long(self, P: np.ndarray) -> np.ndarray:
        L = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
        return self._insert_midpoints(P, L > MAX_EDGE)

    @staticmethod
    def _insert_midpoints(P: np.ndarray, pick: np.ndarray) -> np.ndarray:
        idx = np.nonzero(pick)[0]
        if len(idx) == 0:
            return P
        mids = 0.5 * (P[idx] + P[(idx + 1) % len(P)])
        return np.insert(P, idx + 1, mids, axis=0)

    # ---------------------------------------------------------------- driver
    @property
    def length(self) -> float:
        return closed_length(self.P)

    @property
    def progress(self) -> float:
        return min(1.0, self.length / self.target_length)

    def run(self, stop_event: threading.Event | None = None, callback=None, callback_every: int = 5,
            relax_steps: int = 60):
        p = self.p
        while self.iteration < p.max_iterations:
            if stop_event is not None and stop_event.is_set():
                self.stop_reason = "stopped"
                break
            self.step(grow=True)
            L = self.length
            self._length_history.append(L)
            if L >= self.target_length:
                self.stop_reason = "fill reached"
                break
            if self.iteration > 400:
                old = self._length_history[-300]
                if L < old * 1.004:
                    self.stop_reason = "no more room to grow"
                    break
            if callback is not None and self.iteration % callback_every == 0:
                callback(self)
        else:
            self.stop_reason = "iteration limit"
        # relax without growth so spacing evens out
        if not (stop_event is not None and stop_event.is_set()):
            for _ in range(relax_steps):
                self.step(grow=False)
        self.done = True
        if callback is not None:
            callback(self)
        return self

    # ---------------------------------------------------------------- output
    def points_mm(self) -> np.ndarray:
        return self.P * self.p.spacing_mm

    def boundary_mm(self) -> np.ndarray:
        return self._b_pts * self.p.spacing_mm

    def self_intersections(self) -> int:
        return count_self_intersections(self.P)
