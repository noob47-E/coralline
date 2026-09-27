# Coralline - generative coral lamp patterns drawn as one continuous line
# Copyright (C) 2026 Muhammad Wasiq
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# LICENSE file or <https://www.gnu.org/licenses/> for details.
# SPDX-License-Identifier: GPL-3.0-or-later

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

# Sleeping (speed-up for big patterns, see DifferentialGrowth.step)
SLEEP_MIN_NODES = 3000  # smaller patterns always simulate every node
SLEEP_EVERY = 10        # every Nth step is a full step that re-checks every node
SLEEP_DRIFT = 0.08      # a node that moved less than this since the last full step may sleep
WAKE_MOVE = 0.02        # a node pushed harder than this (per step) wakes / stays awake
SLEEP_PRESSURE = 0.25   # only crowded nodes sleep (free space means it may still grow)
SLEEP_K = 16            # max sleeping neighbours looked up per awake node


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
    max_iterations: int = 20000
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
        self._length = closed_length(self.P)

        # per-node state, kept parallel to self.P
        n = len(self.P)
        self.ids = np.arange(n)            # stable id of each node
        self._next_id = n
        self.asleep = np.zeros(n, dtype=bool)
        self._anchor = self.P.copy()       # position at the last full step
        self._young = np.ones(n, dtype=bool)
        self._allow_sleep = True
        self._sleep_tree = None
        self._n_sleep = 0
        self._pos_of_id = None

        s = params.smoothness
        self.k_smooth = 0.04 + 0.36 * s
        self.k_spring = 0.25
        self.k_rep = 0.35
        self.k_wall = 0.5
        self.dt = 0.5  # time step; 1.0 made nodes overshoot and bounce every step
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
        # cos of the worst angle between the radial direction and the wall normal; lets
        # _wall_forces skip nodes that are provably out of reach of the boundary
        tan = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
        normal = np.column_stack([tan[:, 1], -tan[:, 0]]) / np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-12)[:, None]
        cos = np.abs((normal * pts).sum(axis=1)) / np.maximum(rr, 1e-12)
        self._b_cos_min = max(0.1, float(cos.min()) - 1.5 / self._rb_min)

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
    #
    # Speed-up: most of a big pattern is finished and just sits still, packed tight.
    # Such nodes are put to "sleep": they stop moving and act as fixed obstacles.
    # Only awake nodes (the growing parts) are simulated every step. Every
    # SLEEP_EVERY steps a full step re-checks every node and wakes any that are pushed.

    def step(self, grow: bool = True, full: bool | None = None):
        P = self.P
        N = len(P)
        if full is None:
            every = SLEEP_EVERY if N < 12000 else 2 * SLEEP_EVERY  # full steps are costly when huge
            full = (N < SLEEP_MIN_NODES or self._sleep_tree is None
                    or self.iteration % every == 0)
        act = np.arange(N) if full else np.flatnonzero(~self.asleep)
        if len(act) == 0:
            act, full = np.arange(N), True
        if not full:
            self._pos_of_id = np.full(self._next_id, -1, dtype=np.int64)
            self._pos_of_id[self.ids] = np.arange(N)

        F, pressure = self._forces(P, act, full)

        # integrate: time step, noise on awake nodes, step limit
        disp = F * self.dt
        det_move = np.hypot(disp[:, 0], disp[:, 1])
        noisy = ~self.asleep[act] if full else slice(None)
        disp[noisy] += self.rng.normal(scale=self.noise, size=disp[noisy].shape)
        move = np.hypot(disp[:, 0], disp[:, 1])
        disp *= np.minimum(1.0, MAX_STEP / np.maximum(move, 1e-12))[:, None]
        if full:
            sleeping = self.asleep.copy()
            disp[sleeping] = np.where(det_move[sleeping, None] > WAKE_MOVE, disp[sleeping], 0.0)
        P[act] = self._clamp_walls(P[act] + disp)

        if full:
            self._update_sleep(P, det_move, pressure)

        # topology: grow, collapse tiny edges, split long edges
        if grow and self._length < self.target_length:
            self._grow(P, act, pressure, full)
        self._collapse_short()
        L = self._split_long()
        self._length = float(L.sum())

        if full:
            self._build_sleep_tree()
        self.iteration += 1

    def _forces(self, P: np.ndarray, act: np.ndarray, full: bool):
        """Forces on the nodes `act` (all nodes on a full step). Returns (F, pressure) for them."""
        N = len(P)
        na = len(act)
        p = P[act]
        prv = P[act - 1]            # index -1 wraps to the last node
        nxt = P[(act + 1) % N]

        # 1. smoothing
        F = self.k_smooth * (0.5 * (prv + nxt) - p)
        # 2. springs to both neighbours
        en = nxt - p
        ep = p - prv
        ln = np.maximum(np.hypot(en[:, 0], en[:, 1]), 1e-9)
        lp = np.maximum(np.hypot(ep[:, 0], ep[:, 1]), 1e-9)
        F += (self.k_spring * (ln - REST) / ln)[:, None] * en
        F -= (self.k_spring * (lp - REST) / lp)[:, None] * ep

        # 3. repulsion between awake nodes (both feel it)
        pressure = np.zeros(na)
        pairs = cKDTree(p).query_pairs(1.0, output_type="ndarray")
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            hop = np.abs(act[a] - act[b])
            keep = np.minimum(hop, N - hop) > 2
            a, b = a[keep], b[keep]
            d = p[a] - p[b]
            dist = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
            mag = 1.0 - dist
            f = (self.k_rep * mag / dist)[:, None] * d
            idx = np.concatenate([a, b])
            F[:, 0] += np.bincount(idx, np.concatenate([f[:, 0], -f[:, 0]]), na)
            F[:, 1] += np.bincount(idx, np.concatenate([f[:, 1], -f[:, 1]]), na)
            pressure += np.bincount(idx, np.concatenate([mag, mag]), na)

        # 3b. repulsion from sleeping nodes (fixed obstacles), partial steps only
        if not full and self._n_sleep:
            dd, kk = self._sleep_tree.query(p, k=SLEEP_K, distance_upper_bound=1.0)
            rows, cols = np.nonzero(np.isfinite(dd))
            if len(rows):
                sk = kk[rows, cols]
                g = self._pos_of_id[self._sleep_ids[sk]]
                hop = np.abs(act[rows] - g)
                ok = (g >= 0) & (np.minimum(hop, N - hop) > 2)
                ok[ok] = self.asleep[g[ok]]  # woken since the last full step: already in the awake set
                rows, sk, dist = rows[ok], sk[ok], dd[rows[ok], cols[ok]]
                d = p[rows] - self._sleep_pts[sk]
                dist = np.maximum(dist, 1e-6)
                mag = 1.0 - dist
                f = (self.k_rep * mag / dist)[:, None] * d
                F[:, 0] += np.bincount(rows, f[:, 0], na)
                F[:, 1] += np.bincount(rows, f[:, 1], na)
                pressure += np.bincount(rows, mag, na)

        # 4. walls
        pressure += self._wall_forces(p, F)
        return F, pressure

    def _wall_forces(self, P: np.ndarray, F: np.ndarray) -> np.ndarray:
        pressure = np.zeros(len(P))
        r = np.maximum(np.hypot(P[:, 0], P[:, 1]), 1e-9)
        # outer boundary: only nodes that can be within reach of it
        cand = np.flatnonzero(r > self._rb_min - 1.05 / self._b_cos_min)
        if len(cand):
            gap = self._polar_boundary_r(np.arctan2(P[cand, 1], P[cand, 0])) - r[cand]
            cand = cand[gap * self._b_cos_min < 1.05]
        if len(cand):
            d, idx = self._b_tree.query(P[cand], distance_upper_bound=1.0)
            near = np.isfinite(d)
            if near.any():
                ci = cand[near]
                v = P[ci] - self._b_pts[idx[near]]
                dn = np.maximum(d[near], 1e-6)
                mag = 1.0 - dn
                F[ci] += (self.k_wall * mag / dn)[:, None] * v
                pressure[ci] += mag
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
            dist = np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-9)
            dw = dist - half
            m = dw < 1.0
            if m.any():
                mag = 1.0 - np.maximum(dw[m], 0.0)
                F[m] += (self.k_wall * mag / dist[m])[:, None] * v[m]
                pressure[m] += mag
        return pressure

    def _clamp_walls(self, P: np.ndarray) -> np.ndarray:
        r = np.maximum(np.hypot(P[:, 0], P[:, 1]), 1e-9)
        rmax = np.full(len(P), np.inf)
        outer = np.flatnonzero(r > self._rb_min - WALL_CLEAR)  # the only ones that can be outside
        if len(outer):
            rmax[outer] = self._polar_boundary_r(np.arctan2(P[outer, 1], P[outer, 0])) - WALL_CLEAR
        rmin = self.hole_r + WALL_CLEAR if self.hole_r > 0 else 0.0
        r_new = np.clip(r, rmin, rmax)
        P = P * (r_new / r)[:, None]
        for u, r_in, r_out, half in self.channels:
            t = np.clip(P @ u, r_in, r_out)
            v = P - t[:, None] * u
            dist = np.hypot(v[:, 0], v[:, 1])
            m = dist < half + WALL_CLEAR
            if m.any():
                perp = np.array([-u[1], u[0]])
                side = np.sign(v[m] @ perp)
                side[side == 0] = 1.0
                dirs = np.where(dist[m, None] > 1e-6, v[m] / np.maximum(dist[m, None], 1e-6), side[:, None] * perp)
                P[m] = t[m, None] * u + dirs * (half + WALL_CLEAR)
        return P

    # -------------------------------------------------------------- sleeping
    def _update_sleep(self, P: np.ndarray, det_move: np.ndarray, pressure: np.ndarray):
        """Full step: decide which nodes sleep until the next full step."""
        if not self._allow_sleep or len(P) < SLEEP_MIN_NODES:
            self.asleep[:] = False
            self._anchor = P.copy()
            return
        drift = np.hypot(*(P - self._anchor).T)
        self.asleep = ((drift < SLEEP_DRIFT) & (det_move < WAKE_MOVE)
                       & (pressure > SLEEP_PRESSURE) & ~self._young)
        self._anchor = P.copy()
        self._young[:] = False

    def _build_sleep_tree(self):
        self._n_sleep = int(self.asleep.sum())
        if self._n_sleep:
            self._sleep_pts = self.P[self.asleep].copy()
            self._sleep_ids = self.ids[self.asleep].copy()
            self._sleep_tree = cKDTree(self._sleep_pts)
        else:
            self._sleep_tree = cKDTree(self.P[:1])  # placeholder; _n_sleep == 0 means unused

    @property
    def awake_fraction(self) -> float:
        return 1.0 - float(self.asleep.mean())

    # -------------------------------------------------------------- topology
    def _edge_lengths(self) -> np.ndarray:
        e = np.roll(self.P, -1, axis=0) - self.P
        return np.hypot(e[:, 0], e[:, 1])

    def _curvature(self, P: np.ndarray, act: np.ndarray) -> np.ndarray:
        """Turning angle at each node of `act`, averaged over 7 neighbouring nodes."""
        N = len(P)
        nodes = (act[:, None] + np.arange(-3, 4)) % N
        a = P[nodes] - P[nodes - 1]
        b = P[(nodes + 1) % N] - P[nodes]
        ang = np.abs(np.arctan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0], (a * b).sum(axis=-1)))
        return ang.mean(axis=1)

    def _grow(self, P: np.ndarray, act: np.ndarray, pressure: np.ndarray, full: bool):
        """Split edge i with probability grow_prob * w_edge, where w (0..1) prefers free space and
        curved tips. Done by thinning: first draw edges with probability grow_prob, then keep each
        with probability w_edge. Same odds, but the curvature is only computed for the few drawn edges."""
        N = len(P)
        press = np.full(N, np.inf)  # sleeping / not simulated -> freedom 0 -> never grows
        press[act] = pressure
        if full:
            press[self.asleep] = np.inf
        edges = np.union1d(act, (act - 1) % N)  # edges touching a simulated node
        first = edges[self.rng.random(len(edges)) < self.grow_prob]
        if len(first) == 0:
            return
        ends = np.concatenate([first, (first + 1) % N])
        b = self.p.branchiness
        curv = np.clip(self._curvature(P, ends) / 0.35, 0.0, 1.0)
        w = np.exp(-press[ends] / 0.35) * ((1.0 - b) + b * curv)
        w_edge = 0.5 * (w[:len(first)] + w[len(first):])
        L = np.hypot(*(P[(first + 1) % N] - P[first]).T)
        hit = (self.rng.random(len(first)) < w_edge) & (L > 2.2 * MIN_EDGE)
        pick = np.zeros(N, dtype=bool)
        pick[first[hit]] = True
        self._insert_midpoints(pick)

    def _collapse_short(self):
        if len(self.P) < 10:
            return
        short = self._edge_lengths() < MIN_EDGE
        if not short.any():
            return
        # remove node i+1 for short edge i, never two neighbours in one pass
        rm = np.roll(short, 1)
        rm &= ~np.roll(rm, 1)
        keep = ~rm
        self.P = self.P[keep]
        self.ids = self.ids[keep]
        self.asleep = self.asleep[keep]
        self._anchor = self._anchor[keep]
        self._young = self._young[keep]

    def _split_long(self) -> np.ndarray:
        L = self._edge_lengths()
        long = L > MAX_EDGE
        if long.any():
            self._insert_midpoints(long)
            L = self._edge_lengths()
        return L

    def _insert_midpoints(self, pick: np.ndarray):
        idx = np.nonzero(pick)[0]
        if len(idx) == 0:
            return
        P = self.P
        mids = 0.5 * (P[idx] + P[(idx + 1) % len(P)])
        new_ids = np.arange(self._next_id, self._next_id + len(idx))
        self._next_id += len(idx)
        at = idx + 1
        self.P = np.insert(P, at, mids, axis=0)
        self.ids = np.insert(self.ids, at, new_ids)
        self.asleep = np.insert(self.asleep, at, False)
        self._anchor = np.insert(self._anchor, at, mids, axis=0)
        self._young = np.insert(self._young, at, True)
        # the neighbours of a new node must be free to make room for it
        self.asleep[at - 1 + np.arange(len(at))] = False
        self.asleep[(at + 1 + np.arange(len(at))) % len(self.P)] = False

    # ---------------------------------------------------------------- driver
    @property
    def length(self) -> float:
        return self._length

    @property
    def progress(self) -> float:
        return min(1.0, self._length / self.target_length)

    def run(self, stop_event: threading.Event | None = None, callback=None, callback_every: int = 5,
            relax_steps: int = 60):
        p = self.p
        while self.iteration < p.max_iterations:
            if stop_event is not None and stop_event.is_set():
                self.stop_reason = "stopped"
                break
            self.step(grow=True)
            L = self._length
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
        # relax every node without growth so the spacing evens out
        if not (stop_event is not None and stop_event.is_set()):
            self._allow_sleep = False
            self.asleep[:] = False
            for _ in range(relax_steps):
                self.step(grow=False, full=True)
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
