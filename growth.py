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
from shapes import SHAPES, WALL_CLEAR, Region

BOUNDARIES = SHAPES
START_SHAPES = ["ring", "circle", "star"]

# Simulation constants (spacing units)
REST = 0.42        # rest length of an edge
MAX_EDGE = 0.7     # longer edges are split
MIN_EDGE = 0.14    # shorter edges are collapsed
MAX_STEP = 0.12    # max movement of a node per step
LINE_GAP = 0.92    # typical distance between neighbouring strands when packed

# Sleeping (speed-up for big patterns, see DifferentialGrowth.step)
SLEEP_MIN_NODES = 3000  # smaller patterns always simulate every node
SLEEP_EVERY = 10        # every Nth step is a full step that re-checks every node
SLEEP_DRIFT = 0.08      # a node that moved less than this since the last full step may sleep
WAKE_MOVE = 0.02        # a node pushed harder than this (per step) wakes / stays awake
SLEEP_PRESSURE = 0.25   # only crowded nodes sleep (free space means it may still grow)
SLEEP_K = 16            # max sleeping neighbours looked up per awake node


STYLES = ["coral", "maze", "dendrite", "spiral", "scribble"]


@dataclass
class GrowthParams:
    """All pattern settings (the name is historical: presets store them under "growth")."""
    style: str = "coral"
    # size
    diameter_mm: float = 200.0
    spacing_mm: float = 4.5
    boundary: str = "circle"
    aspect: float = 1.0
    rotation_deg: float = 0.0
    shape_points: int = 5         # star / flower
    shape_depth: float = 0.5      # star / flower
    shape_lumpiness: float = 0.5  # blob
    shape_seed: int = 1           # blob outline
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
    # maze / dendrite / scribble: let the coral physics soften the finished curve
    organic: float = 0.5
    # maze
    maze_corridor: float = 0.7       # 0 = short twisty corridors, 1 = long winding ones
    maze_direction: str = "any"      # any | radial | circular
    # dendrite
    dendrite_branching: float = 0.5  # 0 = blobby, 1 = thin lightning branches
    dendrite_roots: str = "centre"   # centre | edge
    # spiral
    spiral_mode: str = "twist"       # twist (a spiral) | rings (looks concentric)
    spiral_roundness: float = 0.5    # inner rings: follow the shape (0) or turn round (1)
    ring_wobble: float = 0.2
    # scribble
    scribble_density: str = "uniform"  # uniform | radial | clouds
    scribble_gap: float = 1.8          # average gap between dots, in line spacings
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
    """The coral style. Also used by other styles to soften ("organic finish") a finished curve:
    pass `initial_points` (spacing units), the style's `region` and `rng`, and `extra_length`
    (0.1 = the line may grow 10% longer)."""

    def __init__(self, params: GrowthParams, initial_points: np.ndarray | None = None,
                 extra_length: float = 0.0, region: Region | None = None, rng=None,
                 growable: np.ndarray | None = None):
        self.p = params
        self.rng = rng if rng is not None else np.random.default_rng(params.seed)
        self.iteration = 0
        self.done = False
        self.stop_reason = ""
        self.Rb = max(4.0, 0.5 * params.diameter_mm / params.spacing_mm)

        if initial_points is None:
            self.region = Region(params, self.Rb)
            self.hole_r = self.region.hole_r
            self._build_seed()
            self.region.build_channels(self._channel_span)
            self.P = self._seed_pts
            self._estimate_target()
        else:
            self.region = region if region is not None else Region(params, self.Rb)
            self.hole_r = self.region.hole_r
            if growable is None:
                self.P = resample_closed(np.asarray(initial_points, dtype=float), REST)
            else:  # caller already spaced the points; growable[i] says if node i may grow
                self.P = np.asarray(initial_points, dtype=float).copy()
            self.full_length = self.region.area() / LINE_GAP
            self.target_length = closed_length(self.P) * (1.0 + max(0.0, extra_length))
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
        self.growable = None if growable is None else np.asarray(growable, dtype=bool).copy()

        s = params.smoothness
        self.k_smooth = 0.04 + 0.36 * s
        self.k_spring = 0.25
        self.k_rep = 0.35
        self.k_wall = 0.5
        self.dt = 0.5  # time step; 1.0 made nodes overshoot and bounce every step
        self.noise = 0.004 + 0.03 * params.wiggle
        self.grow_prob = 0.002 + 0.028 * params.growth_speed ** 1.5

    # ------------------------------------------------------------------ setup
    def _random_wobble(self, theta: np.ndarray, amp: float) -> np.ndarray:
        w = np.zeros_like(theta)
        for k in range(2, 8):
            w += self.rng.normal() / k * np.sin(k * theta + self.rng.uniform(0, 2 * np.pi))
        return amp * w / max(1e-9, np.abs(w).max())

    def _seed_radius(self, theta: np.ndarray) -> np.ndarray:
        p = self.p
        base = max(1.6, self.hole_r + 1.6)
        if p.start_shape == "ring":
            return self.region.inset_r(theta, 1.3)
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

    def _channel_span(self, th, rb, half):
        """Coral channels start just outside the start shape (or end just inside the ring)."""
        rs = float(self._seed_radius(np.array([th]))[0])
        if self.p.start_shape == "ring":
            return self.hole_r, min(self.p.channel_length * rb, rs - half - 1.4)
        r_in = rs + 0.3 + half + 1.4
        r_out = self.p.channel_length * rb + (half + 1.0 if self.p.channel_length >= 0.999 else 0.0)
        return r_in, r_out

    def _estimate_target(self):
        h = 0.5
        area = self.region.area(h)
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
        P[act] = self.region.clamp(P[act] + disp)

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
        pressure += self.region.wall_forces(p, F, self.k_wall)
        return F, pressure

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
        if self.growable is not None:
            w = w * self.growable[ends]
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
        if self.growable is not None:
            self.growable = self.growable[keep]

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
        if self.growable is not None:
            g = self.growable
            self.growable = np.insert(g, at, g[idx] & g[(idx + 1) % len(g)])
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
        return self.region.b_pts * self.p.spacing_mm

    def self_intersections(self) -> int:
        return count_self_intersections(self.P)
