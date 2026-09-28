# Coralline - generative coral lamp patterns drawn as one continuous line
# Copyright (C) 2026 Muhammad Wasiq
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# LICENSE file or <https://www.gnu.org/licenses/> for details.
# SPDX-License-Identifier: GPL-3.0-or-later

"""One-line scribble style ("TSP art"): one loop that wanders through thousands of dots.

1. Dots: evenly spread random points (Poisson-disk style), their spacing set by scribble_gap and
   varied by scribble_density (uniform, denser in the middle, or in soft clouds).
2. A first tour: the dots are visited in the order of a snake-like maze path (gridtree), so the
   tour never jumps across the shape, holes or channels.
3. Untangle: while any two tour edges cross, apply a 2-opt move (reverse the stretch between
   them). Uncrossing always makes the tour shorter, so this always ends with zero crossings.
   Moves whose new edges would cut through a wall are skipped (the dot is dropped instead).
4. Relax with the coral physics so no two parts of the line come too close (raw tours can almost
   touch), optionally growing a little (organic).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from generators import PatternGenerator
from geometry import resample_closed
from gridtree import Lattice, tree_to_loop

MAX_DOTS = 40000


def _crossing_pairs(P: np.ndarray) -> np.ndarray:
    """(i, j) with i < j for every pair of non-adjacent tour edges that cross."""
    n = len(P)
    A = P
    B = np.roll(P, -1, axis=0)
    mids = 0.5 * (A + B)
    half = 0.5 * np.hypot(*(B - A).T)
    pairs = cKDTree(mids).query_pairs(2.0 * half.max() + 1e-9, output_type="ndarray")
    if len(pairs) == 0:
        return pairs
    i, j = pairs[:, 0], pairs[:, 1]
    hop = np.abs(i - j)
    keep = np.minimum(hop, n - hop) > 1
    i, j = i[keep], j[keep]

    def orient(p, q, r):
        return (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])

    a, b, c, d = A[i], B[i], A[j], B[j]
    x = (orient(a, b, c) * orient(a, b, d) < 0) & (orient(c, d, a) * orient(c, d, b) < 0)
    out = np.column_stack([np.minimum(i[x], j[x]), np.maximum(i[x], j[x])])
    return out[np.argsort(out[:, 0])]


class ScribbleGenerator(PatternGenerator):
    style = "scribble"

    def _dots(self) -> np.ndarray:
        p = self.p
        reg = self.region
        rng = self.rng
        R = float(reg.b_r.max())
        gap = float(np.clip(p.scribble_gap, 1.3, 3.0))
        area = reg.area()
        while area / (0.9 * gap * gap) > MAX_DOTS:
            gap *= 1.1

        def factor(xy):
            r = np.hypot(xy[:, 0], xy[:, 1])
            if p.scribble_density == "radial":
                return 0.6 + 0.8 * np.clip(r / R, 0, 1)
            if p.scribble_density == "clouds":
                f = np.zeros(len(xy))
                g = np.random.default_rng(p.seed + 7)
                for _ in range(5):
                    k = g.normal(size=2) * 2.5 / R
                    f += np.cos(xy @ k + g.uniform(0, 2 * np.pi))
                return 1.0 + 0.4 * f / 5 * 2
            return np.ones(len(xy))

        # dart throwing in chunks: random candidates, keep those not too close to kept dots
        want = int(12 * area / (gap * gap)) + 200
        cand = rng.uniform(-R, R, size=(want, 2))
        cand = cand[reg.sdf(cand) >= 1.0]
        rad = gap * factor(cand)
        kept = np.zeros((0, 2))
        kept_r = np.zeros(0)
        for s in range(0, len(cand), 3000):
            c, cr = cand[s:s + 3000], rad[s:s + 3000]
            if len(kept):
                d, k = cKDTree(kept).query(c)
                ok = d >= np.maximum(cr, kept_r[k])
                c, cr = c[ok], cr[ok]
            # resolve conflicts inside the chunk greedily
            alive = np.ones(len(c), dtype=bool)
            if len(c) > 1:
                pr = cKDTree(c).query_pairs(float(cr.max()), output_type="ndarray")
                if len(pr):
                    dd = np.hypot(*(c[pr[:, 0]] - c[pr[:, 1]]).T)
                    pr = pr[dd < np.maximum(cr[pr[:, 0]], cr[pr[:, 1]])]
                    for a, b in pr:
                        if alive[a] and alive[b]:
                            alive[b] = False
            kept = np.vstack([kept, c[alive]])
            kept_r = np.concatenate([kept_r, cr[alive]])
            self.report(None, 0.1 * min(1.0, (s + 3000) / len(cand)))
        return kept

    def build(self):
        p = self.p
        reg = self.region
        dots = self._dots()
        if len(dots) < 10:
            raise ValueError("The shape is too small for a scribble at this line spacing.")

        # first tour: order the dots along a snake-like maze path through the whole region
        lat = Lattice(reg, p.rotation_deg)
        rng = self.rng
        start = int(lat.cells[0])
        seen = np.zeros(lat.n * lat.n, dtype=bool)
        seen[start] = True
        stack, edges = [start], []
        while stack:  # randomised depth-first search: long snaking corridors
            c = stack[-1]
            nb = [d for d in lat.nbrs[c] if not seen[d]]
            if not nb:
                stack.pop()
                continue
            d = nb[int(rng.integers(len(nb)))]
            seen[d] = True
            edges.append((c, d))
            stack.append(d)
        path = tree_to_loop(lat, np.array(edges))
        _, pos = cKDTree(path).query(dots)
        order = np.lexsort((rng.random(len(dots)), pos))
        tour = dots[order]
        tour_pos = pos[order]
        # where a straight hop would cut a wall, walk along the maze path instead
        nxt = np.roll(tour, -1, axis=0)
        bad = ~reg.segment_ok(tour, nxt, clear=0.5)
        if bad.any():
            pieces = []
            m = len(path)
            for i in range(len(tour)):
                pieces.append(tour[i:i + 1])
                if bad[i]:
                    a, b = tour_pos[i], tour_pos[(i + 1) % len(tour)]
                    steps = (b - a) % m
                    if steps > 1:
                        pieces.append(path[(a + 1 + np.arange(steps - 1)) % m])
            tour = np.vstack(pieces)
        self.report(tour, 0.15)

        # untangle: 2-opt moves until no two edges cross
        passes = 0
        while True:
            pairs = _crossing_pairs(tour)
            if len(pairs) == 0:
                break
            passes += 1
            n = len(tour)
            busy = np.zeros(n + 1, dtype=bool)
            drop = []
            for i, j in pairs:
                if busy[i:j + 2].any():
                    continue
                busy[i:j + 2] = True
                a, b = tour[i], tour[(i + 1) % n]
                c, d = tour[j], tour[(j + 1) % n]
                if reg.segment_ok(np.array([a, b]), np.array([c, d]), clear=0.35).all():
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1].copy()
                else:
                    drop.append((i + 1) % n)
            if drop:
                tour = np.delete(tour, drop, axis=0)
            if passes % 5 == 0:
                self.report(tour, 0.15 + 0.15 * min(1.0, passes / 50))
            if len(tour) < 10:
                raise ValueError("The scribble could not be untangled in this shape.")

        # relax so no two parts of the line nearly touch; optionally grow a little
        line = resample_closed(tour, 0.42)
        self.report(line, 0.3)
        P = self.organic(line, p.organic, 0.3, 1.0, min_steps=300)
        self.finish(P, note=f"{len(dots)} dots")
