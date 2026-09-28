# Coralline - generative coral lamp patterns drawn as one continuous line
# Copyright (C) 2026 Muhammad Wasiq
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# LICENSE file or <https://www.gnu.org/licenses/> for details.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spiral rings style: a closed double spiral (like a Fermat spiral) that follows the outer shape.

1. Nested rings, as a radius on each of M rays from the centre: ring 0 sits 0.5 inside the wall,
   every next ring at least 1 spacing inside the previous one (measured as a true distance).
   Inner rings are smoothed toward circles (spiral_roundness) and may wobble (ring_wobble).
2. A "level" c (0 = ring 0, 1 = ring 1, ... with linear blending between rings) is a radius that
   strictly shrinks as c grows, on every ray.
3. Two lanes: A(phi) goes inward as phi winds round (one level per half turn in "twist" mode, or
   staying on a ring and hopping over in a short window in "rings" mode); B = A + 1 goes back out.
   On every ray all passes of A and B sit on different levels, i.e. different radii, so the line
   cannot cross itself. A short radial step joins A to B at the centre and at the start.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from generators import PatternGenerator
from geometry import closed_length, resample_closed
from growth import LINE_GAP


def _smooth_periodic(r: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return r
    f = np.fft.rfftfreq(len(r), d=1.0 / len(r))
    return np.fft.irfft(np.fft.rfft(r) * np.exp(-strength * f ** 2), n=len(r))


class SpiralGenerator(PatternGenerator):
    style = "spiral"
    uses_channels = False  # a loop around the centre can't coexist with channels

    def _dist_to_wall(self, pts):
        return self.region.sdf(pts)

    def build(self):
        p = self.p
        reg = self.region
        rng = self.rng
        R = float(reg.b_r.max())
        M = int(math.ceil(2 * math.pi * R / 0.2))
        th = np.linspace(0, 2 * np.pi, M, endpoint=False)
        u = np.column_stack([np.cos(th), np.sin(th)])
        stop_r = reg.hole_r + 0.75 if reg.hole_r > 0 else 1.25

        # ring 0: 0.5 inside the outer wall (step inward by exactly the missing distance)
        rho = reg.polar_r(th).copy()
        for _ in range(60):
            d = reg.sdf(rho[:, None] * u)
            short = d < 0.5
            if not short.any():
                break
            rho[short] -= (0.5 - d[short]) + 1e-3
        rings = [rho]

        roundness = float(np.clip(p.spiral_roundness, 0.0, 1.0))
        wobble = float(np.clip(p.ring_wobble, 0.0, 1.0))
        while True:
            prev = rings[-1]
            # blend toward a circle (roundness) and iron out small wiggles, then 1 spacing inward
            mean = prev.mean()
            start = mean + (_smooth_periodic(prev, 2e-4) - mean) * (1.0 - 0.3 * roundness) - 1.0
            if wobble > 0:
                w = np.zeros(M)
                for k in range(2, 9):
                    w += rng.normal() / k * np.cos(k * th + rng.uniform(0, 2 * np.pi))
                start -= 1.2 * wobble * np.abs(w) / max(np.abs(w).max(), 1e-9)
            nxt = np.clip(start, prev - 2.0 - 1.2 * wobble, prev)
            dense = resample_closed(prev[:, None] * u, 0.1)
            tree = cKDTree(dense)
            for _ in range(80):
                d, _ = tree.query(nxt[:, None] * u)
                short = d < 1.0
                if not short.any():
                    break
                nxt[short] -= (1.0 - d[short]) + 1e-3
            if nxt.min() < stop_r or len(rings) > 4000:
                break
            rings.append(nxt)
            self.report(None, 0.2 * min(1.0, (R - nxt.mean()) / max(R - stop_r, 1e-9)))
        K = len(rings)
        if K < 3:  # too narrow for a double spiral: one ring along the outline, coral inside
            loop = rings[0][:, None] * u
            wanted = float(np.clip(max(p.fill, 0.6), 0.0, 1.0)) * reg.area() / LINE_GAP
            extra = max(0.0, wanted / closed_length(loop) - 1.0)
            line = resample_closed(loop, 0.42)
            self.report(line, 0.5)
            P = self.organic(line, 0.0, 0.5, 1.0, min_steps=40, extra=extra,
                             growable=np.ones(len(line), dtype=bool))
            self.finish(P, fallback=loop, note="shape too narrow for rings: outline ring with coral inside")
            return
        if p.spiral_mode == "rings" and (K - 2) % 2 == 1:
            rings.pop()  # rings mode needs an even number of hops
            K -= 1
        levels = np.array(rings)  # (K, M)

        def radius(c, t):
            """Radius at level c (0..K-1, fractional) and angle t, interpolated between rays."""
            pos = np.mod(t, 2 * np.pi) / (2 * np.pi) * M
            j0 = np.floor(pos).astype(int) % M
            j1 = (j0 + 1) % M
            fr = pos - np.floor(pos)
            k0 = np.clip(np.floor(c).astype(int), 0, K - 1)
            k1 = np.clip(k0 + 1, 0, K - 1)
            fc = np.clip(c - k0, 0, 1)
            r0 = levels[k0, j0] * (1 - fr) + levels[k0, j1] * fr
            r1 = levels[k1, j0] * (1 - fr) + levels[k1, j1] * fr
            return r0 * (1 - fc) + r1 * fc

        phi_end = (K - 2) * math.pi
        dphi = 0.2 / R
        phi = np.arange(0.0, phi_end + dphi / 2, dphi)
        phi[-1] = phi_end
        if p.spiral_mode == "rings":
            turn = np.floor(phi / (2 * np.pi))
            x = phi - 2 * np.pi * turn
            mean_r = levels.mean(axis=1)
            ring_r = mean_r[np.clip((2 * turn).astype(int), 0, K - 1)]
            window = np.clip(6.0 / np.maximum(ring_r, 1e-9), 0.3, 2.0)
            s = np.clip((x - (2 * np.pi - window)) / window, 0.0, 1.0)
            cA = 2 * turn + 2 * (s * s * (3 - 2 * s))
            cA[-1] = K - 2
        else:
            cA = phi / math.pi
        cB = cA + 1
        A = radius(cA, phi)[:, None] * np.column_stack([np.cos(phi), np.sin(phi)])
        B = radius(cB, phi)[:, None] * np.column_stack([np.cos(phi), np.sin(phi)])
        loop = np.vstack([A, B[::-1]])
        # evenly spaced points, each remembering its level (to know the innermost lanes)
        lev = np.concatenate([cA, cB[::-1]])
        seg = np.hypot(*(np.roll(loop, -1, axis=0) - loop).T)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        s_new = np.linspace(0.0, cum[-1], max(8, int(round(cum[-1] / 0.42))), endpoint=False)
        closed = np.vstack([loop, loop[:1]])
        loop = np.column_stack([np.interp(s_new, cum, closed[:, 0]), np.interp(s_new, cum, closed[:, 1])])
        lev = np.interp(s_new, cum, np.concatenate([lev, lev[:1]]))
        self.report(loop, 0.5)
        # Rings around one centre can't go deeper than the nearest wall, so lobed shapes (heart,
        # blob) keep an empty middle. Fill amount lets the innermost lane grow coral into it.
        wanted = float(np.clip(p.fill, 0.0, 1.0)) * reg.area() / LINE_GAP
        have = closed_length(loop)
        extra = wanted / have - 1.0 if have > 0 else 0.0
        if extra > 0.03:
            inner = lev >= K - 3.0  # only the innermost lanes grow; the rings stay clean
            P = self.organic(loop, 0.0, 0.5, 1.0, min_steps=40, extra=extra, growable=inner)
            note = f"{K} rings, coral-filled middle"
        else:  # a short relax (no growth) evens out the few tight spots between rays
            P = self.organic(loop, 0.0, 0.5, 1.0, min_steps=40)
            note = f"{K} rings"
        self.finish(P, fallback=loop, note=note)
