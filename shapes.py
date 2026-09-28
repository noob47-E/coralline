# Coralline - generative coral lamp patterns drawn as one continuous line
# Copyright (C) 2026 Muhammad Wasiq
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# LICENSE file or <https://www.gnu.org/licenses/> for details.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Outer shapes and the Region a pattern may fill.

Every shape is "star-shaped" around the origin: a ray from the origin crosses the outline exactly
once, so the outline is a polar function r(theta). That keeps the inside test and the wall clamp
cheap (see Region.clamp). New shapes are validated so that walls are never too slanted for the
radial clamp and there is always a solid middle; unsuitable depth / lumpiness is toned down.

All lengths are in spacing units (1.0 = one line spacing).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from geometry import resample_closed

WALL_CLEAR = 0.35  # hard minimum distance from any wall

# The first six are the original shapes; their outlines must stay exactly as they were.
LEGACY_SHAPES = ["circle", "ellipse", "triangle", "square", "hexagon", "octagon"]
SHAPES = ["circle", "ellipse", "triangle", "square", "rounded square", "pentagon", "hexagon",
          "octagon", "star", "flower", "heart", "teardrop", "blob"]
_SIDES = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6, "octagon": 8}
# which shape sliders each shape uses (the GUI shows only these)
SHAPE_SLIDERS = {"star": ("shape_points", "shape_depth"), "flower": ("shape_points", "shape_depth"),
                 "blob": ("shape_lumpiness",)}

_N_THETA = 4096
_MIN_COS = 0.25        # walls may be at most ~75 degrees off the radial direction
_MIN_INNER = 0.2       # a disk of 0.2 x the max radius around the origin is always inside


# ------------------------------------------------------------------ unit outlines r(theta)
def _polar_of_curve(xy: np.ndarray, harmonics: int, t: np.ndarray) -> np.ndarray:
    """Polar radius r(t) of a closed Cartesian curve around its innermost point, low-pass
    filtered to `harmonics` Fourier terms (rounds cusps like the heart's notch)."""
    # the point furthest from the outline becomes the origin (distance transform on a raster)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    n = 512
    scale = (n - 20) / (hi - lo).max()
    pix = (xy - lo) * scale + 10
    from PIL import Image, ImageDraw
    img = Image.new("L", (n, n), 0)
    ImageDraw.Draw(img).polygon([tuple(p) for p in pix], fill=1)
    dist = ndimage.distance_transform_edt(np.array(img))
    iy, ix = np.unravel_index(np.argmax(dist), dist.shape)
    centre = (np.array([ix, iy], dtype=float) - 10) / scale + lo
    rel = xy - centre
    ang = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2 * np.pi)
    rad = np.hypot(rel[:, 0], rel[:, 1])
    order = np.argsort(ang)
    r = np.interp(t, ang[order], rad[order], period=2 * np.pi)
    spec = np.fft.rfft(r)
    spec[harmonics + 1:] = 0
    return np.fft.irfft(spec, n=len(t))


def _unit_radius(shape: str, t: np.ndarray, points: int, depth: float, lumpiness: float,
                 shape_seed: int) -> np.ndarray:
    """r(t) of a new (non-legacy) shape, before normalising to max radius 1."""
    if shape in _SIDES:  # pentagon
        n = _SIDES[shape]
        return math.cos(math.pi / n) / np.cos(np.mod(t, 2 * np.pi / n) - math.pi / n)
    if shape == "rounded square":
        e = 4.0
        return (np.abs(np.cos(t)) ** e + np.abs(np.sin(t)) ** e) ** (-1.0 / e)
    if shape == "star":
        k = max(3, int(points))
        inner = 1.0 - 0.65 * depth
        seg = 2 * np.pi / k
        # straight edges from outer tip (angle 0 of each segment) to inner corner (half segment)
        u = np.mod(t - np.pi / 2, seg)  # a tip points straight up
        a = np.minimum(u, seg - u)      # 0 at a tip, seg/2 at an inner corner
        tip = np.array([1.0, 0.0])
        corner = inner * np.array([math.cos(seg / 2), math.sin(seg / 2)])
        d = corner - tip
        # intersect the ray at angle a with the line tip + s*d
        cos_a, sin_a = np.cos(a), np.sin(a)
        s = (sin_a * tip[0] - cos_a * tip[1]) / (cos_a * d[1] - sin_a * d[0])
        return np.hypot(tip[0] + s * d[0], tip[1] + s * d[1])
    if shape == "flower":
        k = max(3, int(points))
        a = 0.45 * depth
        return 1.0 - a * (0.5 - 0.5 * np.cos(k * (t - np.pi / 2)))
    if shape == "heart":
        s = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        xy = np.column_stack([16 * np.sin(s) ** 3,
                              13 * np.cos(s) - 5 * np.cos(2 * s) - 2 * np.cos(3 * s) - np.cos(4 * s)])
        return _polar_of_curve(xy, 14, t)
    if shape == "teardrop":
        s = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        xy = np.column_stack([np.sin(s) * np.abs(np.sin(s / 2)) ** 1.4, np.cos(s)])  # tip points up
        return _polar_of_curve(xy, 14, t)
    if shape == "blob":
        rng = np.random.default_rng(shape_seed)
        r = np.ones_like(t)
        for k in range(2, 7):
            r += 0.55 * lumpiness * rng.normal() / k * np.cos(k * t + rng.uniform(0, 2 * np.pi))
        return r
    raise ValueError(f"unknown shape '{shape}'")


def _check_outline(pts: np.ndarray) -> str | None:
    """None if the outline suits the engine, else a short reason."""
    rr = np.hypot(pts[:, 0], pts[:, 1])
    ang = np.unwrap(np.arctan2(pts[:, 1], pts[:, 0]))
    step = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi]))
    if not (np.all(step > 0) or np.all(step < 0)):
        return "not star-shaped"
    tan = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    normal = np.column_stack([tan[:, 1], -tan[:, 0]]) / np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-12)[:, None]
    cos = np.abs((normal * pts).sum(axis=1)) / np.maximum(rr, 1e-12)
    if cos.min() < _MIN_COS:
        return "walls too slanted"
    if rr.min() < _MIN_INNER * rr.max():
        return "middle too thin"
    return None


def outline(params, Rb: float) -> np.ndarray:
    """Dense closed outline (spacing 0.1) of the outer shape, max radius Rb, aspect and rotation applied."""
    p = params
    rot = math.radians(p.rotation_deg)
    t = np.linspace(0, 2 * np.pi, _N_THETA, endpoint=False)
    c, s = math.cos(rot), math.sin(rot)

    def finish(r):
        pts = np.column_stack([r * np.cos(t), r * np.sin(t)])
        aspect = p.aspect if p.boundary != "circle" else 1.0
        pts[:, 1] *= aspect
        pts = pts @ np.array([[c, s], [-s, c]])
        return resample_closed(pts, 0.1)

    if p.boundary in LEGACY_SHAPES:  # exactly the original construction
        if p.boundary in _SIDES:
            n = _SIDES[p.boundary]
            apothem = Rb * math.cos(math.pi / n)
            r = apothem / np.cos(np.mod(t, 2 * np.pi / n) - math.pi / n)
        else:
            r = np.full_like(t, Rb)
        return finish(r)

    depth = float(np.clip(p.shape_depth, 0.0, 1.0))
    lump = float(np.clip(p.shape_lumpiness, 0.0, 1.0))
    last = None
    for _ in range(12):  # tone down depth / lumpiness until the outline suits the engine
        r = _unit_radius(p.boundary, t, p.shape_points, depth, lump, p.shape_seed)
        r = Rb * r / r.max()
        pts = finish(r)
        last = _check_outline(pts)
        if last is None:
            return pts
        depth *= 0.8
        lump *= 0.8
    raise ValueError(f"The {p.boundary} shape can't be used with these settings ({last}). "
                     f"Try less stretching (Height / width closer to 1) or a smaller depth.")


# ------------------------------------------------------------------------------- Region
class Region:
    """Where a pattern may go: inside the outer shape, outside the centre hole and channels."""

    def __init__(self, params, Rb: float):
        self.p = params
        self.Rb = Rb
        pts = outline(params, Rb)
        th = np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2 * np.pi)
        rr = np.hypot(pts[:, 0], pts[:, 1])
        order = np.argsort(th)
        self.b_theta, self.b_r = th[order], rr[order]
        self.b_pts = pts
        self.b_tree = cKDTree(pts)
        self.rb_min = float(rr.min())
        # cos of the worst angle between the radial direction and the wall normal; lets
        # wall_forces skip nodes that are provably out of reach of the boundary
        tan = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
        normal = np.column_stack([tan[:, 1], -tan[:, 0]]) / np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-12)[:, None]
        cos = np.abs((normal * pts).sum(axis=1)) / np.maximum(rr, 1e-12)
        self.b_cos = cos[order]  # per boundary sample, sorted by angle
        self.b_cos_min = max(0.1, float(cos.min()) - 1.5 / self.rb_min)
        self.legacy = params.boundary in LEGACY_SHAPES
        self.hole_r = params.hole * self.rb_min if params.hole > 0 else 0.0
        self.channels: list = []

    # --- lookups
    def polar_r(self, theta: np.ndarray) -> np.ndarray:
        return np.interp(np.mod(theta, 2 * np.pi), self.b_theta, self.b_r, period=2 * np.pi)

    def wall_cos(self, theta: np.ndarray) -> np.ndarray:
        """cos of the angle between the radial direction and the wall at angle theta."""
        return np.interp(np.mod(theta, 2 * np.pi), self.b_theta, self.b_cos, period=2 * np.pi)

    def inset_r(self, theta: np.ndarray, inset: float) -> np.ndarray:
        """Radius on each ray that lies about `inset` inside the outer wall (measured square to
        the wall for the new shapes; the original shapes keep the plain radial inset)."""
        if self.legacy:
            return self.polar_r(theta) - inset
        return self.polar_r(theta) - inset / np.maximum(self.wall_cos(theta), _MIN_COS)

    # --- channels
    def build_channels(self, span):
        """span(theta, rb) -> (r_in, r_out) of the channel at angle theta."""
        p = self.p
        self.channels = []
        if p.channels <= 0:
            return
        half = 0.5 * p.channel_width
        rot = math.radians(p.channel_rotation_deg)
        for k in range(int(p.channels)):
            th = rot + 2 * math.pi * k / p.channels
            rb = float(self.polar_r(np.array([th]))[0])
            r_in, r_out = span(th, rb, half)
            if r_out > r_in + 0.5:
                u = np.array([math.cos(th), math.sin(th)])
                self.channels.append((u, r_in, r_out, half))

    def default_channel_span(self, th, rb, half):
        """For the non-coral styles: channels start ~5 spacings out, so a ring of grid cells
        (2 spacings wide, kept 1.2 from every wall) always connects the parts between channels."""
        r_in = self.hole_r + 5.5 + half
        r_out = self.p.channel_length * rb + (half + 1.0 if self.p.channel_length >= 0.999 else 0.0)
        return r_in, r_out

    # --- tests
    def allowed_mask(self, pts: np.ndarray) -> np.ndarray:
        th = np.arctan2(pts[:, 1], pts[:, 0])
        r = np.hypot(pts[:, 0], pts[:, 1])
        ok = r < self.polar_r(th)
        if self.hole_r > 0:
            ok &= r > self.hole_r
        for u, r_in, r_out, half in self.channels:
            t = np.clip(pts @ u, r_in, r_out)
            ok &= np.hypot(pts[:, 0] - t * u[0], pts[:, 1] - t * u[1]) > half
        return ok

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        """Distance to the nearest wall (outer shape, hole, channels); negative outside the region."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        d, _ = self.b_tree.query(pts)
        r = np.hypot(pts[:, 0], pts[:, 1])
        inside = r < self.polar_r(np.arctan2(pts[:, 1], pts[:, 0]))
        out = np.where(inside, d, -d)
        if self.hole_r > 0:
            out = np.minimum(out, r - self.hole_r)
        for u, r_in, r_out, half in self.channels:
            t = np.clip(pts @ u, r_in, r_out)
            out = np.minimum(out, np.hypot(pts[:, 0] - t * u[0], pts[:, 1] - t * u[1]) - half)
        return out

    def segment_ok(self, a: np.ndarray, b: np.ndarray, clear: float = WALL_CLEAR) -> np.ndarray:
        """For arrays of segments a[i]->b[i]: True where the whole segment keeps `clear` from every wall."""
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        L = np.hypot(*(b - a).T)
        n = int(max(2, np.ceil(L.max() / 0.2) + 1)) if len(L) else 2
        s = np.linspace(0, 1, n)
        samples = a[:, None, :] + s[None, :, None] * (b - a)[:, None, :]
        return (self.sdf(samples.reshape(-1, 2)).reshape(len(a), n) >= clear).all(axis=1)

    def area(self, h: float = 0.5) -> float:
        R = float(self.b_r.max())
        g = np.arange(-R, R + h, h)
        X, Y = np.meshgrid(g, g)
        return float(self.allowed_mask(np.column_stack([X.ravel(), Y.ravel()])).sum() * h * h)

    # --- used by the physics engine (moved here unchanged from growth.py)
    def wall_forces(self, P: np.ndarray, F: np.ndarray, k_wall: float) -> np.ndarray:
        pressure = np.zeros(len(P))
        r = np.maximum(np.hypot(P[:, 0], P[:, 1]), 1e-9)
        # outer boundary: only nodes that can be within reach of it
        cand = np.flatnonzero(r > self.rb_min - 1.05 / self.b_cos_min)
        if len(cand):
            gap = self.polar_r(np.arctan2(P[cand, 1], P[cand, 0])) - r[cand]
            cand = cand[gap * self.b_cos_min < 1.05]
        if len(cand):
            d, idx = self.b_tree.query(P[cand], distance_upper_bound=1.0)
            near = np.isfinite(d)
            if near.any():
                ci = cand[near]
                v = P[ci] - self.b_pts[idx[near]]
                dn = np.maximum(d[near], 1e-6)
                mag = 1.0 - dn
                F[ci] += (k_wall * mag / dn)[:, None] * v
                pressure[ci] += mag
        # centre hole
        if self.hole_r > 0:
            dw = r - self.hole_r
            m = dw < 1.0
            if m.any():
                mag = 1.0 - np.maximum(dw[m], 0.0)
                F[m] += (k_wall * mag / r[m])[:, None] * P[m]
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
                F[m] += (k_wall * mag / dist[m])[:, None] * v[m]
                pressure[m] += mag
        return pressure

    def clamp(self, P: np.ndarray) -> np.ndarray:
        r = np.maximum(np.hypot(P[:, 0], P[:, 1]), 1e-9)
        rmax = np.full(len(P), np.inf)
        outer = np.flatnonzero(r > self.rb_min - WALL_CLEAR)  # the only ones that can be outside
        if len(outer):
            rmax[outer] = self.polar_r(np.arctan2(P[outer, 1], P[outer, 0])) - WALL_CLEAR
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
