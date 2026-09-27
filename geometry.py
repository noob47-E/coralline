# Coralline - generative coral lamp patterns drawn as one continuous line
# Copyright (C) 2026 Muhammad Wasiq
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# LICENSE file or <https://www.gnu.org/licenses/> for details.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Geometry helpers for closed polylines (arrays of shape (N, 2), last point != first)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def edge_lengths(P: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)


def closed_length(P: np.ndarray) -> float:
    return float(edge_lengths(P).sum())


def signed_area(P: np.ndarray) -> float:
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def resample_closed(P: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a closed polyline to (roughly) uniform arc-length spacing."""
    seg = edge_lengths(P)
    total = seg.sum()
    n = max(8, int(round(total / spacing)))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    closed = np.vstack([P, P[:1]])
    s = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(s, cum, closed[:, 0])
    y = np.interp(s, cum, closed[:, 1])
    return np.column_stack([x, y])


def chaikin_closed(P: np.ndarray, iterations: int) -> np.ndarray:
    """Chaikin corner cutting; each pass doubles the points and rounds corners."""
    for _ in range(max(0, int(iterations))):
        Q = np.roll(P, -1, axis=0)
        a = 0.75 * P + 0.25 * Q
        b = 0.25 * P + 0.75 * Q
        P = np.empty((2 * len(P), 2))
        P[0::2] = a
        P[1::2] = b
    return P


def laplacian_smooth_closed(P: np.ndarray, iterations: int, strength: float = 0.5) -> np.ndarray:
    for _ in range(max(0, int(iterations))):
        mid = 0.5 * (np.roll(P, 1, axis=0) + np.roll(P, -1, axis=0))
        P = P + strength * (mid - P)
    return P


def catmull_rom_beziers(P: np.ndarray):
    """Closed Catmull-Rom spline through P as cubic Bezier segments.

    Returns (start, c1, c2, end) arrays, each (N, 2); segment i runs P[i] -> P[i+1].
    """
    prev = np.roll(P, 1, axis=0)
    nxt = np.roll(P, -1, axis=0)
    nxt2 = np.roll(P, -2, axis=0)
    c1 = P + (nxt - prev) / 6.0
    c2 = nxt - (nxt2 - P) / 6.0
    return P, c1, c2, nxt


def count_self_intersections(P: np.ndarray) -> int:
    """Number of crossing pairs of non-adjacent segments of a closed polyline."""
    n = len(P)
    if n < 4:
        return 0
    A = P
    B = np.roll(P, -1, axis=0)
    mids = 0.5 * (A + B)
    half = 0.5 * np.linalg.norm(B - A, axis=1)
    pairs = cKDTree(mids).query_pairs(2.0 * half.max() + 1e-9, output_type="ndarray")
    if len(pairs) == 0:
        return 0
    i, j = pairs[:, 0], pairs[:, 1]
    hop = np.abs(i - j)
    keep = np.minimum(hop, n - hop) > 1
    i, j = i[keep], j[keep]

    def orient(p, q, r):
        return (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])

    a, b, c, d = A[i], B[i], A[j], B[j]
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    crossing = (o1 * o2 < 0) & (o3 * o4 < 0)
    return int(crossing.sum())


def prepare_output(P_mm: np.ndarray, smoothing: int, point_spacing_mm: float) -> np.ndarray:
    """Smooth the raw simulation curve (already in mm) for export."""
    P = resample_closed(P_mm, point_spacing_mm)
    if smoothing > 0:
        P = laplacian_smooth_closed(P, smoothing * 2, 0.5)
        P = chaikin_closed(P, 2)
        P = resample_closed(P, point_spacing_mm)
    return P
