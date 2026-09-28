# Coralline - generative coral lamp patterns drawn as one continuous line
# Copyright (C) 2026 Muhammad Wasiq
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# LICENSE file or <https://www.gnu.org/licenses/> for details.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Grid trees -> one closed line (used by the maze, dendrite and scribble styles).

The region is covered by a lattice of square "cells" 2 spacings wide. A tree over those cells (any
tree: a maze, a branching dendrite) is turned into ONE closed line that runs all the way around it:

  * every cell owns 4 "fine" points (a 2x2 block, 1 spacing apart) joined in a small square loop;
  * for every tree edge between two cells, the two facing sides of their squares are removed and
    two bridges are added, which merges the two loops into one.

Because the cells form a single tree (connected, no cycles), all the little loops merge into exactly
one closed line through all 4 x cells points, strands exactly 1 spacing apart, never crossing.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

CELL = 2.0            # cell size in spacing units (fine points are 1 apart)
CELL_CLEAR = 1.21     # distance a cell centre / tree edge midpoint must keep from walls


class Lattice:
    """Cells of the grid that lie inside the region, and which neighbours may be joined."""

    def __init__(self, region, rotation_deg: float = 0.0):
        R = float(region.b_r.max())
        self.n = n = int(math.ceil(2 * R / CELL)) + 2
        self.origin = -0.5 * n * CELL
        rot = math.radians(rotation_deg)
        self.rot = np.array([[math.cos(rot), math.sin(rot)], [-math.sin(rot), math.cos(rot)]])
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        self.centres = self.to_world(np.column_stack([ii.ravel() * 2 + 1.0, jj.ravel() * 2 + 1.0]))
        valid = region.sdf(self.centres) >= CELL_CLEAR
        # joinable neighbours: east (i+1, j) and north (i, j+1)
        idx = np.arange(n * n).reshape(n, n)
        east = np.column_stack([idx[:-1, :].ravel(), idx[1:, :].ravel()])
        north = np.column_stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()])
        edges = np.vstack([east, north])
        edges = edges[valid[edges[:, 0]] & valid[edges[:, 1]]]
        mids = 0.5 * (self.centres[edges[:, 0]] + self.centres[edges[:, 1]])
        edges = edges[region.sdf(mids) >= CELL_CLEAR]
        # keep only the largest connected group of cells
        m = n * n
        g = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(m, m))
        _, labels = connected_components(g, directed=False)
        sizes = np.bincount(labels[valid], minlength=labels.max() + 1)
        best = int(np.argmax(sizes)) if sizes.size else 0
        self.valid = valid & (labels == best)
        self.dropped = int(valid.sum() - self.valid.sum())
        self.edges = edges[self.valid[edges[:, 0]]]
        # neighbour lists (for tree-growing algorithms)
        self.nbrs: list[list[int]] = [[] for _ in range(m)]
        for a, b in self.edges:
            self.nbrs[a].append(int(b))
            self.nbrs[b].append(int(a))
        self.cells = np.flatnonzero(self.valid)

    def to_world(self, lattice_xy: np.ndarray) -> np.ndarray:
        return (lattice_xy + self.origin) @ self.rot

    def cell_ij(self, c):
        return np.divmod(c, self.n)


def tree_to_loop(lat: Lattice, tree_edges: np.ndarray) -> np.ndarray:
    """The closed line around a tree of cells, as points in spacing units (fine points 1 apart).

    tree_edges: (k, 2) array of neighbouring cell indices forming ONE tree (k = cells - 1)."""
    tree_edges = np.asarray(tree_edges, dtype=np.int64).reshape(-1, 2)
    if len(tree_edges) == 0:
        raise ValueError("the tree needs at least one edge")
    cells = np.unique(tree_edges)
    n = lat.n
    fn = 2 * n + 1
    H = np.zeros((fn, fn), dtype=bool)  # H[x, y]: fine (x, y) -- (x+1, y)
    V = np.zeros((fn, fn), dtype=bool)  # V[x, y]: fine (x, y) -- (x, y+1)
    ci, cj = lat.cell_ij(cells)
    x, y = 2 * ci, 2 * cj
    H[x, y] = H[x, y + 1] = True       # every cell: its own little square
    V[x, y] = V[x + 1, y] = True
    a = np.minimum(tree_edges[:, 0], tree_edges[:, 1])
    b = np.maximum(tree_edges[:, 0], tree_edges[:, 1])
    ai, aj = lat.cell_ij(a)
    bi, bj = lat.cell_ij(b)
    east = bi == ai + 1                 # cell index = i * n + j, so b - a == n means east
    x, y = 2 * ai[east], 2 * aj[east]   # east edge (i,j)-(i+1,j)
    V[x + 1, y] = V[x + 2, y] = False
    H[x + 1, y] = H[x + 1, y + 1] = True
    x, y = 2 * ai[~east], 2 * aj[~east]  # north edge (i,j)-(i,j+1)
    H[x, y + 1] = H[x, y + 2] = False
    V[x, y + 1] = V[x + 1, y + 1] = True

    hx, hy = np.nonzero(H)
    vx, vy = np.nonzero(V)
    u = np.concatenate([hx * fn + hy, vx * fn + vy])
    v = np.concatenate([(hx + 1) * fn + hy, vx * fn + vy + 1])
    ends = np.concatenate([u, v])
    other = np.concatenate([v, u])
    order = np.argsort(ends, kind="stable")
    ends, other = ends[order], other[order]
    nodes, first, count = np.unique(ends, return_index=True, return_counts=True)
    if not np.all(count == 2):
        raise AssertionError("tree_to_loop: the cells are not a single tree")
    nb0, nb1 = other[first], other[first + 1]
    pos = {int(k): i for i, k in enumerate(nodes)}
    # walk the loop
    m = len(nodes)
    walk = np.empty(m, dtype=np.int64)
    prev, cur = -1, int(nodes[0])
    for k in range(m):
        walk[k] = cur
        i = pos[cur]
        nxt = int(nb0[i]) if int(nb0[i]) != prev else int(nb1[i])
        prev, cur = cur, nxt
    if cur != walk[0] or m != 4 * len(cells):
        raise AssertionError("tree_to_loop: the line did not close through every point")
    fx, fy = np.divmod(walk, fn)
    return lat.to_world(np.column_stack([fx + 0.5, fy + 0.5]).astype(float))


def bfs_tree(lat: Lattice, start_cells, allowed: np.ndarray | None = None) -> np.ndarray:
    """A breadth-first spanning tree over the cells reachable from start_cells (within `allowed`)."""
    from collections import deque
    seen = np.zeros(lat.n * lat.n, dtype=bool)
    ok = lat.valid if allowed is None else (lat.valid & allowed)
    q = deque()
    edges = []
    first = True
    for c in start_cells:
        c = int(c)
        if ok[c] and not seen[c]:
            if not first:
                continue  # only one root: every other start cell must be reached through the tree
            seen[c] = True
            q.append(c)
            first = False
    while q:
        c = q.popleft()
        for d in lat.nbrs[c]:
            if ok[d] and not seen[d]:
                seen[d] = True
                edges.append((c, d))
                q.append(d)
    return np.array(edges, dtype=np.int64).reshape(-1, 2)
