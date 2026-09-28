# Coralline - generative coral lamp patterns drawn as one continuous line
# Copyright (C) 2026 Muhammad Wasiq
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# LICENSE file or <https://www.gnu.org/licenses/> for details.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Branching dendrite style: lightning / coral branches, drawn as the one line around them.

Growth follows the dielectric breakdown model (a cousin of diffusion-limited aggregation):
a potential phi is solved on the cell grid (Laplace's equation, phi = 0 on the branches,
phi = 1 at the "source": the outer wall when growing from the centre, the middle when growing
from the edge). Empty cells next to a branch are added with probability ~ phi^eta. Tips that stick
out see the largest phi, so they grow fastest and split: eta (from dendrite_branching) sets how
thin and lightning-like the branches get. Every new cell hangs on exactly ONE existing cell, so
the branches always form a single tree -> one closed line via gridtree.tree_to_loop.
"""
from __future__ import annotations

import numpy as np

from generators import PatternGenerator
from geometry import chaikin_closed
from gridtree import Lattice, bfs_tree, tree_to_loop

SWEEPS = 24  # relaxation sweeps of the potential between growth batches


def _trunk(lat: Lattice, band: np.ndarray, root: int) -> np.ndarray:
    """One tree along the `band` cells. The band may be in several pieces (a narrow neck, a notch):
    each extra piece is linked to the tree by the single shortest path through the inside."""
    from collections import deque
    if root < 0:
        return np.empty((0, 2), dtype=np.int64)
    m = lat.n * lat.n
    in_tree = np.zeros(m, dtype=bool)
    edges = [tuple(e) for e in bfs_tree(lat, [root], allowed=band)]
    in_tree[root] = True
    for a, b in edges:
        in_tree[a] = in_tree[b] = True
    while True:
        missing = band & ~in_tree
        if not missing.any():
            break
        # breadth-first search from the whole tree to the nearest missing band cell
        parent = np.full(m, -1, dtype=np.int64)
        seen = in_tree.copy()
        q = deque(np.flatnonzero(in_tree).tolist())
        hit = -1
        while q and hit < 0:
            c = q.popleft()
            for d in lat.nbrs[c]:
                if not seen[d]:
                    seen[d] = True
                    parent[d] = c
                    if missing[d]:
                        hit = d
                        break
                    q.append(d)
        if hit < 0:
            break  # unreachable pieces stay empty
        path = [hit]
        while not in_tree[path[-1]]:
            path.append(int(parent[path[-1]]))
        for a, b in zip(path[1:], path[:-1]):  # from the tree out to the new piece
            edges.append((a, b))
            in_tree[b] = True
        piece = bfs_tree(lat, [hit], allowed=band & ~in_tree | (np.arange(m) == hit))
        for a, b in piece:
            edges.append((int(a), int(b)))
            in_tree[a] = in_tree[b] = True
    return np.array(edges, dtype=np.int64).reshape(-1, 2)


class DendriteGenerator(PatternGenerator):
    style = "dendrite"

    def build(self):
        p = self.p
        reg = self.region
        lat = Lattice(reg, p.rotation_deg)
        cells = lat.cells
        if len(cells) < 4:
            raise ValueError("The shape is too small for a dendrite at this line spacing.")
        n = lat.n
        rng = self.rng
        eta = 0.3 + 2.7 * float(np.clip(p.dendrite_branching, 0.0, 1.0))
        c_xy = lat.centres
        r = np.hypot(c_xy[:, 0], c_xy[:, 1])
        gap_out = reg.polar_r(np.arctan2(c_xy[:, 1], c_xy[:, 0])) - r  # distance-ish to the outer wall
        valid = lat.valid

        occupied = np.zeros(n * n, dtype=bool)
        if p.dendrite_roots == "edge":
            # trunk: every cell near the outer wall, joined into ONE tree (pieces of the band that
            # are split by a narrow neck are linked through the shortest path inside)
            band = valid & (gap_out < 4.0)
            edges = _trunk(lat, band, int(np.flatnonzero(band)[np.argmax(r[band])]) if band.any() else -1)
            if len(edges):
                occupied[np.unique(edges)] = True
            src_r = max(reg.hole_r + 3.0, 3.0)
            blocked = r < src_r + 3.0  # the middle stays open and nothing piles up around it
        else:
            if reg.hole_r > 0:
                ring = valid & (r < reg.hole_r + 4.5)
                edges = bfs_tree(lat, np.flatnonzero(ring)[np.argsort(r[ring])], allowed=ring)
            else:
                edges = np.empty((0, 2), dtype=np.int64)
            if len(edges):
                occupied[np.unique(edges)] = True
            else:
                occupied[int(cells[np.argmin(r[cells])])] = True
            blocked = gap_out < 3.0  # don't crawl along the outer wall
        edge_list = [tuple(e) for e in edges]

        # Potential on a padded grid: the shape does not block the field ("DLA in open space");
        # it only limits where cells may be added. Centre growth: the pull comes from far away
        # (the padded border). Edge growth: from a small disk in the middle.
        pad = max(4, n // 2)
        N2 = n + 2 * pad
        ii, jj = np.meshgrid(np.arange(N2), np.arange(N2), indexing="ij")
        red = (ii + jj) % 2 == 0
        src = np.zeros((N2, N2), dtype=bool)
        R = max(float(r[valid].max()), 1.0)
        cx = (ii - pad) * 2 + 1.0 + lat.origin  # cell centre in lattice frame (rotation is irrelevant for r)
        cy = (jj - pad) * 2 + 1.0 + lat.origin
        rr = np.hypot(cx, cy)
        if p.dendrite_roots == "edge":
            src[rr < src_r] = True
            phi = np.clip(1 - rr / R, 0, 1)
        else:
            src[[0, -1], :] = True
            src[:, [0, -1]] = True
            phi = np.clip(rr / (R * 1.8), 0, 1)
        inner = (slice(pad, pad + n), slice(pad, pad + n))

        target = float(np.clip(p.fill, 0.05, 1.0)) * len(cells)
        E = lat.edges
        step = 0
        while occupied.sum() < target:
            occ2 = np.zeros((N2, N2), dtype=bool)
            occ2[inner] = occupied.reshape(n, n)
            fixed = occ2 | src
            phi[occ2] = 0.0
            phi[src & ~occ2] = 1.0
            for _ in range(SWEEPS):
                for colour in (red, ~red):
                    q = np.pad(phi, 1, mode="edge")  # the outer edge reflects
                    avg = 0.25 * (q[:-2, 1:-1] + q[2:, 1:-1] + q[1:-1, :-2] + q[1:-1, 2:])
                    upd = colour & ~fixed
                    phi[upd] = avg[upd]
            phi_cells = phi[inner].ravel()
            # frontier: empty cells next to a branch (each with one parent)
            oa, ob = occupied[E[:, 0]], occupied[E[:, 1]]
            new = np.concatenate([E[oa & ~ob, 1], E[ob & ~oa, 0]])
            par = np.concatenate([E[oa & ~ob, 0], E[ob & ~oa, 1]])
            # a new cell may touch only its parent: branches never grow side by side, so they keep
            # open gaps between them (the look of lightning / coral fans, not a maze)
            og = occupied.reshape(n, n)
            q = np.pad(og, 1).astype(np.int8)
            touching = (q[:-2, 1:-1] + q[2:, 1:-1] + q[1:-1, :-2] + q[1:-1, 2:]).ravel()
            keep = ~blocked[new] & (touching[new] == 1)
            new, par = new[keep], par[keep]
            if len(new) == 0:
                break
            perm = rng.permutation(len(new))
            new, par = new[perm], par[perm]
            new, first = np.unique(new, return_index=True)
            par = par[first]
            w = np.maximum(phi_cells[new], 0.0) ** eta + 1e-12
            k = int(min(len(new), max(1, 0.02 * occupied.sum()), target - occupied.sum() + 1))
            pick = rng.choice(len(new), size=k, replace=False, p=w / w.sum())
            occupied[new[pick]] = True
            edge_list.extend(zip(par[pick].tolist(), new[pick].tolist()))
            step += 1
            if step % 15 == 0 and edge_list:
                self.report(tree_to_loop(lat, np.array(edge_list)), 0.35 * occupied.sum() / target)

        if not edge_list:
            raise ValueError("The dendrite could not grow in this shape; try a bigger pattern.")
        loop = tree_to_loop(lat, np.array(edge_list))
        rounded = chaikin_closed(loop, 2)
        self.report(rounded, 0.35)
        P = self.organic(rounded, p.organic, 0.35, 1.0)
        self.finish(P, fallback=rounded, note=f"dendrite of {int(occupied.sum())} cells")
