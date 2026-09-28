# Coralline - generative coral lamp patterns drawn as one continuous line
# Copyright (C) 2026 Muhammad Wasiq
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# LICENSE file or <https://www.gnu.org/licenses/> for details.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Maze / labyrinth style: a random maze (spanning tree of grid cells) filling the whole shape,
drawn as the one closed line that runs around all its corridors.

The maze is grown with the "growing tree" algorithm: keep a list of active cells; each step pick
either the NEWEST active cell (long winding corridors, like depth-first search) or a RANDOM one
(short twisty branches); join it to an unvisited neighbour. maze_corridor sets that choice.
maze_direction prefers corridors that run radially (sunburst) or around the centre (rings).
"""
from __future__ import annotations

import numpy as np

from generators import PatternGenerator
from geometry import chaikin_closed
from gridtree import Lattice, tree_to_loop


class MazeGenerator(PatternGenerator):
    style = "maze"

    def build(self):
        p = self.p
        lat = Lattice(self.region, p.rotation_deg)
        cells = lat.cells
        if len(cells) < 4:
            raise ValueError("The shape is too small for a maze at this line spacing.")
        rng = self.rng
        n_total = len(cells)
        visited = np.zeros(lat.n * lat.n, dtype=bool)
        start = int(cells[np.argmin(np.hypot(*lat.centres[cells].T))])  # the middle cell
        visited[start] = True
        active = [start]
        edges: list[tuple[int, int]] = []
        corridor = float(np.clip(p.maze_corridor, 0.0, 1.0))
        bias = p.maze_direction
        every = max(40, n_total // 25)

        while active:
            k = len(active) - 1 if rng.random() < corridor else int(rng.integers(len(active)))
            c = active[k]
            cand = [d for d in lat.nbrs[c] if not visited[d]]
            if not cand:
                active[k] = active[-1]
                active.pop()
                continue
            if bias in ("radial", "circular") and len(cand) > 1:
                pc = lat.centres[c]
                radial = pc / max(np.hypot(*pc), 1e-9)
                dirs = (lat.centres[cand] - pc) / 2.0
                align = np.abs(dirs @ radial)
                if bias == "circular":
                    align = 1.0 - align
                w = 0.12 + align ** 2
                d = int(rng.choice(cand, p=w / w.sum()))
            else:
                d = int(cand[int(rng.integers(len(cand)))])
            visited[d] = True
            edges.append((c, d))
            active.append(d)
            if len(edges) % every == 0:
                self.report(tree_to_loop(lat, np.array(edges)), 0.3 * len(edges) / n_total)

        loop = tree_to_loop(lat, np.array(edges))
        rounded = chaikin_closed(loop, 2)  # soften the right angles; still valid by construction
        self.report(rounded, 0.3)
        P = self.organic(rounded, p.organic, 0.3, 1.0)
        note = f"maze of {n_total} cells"
        if lat.dropped:
            note += f", {lat.dropped} cells cut off by channels were left empty"
        self.finish(P, fallback=rounded, note=note)
