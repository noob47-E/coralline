# Coralline - generative coral lamp patterns drawn as one continuous line
# Copyright (C) 2026 Muhammad Wasiq
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# LICENSE file or <https://www.gnu.org/licenses/> for details.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pattern styles and the shared machinery behind them.

Every style produces ONE closed curve in spacing units, exposed through the same small interface
that the app, the CLI and the line test use (DifferentialGrowth, the coral style, has it too):

    gen = make_generator(params)
    gen.run(stop_event, callback)     # callback(gen) is called now and then while it works
    gen.P, gen.progress, gen.iteration, gen.done, gen.stop_reason
    gen.points_mm(), gen.boundary_mm(), gen.self_intersections()

Non-coral styles subclass PatternGenerator and implement build(): construct a curve that is
valid by construction, optionally soften it with the coral physics (organic()), then finish().
finish() runs the safety gate and falls back to the construction curve if softening broke it.
"""
from __future__ import annotations

import threading

import numpy as np
from scipy.spatial import cKDTree

from geometry import closed_length, count_self_intersections, resample_closed
from growth import MAX_EDGE, REST, STYLES, DifferentialGrowth, GrowthParams
from shapes import WALL_CLEAR, Region

GATE_CLEARANCE = 0.35  # different parts of the line must stay this far apart (line test: 0.2)


class Stopped(Exception):
    """Raised inside build() when the user presses Stop."""


def gate(P: np.ndarray, region: Region) -> str | None:
    """None if P is one valid closed line for this region, else the reason it is not."""
    if len(P) < 8:
        return "too few points"
    e = np.roll(P, -1, axis=0) - P
    if np.hypot(e[:, 0], e[:, 1]).max() > MAX_EDGE * 1.001:
        return "gap in the line"
    if region.sdf(P).min() < WALL_CLEAR - 0.05:
        return "too close to a wall"
    if count_self_intersections(P) != 0:
        return "the line crosses itself"
    # clearance between parts of the line that are far apart along it
    steps = np.hypot(e[:, 0], e[:, 1])
    arc = np.concatenate([[0.0], np.cumsum(steps)[:-1]])
    total = steps.sum()
    pairs = cKDTree(P).query_pairs(GATE_CLEARANCE, output_type="ndarray")
    if len(pairs):
        da = np.abs(arc[pairs[:, 0]] - arc[pairs[:, 1]])
        if (np.minimum(da, total - da) > 2.5).any():
            return "the line touches itself"
    return None


class PatternGenerator:
    """Base class for the non-coral styles."""

    style = ""
    uses_channels = True

    def __init__(self, params: GrowthParams):
        self.p = params
        self.Rb = max(4.0, 0.5 * params.diameter_mm / params.spacing_mm)
        self.region = Region(params, self.Rb)
        if self.uses_channels:
            self.region.build_channels(self.region.default_channel_span)
        seq = np.random.SeedSequence(params.seed)
        self.rng, self.rng_organic = (np.random.default_rng(s) for s in seq.spawn(2))
        self.P = np.zeros((0, 2))
        self.iteration = 0
        self.progress = 0.0
        self.done = False
        self.stop_reason = ""
        self._stop: threading.Event | None = None
        self._callback = None

    # --- to implement
    def build(self):
        raise NotImplementedError

    # --- helpers for build()
    def report(self, P: np.ndarray | None = None, progress: float | None = None):
        """Show progress (and optionally a partial curve); raises Stopped when Stop was pressed."""
        if P is not None:
            self.P = P
        if progress is not None:
            self.progress = float(min(1.0, max(0.0, progress)))
        self.iteration += 1
        if self._stop is not None and self._stop.is_set():
            raise Stopped
        if self._callback is not None and len(self.P) >= 3:
            self._callback(self)

    def organic(self, P: np.ndarray, amount: float, p0: float, p1: float,
                min_steps: int = 0, extra: float | None = None,
                growable: np.ndarray | None = None) -> np.ndarray:
        """Soften P with the coral physics. amount 0..1: 0 = untouched (unless min_steps),
        1 = relax and let the line grow up to 20% longer into free space.
        extra: grow by this fraction instead (e.g. to fill an empty middle)."""
        if amount <= 0 and min_steps <= 0 and not extra:
            return P
        grow = 0.2 * max(0.0, amount) if extra is None else max(0.0, extra)
        sim = DifferentialGrowth(self.p, initial_points=P, extra_length=grow,
                                 region=self.region, rng=self.rng_organic, growable=growable)
        relax = int(max(min_steps, 60 * amount))
        budget = int(1.2e7 / max(len(sim.P), 1))  # keeps huge patterns from taking forever
        steps = 1500 * amount if extra is None else 400 + 6000 * min(extra, 1.0)
        sim.p = _with(self.p, max_iterations=int(min(self.p.max_iterations, max(1, steps), budget)))

        def cb(s):
            self.report(s.P, p0 + (p1 - p0) * s.progress)

        sim.run(stop_event=self._stop, callback=cb, callback_every=4, relax_steps=min(relax, budget))
        if self._stop is not None and self._stop.is_set():
            raise Stopped
        return sim.P

    def finish(self, P: np.ndarray, fallback: np.ndarray | None = None, note: str = ""):
        """Resample, run the safety gate, and fall back to the construction curve if needed."""
        P = resample_closed(np.asarray(P, dtype=float), REST)
        why = gate(P, self.region)
        if why is not None and fallback is not None:
            F = resample_closed(np.asarray(fallback, dtype=float), REST)
            if gate(F, self.region) is None:
                self.P = F
                self.stop_reason = f"{note or 'done'} (organic finish skipped: {why})"
                return
        self.P = P
        self.stop_reason = (note or "done") if why is None else f"WARNING: {why}"

    # --- driver
    def run(self, stop_event: threading.Event | None = None, callback=None, callback_every: int = 1,
            relax_steps: int = 0):
        self._stop, self._callback = stop_event, callback
        try:
            self.build()
            self.progress = 1.0
        except Stopped:
            self.stop_reason = "stopped"
        self.done = True
        if callback is not None and len(self.P) >= 3:
            callback(self)
        return self

    # --- output (same as DifferentialGrowth)
    @property
    def length(self) -> float:
        return closed_length(self.P) if len(self.P) > 1 else 0.0

    def points_mm(self) -> np.ndarray:
        return self.P * self.p.spacing_mm

    def boundary_mm(self) -> np.ndarray:
        return self.region.b_pts * self.p.spacing_mm

    def self_intersections(self) -> int:
        return count_self_intersections(self.P)


def _with(params: GrowthParams, **changes) -> GrowthParams:
    d = params.to_dict()
    d.update(changes)
    return GrowthParams.from_dict(d)


def _registry() -> dict:
    # static imports so PyInstaller bundles every style
    from dendrite import DendriteGenerator
    from maze import MazeGenerator
    from scribble import ScribbleGenerator
    from spiral import SpiralGenerator
    return {"coral": DifferentialGrowth, "maze": MazeGenerator, "dendrite": DendriteGenerator,
            "spiral": SpiralGenerator, "scribble": ScribbleGenerator}


def make_generator(params: GrowthParams):
    """The generator for params.style (coral by default)."""
    reg = _registry()
    if params.style not in reg:
        raise ValueError(f"unknown pattern type '{params.style}'; choose one of {', '.join(STYLES)}")
    return reg[params.style](params)
