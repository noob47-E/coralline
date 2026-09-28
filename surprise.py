# Coralline - generative coral lamp patterns drawn as one continuous line
# Copyright (C) 2026 Muhammad Wasiq
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# LICENSE file or <https://www.gnu.org/licenses/> for details.
# SPDX-License-Identifier: GPL-3.0-or-later

"""The "Surprise me" button: random pattern type, shape and settings from curated, safe ranges.

Size, line spacing, the iteration limit and every export setting are never touched.
keep_shape keeps the outer shape (incl. its points/depth/lumpiness and the blob's shape seed);
keep_style keeps the pattern type (its settings are still randomized).
"""
from __future__ import annotations


import numpy as np

from growth import STYLES, GrowthParams
from shapes import SHAPES, outline

SHAPE_FIELDS = ("boundary", "aspect", "rotation_deg", "shape_points", "shape_depth",
                "shape_lumpiness", "shape_seed")
# upright shapes look best without random rotation
_UPRIGHT = {"heart", "teardrop"}


def _pick(rng, options, weights=None):
    w = None if weights is None else np.array(weights, dtype=float) / sum(weights)
    return options[int(rng.choice(len(options), p=w))]


def _random_shape(rng, d: dict):
    d["boundary"] = _pick(rng, SHAPES)
    b = d["boundary"]
    d["aspect"] = 1.0 if b == "circle" or rng.random() < 0.7 else round(float(rng.uniform(0.75, 1.3)), 2)
    if b == "circle" or b in _UPRIGHT:
        d["rotation_deg"] = 0.0
    else:
        d["rotation_deg"] = float(rng.choice([0, 0, 0, 15, 30, 45, 90]))
    d["shape_points"] = int(rng.integers(4, 10))
    d["shape_depth"] = round(float(rng.uniform(0.3, 0.9)), 2)
    d["shape_lumpiness"] = round(float(rng.uniform(0.3, 1.0)), 2)
    d["shape_seed"] = int(rng.integers(1, 10000))


def _structure(rng, d: dict, hole_p: float, channel_p: float):
    d["hole"] = round(float(rng.uniform(0.08, 0.2)), 2) if rng.random() < hole_p else 0.0
    if rng.random() < channel_p:
        d["channels"] = int(rng.integers(3, 9))
        d["channel_width"] = round(float(rng.uniform(1.5, 3.0)), 1)
        d["channel_length"] = 1.0 if rng.random() < 0.5 else round(float(rng.uniform(0.6, 0.95)), 2)
        d["channel_rotation_deg"] = float(rng.choice([0, 90, 45, 30, float(rng.integers(0, 360))]))
    else:
        d["channels"] = 0


def _random_settings(rng, d: dict):
    u = lambda a, b: round(float(rng.uniform(a, b)), 2)  # noqa: E731
    style = d["style"]
    d["smoothness"] = u(0.4, 0.9)
    d["wiggle"] = u(0.05, 0.4)
    if style == "coral":
        d["fill"] = u(0.6, 0.9)
        d["growth_speed"] = u(0.3, 0.7)
        d["branchiness"] = u(0.2, 0.9)
        d["start_shape"] = _pick(rng, ["circle", "ring", "star"], [0.5, 0.3, 0.2])
        d["start_size"] = u(0.08, 0.25)
        d["star_arms"] = int(rng.integers(3, 9))
        _structure(rng, d, 0.35, 0.45)
    elif style == "maze":
        d["maze_corridor"] = u(0.2, 1.0)
        d["maze_direction"] = _pick(rng, ["any", "radial", "circular"], [0.5, 0.25, 0.25])
        d["organic"] = u(0.2, 0.8)
        _structure(rng, d, 0.3, 0.25)
    elif style == "dendrite":
        d["dendrite_branching"] = u(0.3, 1.0)
        d["dendrite_roots"] = _pick(rng, ["centre", "edge"])
        d["fill"] = u(0.4, 0.7)
        d["organic"] = u(0.3, 0.8)
        _structure(rng, d, 0.3, 0.25)
    elif style == "spiral":
        d["spiral_mode"] = _pick(rng, ["twist", "rings"])
        d["spiral_roundness"] = u(0.0, 0.8)
        d["ring_wobble"] = u(0.0, 0.6)
        d["fill"] = u(0.5, 0.9)
        _structure(rng, d, 0.35, 0.0)
    elif style == "scribble":
        d["scribble_density"] = _pick(rng, ["uniform", "radial", "clouds"])
        d["scribble_gap"] = u(1.3, 2.4)
        d["organic"] = u(0.0, 0.5)
        _structure(rng, d, 0.25, 0.2)


def describe(gp: GrowthParams) -> str:
    shape = gp.boundary
    article = "an" if shape[0] in "aeiou" else "a"
    return f"{gp.style} in {article} {shape}"


def randomize(gp: GrowthParams, rng: np.random.Generator, keep_shape: bool = False,
              keep_style: bool = False) -> GrowthParams:
    """A new random GrowthParams (never changes size, spacing, iteration limit or export settings)."""
    base = gp.to_dict()
    for _ in range(20):
        d = dict(base)
        if not keep_style:
            d["style"] = _pick(rng, STYLES)
        if not keep_shape:
            _random_shape(rng, d)
        _random_settings(rng, d)
        d["seed"] = int(rng.integers(1, 1_000_000))
        new = GrowthParams.from_dict(d)
        try:  # the shape must suit the engine (e.g. not a heart squashed flat)
            outline(new, max(4.0, 0.5 * new.diameter_mm / new.spacing_mm))
        except ValueError:
            continue
        return new
    return GrowthParams.from_dict(dict(base, seed=int(rng.integers(1, 1_000_000))))
