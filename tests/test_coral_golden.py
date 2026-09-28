"""Golden tests: the coral style must keep producing bit-identical patterns.

Presets and "Regenerate same seed" depend on this. If a change to the engine is intended to alter
coral output, re-record the fingerprints with:  python tests/test_coral_golden.py --update
"""
import hashlib
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from growth import DifferentialGrowth, GrowthParams  # noqa: E402

GOLDEN_FILE = os.path.join(HERE, "golden_coral.json")

CONFIGS = {
    "default_circle": dict(seed=1, diameter_mm=120),
    "cross_hole_channels": dict(seed=2, diameter_mm=120, hole=0.12, channels=4, channel_width=2.5,
                                smoothness=0.8, wiggle=0.15),
    "ring_hexagon": dict(seed=3, diameter_mm=130, boundary="hexagon", start_shape="ring", hole=0.15, fill=0.7),
    "star_ellipse": dict(seed=4, diameter_mm=130, boundary="ellipse", aspect=0.7, start_shape="star",
                         start_size=0.3, channels=5, channel_length=0.8),
    "sleeping_big": dict(seed=5, diameter_mm=180, spacing_mm=3.5),  # > 3000 nodes: exercises sleeping
}


def fingerprint(cfg: dict) -> str:
    sim = DifferentialGrowth(GrowthParams.from_dict(cfg)).run()
    return hashlib.sha256(np.round(sim.P, 9).tobytes()).hexdigest()


class CoralGolden(unittest.TestCase):
    def test_bit_identical(self):
        with open(GOLDEN_FILE, encoding="utf-8") as f:
            golden = json.load(f)
        for name, cfg in CONFIGS.items():
            with self.subTest(config=name):
                self.assertEqual(fingerprint(cfg), golden[name],
                                 f"coral output changed for '{name}' (see module docstring)")


if __name__ == "__main__":
    if "--update" in sys.argv:
        out = {name: fingerprint(cfg) for name, cfg in CONFIGS.items()}
        with open(GOLDEN_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("recorded", GOLDEN_FILE)
    else:
        unittest.main()
