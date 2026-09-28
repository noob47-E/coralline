"""Every pattern type, in several shapes and seeds, must be ONE continuous closed line.

Runs each generator and puts the result through the same line test as the app's Test button,
with the raw export (smoothing 0) and the default smoothing (4).
"""
import unittest

import numpy as np

from helpers import failed_checks  # noqa: F401  (also sets up the import path)
from generators import make_generator
from growth import STYLES, GrowthParams

SHAPE_CASES = [
    dict(boundary="circle"),
    dict(boundary="heart"),
    dict(boundary="star", shape_depth=0.8),
    dict(boundary="hexagon", hole=0.15, channels=4),
    dict(boundary="blob", shape_seed=3, shape_lumpiness=0.9),
    dict(boundary="flower", shape_points=6, aspect=0.8, rotation_deg=20),
]
STYLE_EXTRAS = {
    "coral": [{}],
    "maze": [dict(maze_direction="radial"), dict(maze_direction="circular", maze_corridor=0.2, organic=0.9)],
    "dendrite": [dict(dendrite_roots="centre", fill=0.5), dict(dendrite_roots="edge", fill=0.5)],
    "spiral": [dict(spiral_mode="twist"), dict(spiral_mode="rings", ring_wobble=0.6, spiral_roundness=0.8)],
    "scribble": [dict(scribble_density="clouds"), dict(scribble_density="radial", organic=0.5)],
}


class StyleMatrix(unittest.TestCase):
    def test_every_style_every_shape(self):
        k = 0
        for style in STYLES:
            for extra in STYLE_EXTRAS[style]:
                for shape in SHAPE_CASES:
                    k += 1
                    cfg = dict(style=style, seed=100 + k, diameter_mm=110, **shape, **extra)
                    with self.subTest(**{kk: str(v) for kk, v in cfg.items()}):
                        gen = make_generator(GrowthParams.from_dict(cfg)).run()
                        self.assertFalse(gen.stop_reason.startswith("WARNING"), gen.stop_reason)
                        pts = gen.points_mm()
                        for smoothing in (0, 4):
                            self.assertEqual(failed_checks(pts, 4.5, smoothing), [],
                                             f"smoothing {smoothing}: {gen.stop_reason}")

    def test_same_seed_same_pattern(self):
        for style in STYLES:
            with self.subTest(style=style):
                cfg = dict(style=style, seed=7, diameter_mm=100, boundary="blob")
                a = make_generator(GrowthParams.from_dict(cfg)).run().P
                b = make_generator(GrowthParams.from_dict(cfg)).run().P
                c = make_generator(GrowthParams.from_dict(dict(cfg, seed=8))).run().P
                self.assertTrue(a.shape == b.shape and np.array_equal(a, b))
                self.assertFalse(a.shape == c.shape and np.allclose(a, c))

    def test_stop_is_honoured(self):
        import threading
        for style in STYLES:
            with self.subTest(style=style):
                ev = threading.Event()
                ev.set()
                gen = make_generator(GrowthParams(style=style, seed=3, diameter_mm=150)).run(stop_event=ev)
                self.assertTrue(gen.done)
                self.assertEqual(gen.stop_reason, "stopped")


if __name__ == "__main__":
    unittest.main()
