"""Surprise me always gives a valid one-line pattern; the GUI metadata is consistent."""
import os
import unittest

import numpy as np

from helpers import ROOT, failed_checks
from generators import make_generator
from growth import STYLES, GrowthParams
from surprise import SHAPE_FIELDS, randomize


class Surprise(unittest.TestCase):
    def test_thirty_surprises_are_valid(self):
        rng = np.random.default_rng(2026)
        base = GrowthParams(diameter_mm=110, spacing_mm=4.5)
        seen_styles, seen_shapes = set(), set()
        for k in range(30):
            gp = randomize(base, rng)
            seen_styles.add(gp.style)
            seen_shapes.add(gp.boundary)
            with self.subTest(k=k, style=gp.style, shape=gp.boundary):
                self.assertEqual((gp.diameter_mm, gp.spacing_mm), (110, 4.5), "size must not change")
                gen = make_generator(gp).run()
                self.assertFalse(gen.stop_reason.startswith("WARNING"), gen.stop_reason)
                self.assertEqual(failed_checks(gen.points_mm(), gp.spacing_mm, 4), [])
        self.assertGreaterEqual(len(seen_styles), 4)
        self.assertGreaterEqual(len(seen_shapes), 6)

    def test_keep_boxes(self):
        rng = np.random.default_rng(5)
        base = GrowthParams(style="spiral", boundary="flower", shape_points=7, shape_depth=0.7)
        for _ in range(10):
            kept = randomize(base, rng, keep_shape=True, keep_style=True)
            self.assertEqual(kept.style, "spiral")
            for f in SHAPE_FIELDS:
                self.assertEqual(getattr(kept, f), getattr(base, f), f)
            self.assertNotEqual(kept.seed, base.seed)


class GuiMetadata(unittest.TestCase):
    def test_every_control_has_a_tip_and_a_setting(self):
        import app
        from exporters import ExportSettings
        fields = set(GrowthParams().to_dict()) | set(ExportSettings().to_dict())
        for _, controls in app.GROWTH_CONTROLS + app.EXPORT_CONTROLS:
            for key, *_ in controls:
                with self.subTest(key=key):
                    self.assertIn(key, fields)
                    self.assertIn(key, app.TIPS)
        for key in app.VISIBLE:
            self.assertIn(key, fields)
        for style in STYLES:
            self.assertIn(f"- {style}:", app.TIPS["style"])

    def test_presets_load_and_run(self):
        import json
        folder = os.path.join(ROOT, "presets")
        for name in sorted(os.listdir(folder)):
            with self.subTest(preset=name):
                with open(os.path.join(folder, name), encoding="utf-8") as f:
                    data = json.load(f)
                gp = GrowthParams.from_dict(dict(data["growth"], diameter_mm=100, seed=3))
                gen = make_generator(gp).run()
                self.assertFalse(gen.stop_reason.startswith("WARNING"), gen.stop_reason)


if __name__ == "__main__":
    unittest.main()
