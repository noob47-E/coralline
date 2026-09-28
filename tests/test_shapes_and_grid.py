"""Shapes, the Region distance function and the grid-tree -> one-loop builder."""
import unittest

import numpy as np

from helpers import ROOT  # noqa: F401  (sets up the import path)
from geometry import count_self_intersections
from growth import GrowthParams
from gridtree import Lattice, bfs_tree, tree_to_loop
from shapes import SHAPES, WALL_CLEAR, Region, _check_outline, outline


class Shapes(unittest.TestCase):
    def test_every_shape_suits_the_engine(self):
        for shape in SHAPES:
            for aspect in (0.7, 1.0, 1.4):
                for depth in (0.0, 1.0):
                    for lump in (0.0, 1.0):
                        gp = GrowthParams(boundary=shape, aspect=aspect, shape_depth=depth,
                                          shape_lumpiness=lump, shape_points=9, shape_seed=5)
                        with self.subTest(shape=shape, aspect=aspect, depth=depth, lump=lump):
                            try:
                                pts = outline(gp, 20.0)
                            except ValueError:
                                self.assertIn(shape, ("heart", "teardrop"), "only fixed shapes may refuse")
                                continue
                            self.assertIsNone(_check_outline(pts))
                            r = np.hypot(pts[:, 0], pts[:, 1])
                            self.assertAlmostEqual(r.max() / max(1.0, aspect), 20.0, delta=20.0 * 0.45)

    def test_many_random_blobs(self):
        for seed in range(1, 60):
            pts = outline(GrowthParams(boundary="blob", shape_seed=seed, shape_lumpiness=1.0), 25.0)
            self.assertIsNone(_check_outline(pts), f"blob {seed}")

    def test_sdf_matches_brute_force(self):
        gp = GrowthParams(boundary="flower", hole=0.2, channels=3, shape_points=5)
        reg = Region(gp, 20.0)
        reg.build_channels(reg.default_channel_span)
        rng = np.random.default_rng(0)
        pts = rng.uniform(-22, 22, size=(400, 2))
        fast = reg.sdf(pts)
        d = np.min(np.hypot(pts[:, None, 0] - reg.b_pts[None, :, 0], pts[:, None, 1] - reg.b_pts[None, :, 1]), axis=1)
        r = np.hypot(pts[:, 0], pts[:, 1])
        brute = np.where(reg.allowed_mask(pts) | (r < reg.polar_r(np.arctan2(pts[:, 1], pts[:, 0]))), d, -d)
        brute = np.minimum(brute, r - reg.hole_r)
        for u, r_in, r_out, half in reg.channels:
            t = np.clip(pts @ u, r_in, r_out)
            brute = np.minimum(brute, np.hypot(pts[:, 0] - t * u[0], pts[:, 1] - t * u[1]) - half)
        np.testing.assert_allclose(fast, brute, atol=1e-9)


class GridTree(unittest.TestCase):
    def test_any_tree_gives_one_loop(self):
        rng = np.random.default_rng(1)
        for shape in ("circle", "heart", "star", "blob"):
            gp = GrowthParams(boundary=shape, hole=0.15, channels=4, shape_seed=4)
            reg = Region(gp, 18.0)
            reg.build_channels(reg.default_channel_span)
            lat = Lattice(reg, rotation_deg=17)
            for trial in range(4):
                with self.subTest(shape=shape, trial=trial):
                    order = rng.permutation(lat.cells)
                    edges = bfs_tree(lat, order[:1])
                    if trial % 2:  # a random sub-tree: keep a random connected part
                        edges = edges[: max(1, int(len(edges) * rng.uniform(0.2, 0.9)))]
                    loop = tree_to_loop(lat, edges)
                    self.assertEqual(len(loop), 4 * len(np.unique(edges)))
                    self.assertEqual(count_self_intersections(loop), 0)
                    self.assertGreaterEqual(reg.sdf(loop).min(), WALL_CLEAR)

    def test_a_cycle_is_rejected(self):
        reg = Region(GrowthParams(), 10.0)
        lat = Lattice(reg)
        c = int(lat.cells[len(lat.cells) // 2])
        n = lat.n
        square = np.array([(c, c + 1), (c + 1, c + 1 + n), (c + n, c + 1 + n), (c, c + n)])
        with self.assertRaises(AssertionError):
            tree_to_loop(lat, square)


if __name__ == "__main__":
    unittest.main()
