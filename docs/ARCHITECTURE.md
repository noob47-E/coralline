# Coralline architecture

This is a guide for anyone who wants to change the code: how the pieces fit together, the one rule
every piece must keep, and where to look when something goes wrong.

## The one rule

**Every pattern is ONE closed line that never crosses or touches itself.** Lamps, pen plotters and
laser cutters all depend on it. Every part of the code either builds such a line or keeps it valid:

| Guarantee | Where it comes from |
|---|---|
| One closed loop | Each style builds it that way (see below). The line is always stored as one array of points, joined end to end. |
| No gaps | Points are at most `MAX_EDGE` = 0.7 spacings apart (`geometry.resample_closed`, and `DifferentialGrowth._split_long`). |
| No crossings, no touching | Repulsion in the physics engine; and for the grid and spiral styles, the construction itself. |
| Inside the shape | `Region.clamp`, plus the wall forces in the physics engine. |
| Checked | `generators.gate()` runs at the end of every non-coral style, and `linetest.run_line_test()` is behind the app's Test button and the tests. |

All geometry is in **spacing units**: 1.0 = one line spacing. It is converted to millimetres only on
export (`points_mm()`).

## Modules

```
app.py          Tkinter GUI: controls (declared in GROWTH_CONTROLS / VISIBLE / TIPS), threading, preview,
                line-test animation, exports, Surprise me
cli.py          batch generator (same generators, same line test)

growth.py       GrowthParams (every setting, for every style) and DifferentialGrowth (the coral style,
                also used by other styles for their "organic finish")
generators.py   PatternGenerator base class, gate(), make_generator() registry
shapes.py       outer-shape library (polar r(theta) outlines) and Region (shape + hole + channels:
                inside test, distance to walls, wall forces, clamp)
gridtree.py     Lattice (grid cells inside a Region) and tree_to_loop (any tree of cells -> one loop)
maze.py         maze style (growing-tree maze)            } built on gridtree
dendrite.py     dendrite style (dielectric-breakdown growth) }
spiral.py       spiral-rings style (nested rings -> double spiral)
scribble.py     one-line scribble style (dots -> tour -> untangle -> relax)
surprise.py     Surprise me: randomize(params, rng, keep_shape, keep_style)

geometry.py     closed-polyline helpers: resampling, smoothing, crossing count
exporters.py    SVG / PDF / DXF / PNG writers (no extra libraries)
linetest.py     the "one continuous line" proof used by the app and the tests
```

## How each style guarantees one line

- **Coral** (`growth.DifferentialGrowth`): starts from a small loop and only ever inserts points into
  it, so it stays one loop. Points push apart within 1 spacing and are clamped inside the region.
  The time step (`dt = 0.5`) keeps the integration stable; at 1.0, points bounced back and forth.
  *Sleeping*: in big patterns, crowded points that have stopped moving are frozen and act as fixed
  obstacles. Only the growing parts are simulated, and every 10–20 steps a full step re-checks
  everything.
- **Maze and dendrite** (`gridtree.tree_to_loop`): a tree of 2×2-spacing cells. Each cell owns a small
  square loop, and each tree edge merges two loops into one. Because the cells form one tree (connected,
  no cycles), the result is exactly one loop through 4 × cells points. The function asserts this.
  - The maze makes the tree with the growing-tree algorithm.
  - The dendrite adds cells with probability φ^η, where φ is a Laplace potential. Each new cell hangs on
    exactly one existing cell, and may not touch any other branch.
- **Spiral** (`spiral.py`): rings are radii on rays from the centre. Every ring is at least 1 spacing
  inside the previous one, so on each ray the "level" maps to a strictly decreasing radius. Lane A
  moves one level every half turn, and lane B = A + 1. All passes on a ray therefore sit at different
  radii, so the lanes can't cross. If fill is asked for, the innermost lanes may then grow coral
  into the middle (the `growable` mask in `DifferentialGrowth`).
- **Scribble** (`scribble.py`): a tour through dots, ordered along a maze path. A 2-opt move is applied
  while any two edges cross. Each move shortens the tour, so the loop always ends with zero crossings.
  Then the physics relaxes it so no two parts nearly touch.

`PatternGenerator.finish()` resamples the curve and runs `gate()`. If the organic finish ever breaks
the curve, it falls back to the construction curve, which is valid by construction.

## Shapes

All shapes are *star-shaped* around the origin: a ray from the centre crosses the outline once. That
lets `Region.polar_r(theta)` give the wall radius in any direction, and lets `Region.clamp` push a stray
point back along its ray.

New shapes are checked by `shapes._check_outline`:
- the outline goes round monotonically;
- walls are at most ~75° off radial (cos ≥ 0.25);
- there is a solid middle (inner radius ≥ 0.2 × outer).

Star and flower depth and blob lumpiness are toned down automatically until the checks pass. The six
original shapes (`LEGACY_SHAPES`) keep their exact original construction, so old seeds reproduce.

## Tests

```
python -m unittest discover -s tests -t tests      (about 1 minute)
```

| File | What it checks |
|---|---|
| `test_coral_golden.py` | Coral output is bit-identical to the recorded fingerprints (presets and saved seeds depend on it). Re-record deliberately with `python tests/test_coral_golden.py --update`. |
| `test_styles.py` | Every style × 6 shapes × 2 variants passes the line test at smoothing 0 and 4; same seed gives the same pattern; Stop works. |
| `test_shapes_and_grid.py` | Every shape at extreme settings, 60 random blobs, `Region.sdf` against brute force, `tree_to_loop` on random trees, and a cycle is rejected. |
| `test_surprise_and_gui.py` | 30 random Surprises pass; the keep boxes work; every control has a setting and a tip; every preset runs. |

## Adding a new style

1. Write `mystyle.py` with `class MyGenerator(PatternGenerator)` and a `build()` that constructs a
   valid loop (in spacing units), calls `self.report(P, progress)` now and then, optionally calls
   `self.organic(...)`, and ends with `self.finish(P, fallback=...)`.
2. Add it to `growth.STYLES` and to `generators._registry()` (use a static import so PyInstaller
   bundles it).
3. Add its settings to `GrowthParams`, its controls to `app.GROWTH_CONTROLS`, a rule to `app.VISIBLE`,
   a tip to `app.TIPS` (and to the `style` tip), and ranges to `surprise._random_settings`.
4. Add it to `STYLE_EXTRAS` in `tests/test_styles.py` and run the tests.
