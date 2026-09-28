# Coralline - generative coral lamp patterns drawn as one continuous line
# Copyright (C) 2026 Muhammad Wasiq
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# LICENSE file or <https://www.gnu.org/licenses/> for details.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Batch generator: make many patterns without the GUI.

Examples:
  python cli.py --count 10 --formats svg,dxf --out patterns
  python cli.py --preset "presets/Coral cross.json" --seed 42 --formats svg,pdf,dxf,png
  python cli.py --set channels=6 --set hole=0.15 --count 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

from exporters import FORMATS, ExportSettings, export
from generators import make_generator
from growth import GrowthParams
from linetest import run_line_test


def main():
    ap = argparse.ArgumentParser(description="Generate one-line patterns (coral, maze, dendrite, spiral, scribble).")
    ap.add_argument("--preset", help="preset JSON file (from the app's Save preset)")
    ap.add_argument("--seed", type=int, help="random seed (default: random); with --count, seeds count up from it")
    ap.add_argument("--count", type=int, default=1, help="number of patterns to make")
    ap.add_argument("--formats", default="svg,pdf,dxf,png", help="comma list of: " + ",".join(FORMATS))
    ap.add_argument("--out", default="output", help="output folder")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any setting, e.g. --set spacing_mm=4 --set smoothing=4")
    args = ap.parse_args()

    growth, exp = {}, {}
    if args.preset:
        with open(args.preset, encoding="utf-8") as f:
            data = json.load(f)
        growth, exp = data.get("growth", {}), data.get("export", {})
    gkeys, ekeys = GrowthParams().to_dict(), ExportSettings().to_dict()
    for item in args.set:
        k, _, v = item.partition("=")
        if k in gkeys:
            growth[k] = v if isinstance(gkeys[k], str) else float(v)
        elif k in ekeys:
            exp[k] = v if isinstance(ekeys[k], str) else float(v)
        else:
            ap.error(f"unknown setting '{k}'. Known: {', '.join(list(gkeys) + list(ekeys))}")

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    for f in formats:
        if f not in FORMATS:
            ap.error(f"unknown format '{f}'")
    os.makedirs(args.out, exist_ok=True)
    es = ExportSettings.from_dict(exp)
    first_seed = args.seed if args.seed is not None else random.randint(1, 999_999)

    for n in range(args.count):
        gp = GrowthParams.from_dict(growth)
        gp.seed = first_seed + n
        t0 = time.time()
        sim = make_generator(gp).run()
        P = sim.points_mm()
        _, checks = run_line_test(P, es, gp.spacing_mm)
        failed = [c.title for c in checks if not c.passed]
        name = f"{gp.style}_seed{gp.seed}"
        for f in formats:
            export(P, os.path.join(args.out, f"{name}.{f}"), f, es)
        with open(os.path.join(args.out, f"{name}_settings.json"), "w", encoding="utf-8") as fh:
            json.dump({"growth": gp.to_dict(), "export": es.to_dict()}, fh, indent=2)
        status = ("line test PASSED (one continuous closed line)" if not failed
                  else "line test FAILED: " + "; ".join(failed))
        print(f"{name}: {len(sim.P)} points, {sim.iteration} steps, {sim.stop_reason}, "
              f"{time.time() - t0:.1f}s, {status}")


if __name__ == "__main__":
    main()
