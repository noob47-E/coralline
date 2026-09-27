"""Proof that a pattern is ONE continuous, closed line that never crosses or touches itself.

The checks run on exactly the line that gets exported (same smoothing and point spacing),
and the last check builds the real SVG / PDF / DXF files in memory and counts the lines in them.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from exporters import ExportSettings, dxf_string, pdf_bytes, smooth_points, svg_string
from geometry import count_self_intersections
from growth import MAX_EDGE


@dataclass
class Check:
    passed: bool
    title: str
    detail: str


def _clearance(P: np.ndarray, steps: np.ndarray, spacing_mm: float) -> float:
    """Smallest distance between two parts of the line that are far apart ALONG the line."""
    arc = np.concatenate([[0.0], np.cumsum(steps)[:-1]])
    total = steps.sum()
    pairs = cKDTree(P).query_pairs(2.0 * spacing_mm, output_type="ndarray")
    if len(pairs) == 0:
        return float("inf")
    da = np.abs(arc[pairs[:, 0]] - arc[pairs[:, 1]])
    da = np.minimum(da, total - da)
    far = pairs[da > 2.5 * spacing_mm]
    if len(far) == 0:
        return float("inf")
    return float(np.linalg.norm(P[far[:, 0]] - P[far[:, 1]], axis=1).min())


def _file_check(P_mm: np.ndarray, es: ExportSettings, n_points: int) -> Check:
    problems = []
    # SVG: one <path>, one pen-down (M), closed (Z)
    svg = svg_string(P_mm, es)
    n_paths = svg.count("<path")
    d = re.search(r' d="([^"]+)"', svg).group(1)
    n_moves = d.count("M")
    svg_ok = n_paths == 1 and n_moves == 1 and d.rstrip().endswith("Z")
    if not svg_ok:
        problems.append("SVG")
    # PDF: one moveto (m), closed and stroked once (h S)
    pdf = pdf_bytes(P_mm, es)
    start = pdf.index(b"stream\n") + 7
    ops = zlib.decompress(pdf[start:pdf.index(b"\nendstream")]).decode("ascii").splitlines()
    n_m = sum(1 for line in ops if line.endswith(" m"))
    n_close = sum(1 for line in ops if line.strip() == "h S")
    pdf_ok = n_m == 1 and n_close == 1
    if not pdf_ok:
        problems.append("PDF")
    # DXF: one POLYLINE, closed flag set, all points present
    codes = dxf_string(P_mm, es).splitlines()
    pairs = list(zip(codes[0::2], codes[1::2]))
    n_poly = sum(1 for c, v in pairs if c == "0" and v == "POLYLINE")
    n_vert = sum(1 for c, v in pairs if c == "0" and v == "VERTEX")
    poly_i = next(i for i, (c, v) in enumerate(pairs) if c == "0" and v == "POLYLINE")
    closed_flag = next(v for c, v in pairs[poly_i:] if c == "70") == "1"
    dxf_ok = n_poly == 1 and closed_flag and n_vert == n_points
    if not dxf_ok:
        problems.append("DXF")
    detail = (f"SVG: {n_paths} path with {n_moves} pen-down, closed.   "
              f"PDF: {n_m} pen-down, closed.   "
              f"DXF: {n_poly} closed polyline with {n_vert:,} points.")
    if problems:
        detail += "  Problem in: " + ", ".join(problems)
    return Check(not problems, "The exported files contain exactly one line", detail)


def run_line_test(P_mm: np.ndarray, es: ExportSettings, spacing_mm: float):
    """Return (exported_points_mm, [Check, ...]).

    P_mm is the raw grown line. Gaps are checked on it, because the export step respaces points
    evenly and would quietly bridge a gap with a straight segment.
    """
    raw_steps = np.linalg.norm(np.roll(P_mm, -1, axis=0) - P_mm, axis=1)
    limit = MAX_EDGE * spacing_mm * 1.05  # growth splits any longer step, so nothing bigger may exist
    checks = []

    worst = float(raw_steps[:-1].max())
    checks.append(Check(
        worst <= limit, "One single piece - no gaps or breaks",
        f"The grown line is one chain of {len(P_mm):,} points, each connected to the next. The biggest "
        f"jump between two neighbouring points is {worst:.2f} mm (the growth never allows more than "
        f"{MAX_EDGE * spacing_mm:.2f} mm), so the pen never has to lift."))

    gap = float(raw_steps[-1])
    checks.append(Check(
        gap <= limit, "Closed loop - the end joins the start",
        f"The last point connects back to the first with a {gap:.2f} mm step, just like every other "
        f"step. The line has no loose ends."))

    P = smooth_points(P_mm, es)
    n = len(P)
    steps = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)

    crossings = count_self_intersections(P)
    checks.append(Check(
        crossings == 0, "Never crosses itself",
        f"Every one of the {n:,} segments of the exported line was checked against all nearby "
        f"segments: {crossings} crossings found. Total line length: {steps.sum() / 1000:.2f} m."))

    clear = _clearance(P, steps, spacing_mm)
    min_ok = max(es.stroke_mm, 0.2 * spacing_mm)
    clear_txt = "no other part of the line comes near" if clear == float("inf") else f"{clear:.2f} mm"
    checks.append(Check(
        clear > min_ok, "Never touches itself",
        f"The closest that two different parts of the line come to each other: {clear_txt} "
        f"(Line spacing is {spacing_mm:g} mm). Keep the lamp's wall thickness below this so the walls "
        f"don't merge."))

    checks.append(_file_check(P_mm, es, n))
    return P, checks
