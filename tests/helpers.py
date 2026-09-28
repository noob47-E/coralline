"""Shared helpers for the test suite."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from exporters import ExportSettings  # noqa: E402
from linetest import run_line_test  # noqa: E402


def failed_checks(points_mm, spacing_mm, smoothing):
    """Titles of the line-test checks that fail (empty list = one continuous closed line)."""
    _, checks = run_line_test(points_mm, ExportSettings(smoothing=smoothing), spacing_mm)
    return [c.title for c in checks if not c.passed]
