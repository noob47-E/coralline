"""Export a closed curve (points in millimetres, y axis up) to SVG, PDF, DXF and PNG.

No extra libraries are needed except Pillow for PNG.
"""
from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass, fields

import numpy as np
from PIL import Image, ImageDraw

from geometry import catmull_rom_beziers, prepare_output

MM_TO_PT = 72.0 / 25.4
FORMATS = ["svg", "pdf", "dxf", "png"]
MAX_PNG_PIXELS = 150_000_000   # final PNG (about 12,000 x 12,000)
MAX_WORK_PIXELS = 160_000_000  # supersampled drawing buffer (~480 MB)


@dataclass
class ExportSettings:
    smoothing: int = 3            # 0 = raw simulation line, higher = rounder
    point_spacing_mm: float = 1.2  # distance between exported points
    stroke_mm: float = 0.5        # line width in SVG / PDF / PNG
    margin_mm: float = 5.0
    png_dpi: int = 300
    png_style: str = "line"       # line | filled
    line_color: str = "#000000"
    fill_color: str = "#f2c46d"
    background: str = "#ffffff"

    @classmethod
    def from_dict(cls, d: dict) -> "ExportSettings":
        names = {f.name for f in fields(cls)}
        out = cls()
        for k, v in d.items():
            if k in names:
                setattr(out, k, type(getattr(out, k))(v))
        return out

    def to_dict(self) -> dict:
        return asdict(self)


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _layout(P: np.ndarray, margin: float):
    """Shift points so the drawing starts at (margin, margin); return points, width, height."""
    lo = P.min(axis=0)
    hi = P.max(axis=0)
    Q = P - lo + margin
    w, h = (hi - lo) + 2 * margin
    return Q, float(w), float(h)


def smooth_points(P_mm: np.ndarray, s: ExportSettings) -> np.ndarray:
    return prepare_output(P_mm, s.smoothing, s.point_spacing_mm)


def svg_string(P_mm: np.ndarray, s: ExportSettings) -> str:
    Q, w, h = _layout(smooth_points(P_mm, s), s.margin_mm)
    Q[:, 1] = h - Q[:, 1]  # SVG y axis points down
    a, c1, c2, b = catmull_rom_beziers(Q)
    parts = [f"M{a[0, 0]:.3f},{a[0, 1]:.3f}"]
    for i in range(len(a)):
        parts.append(f"C{c1[i, 0]:.3f},{c1[i, 1]:.3f} {c2[i, 0]:.3f},{c2[i, 1]:.3f} {b[i, 0]:.3f},{b[i, 1]:.3f}")
    parts.append("Z")
    d = " ".join(parts)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.3f}mm" height="{h:.3f}mm" '
        f'viewBox="0 0 {w:.3f} {h:.3f}">\n'
        f'  <path d="{d}" fill="none" stroke="{s.line_color}" stroke-width="{s.stroke_mm}" '
        'stroke-linejoin="round"/>\n'
        "</svg>\n"
    )


def export_svg(P_mm: np.ndarray, path: str, s: ExportSettings):
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_string(P_mm, s))


def pdf_bytes(P_mm: np.ndarray, s: ExportSettings) -> bytes:
    Q, w, h = _layout(smooth_points(P_mm, s), s.margin_mm)
    Q = Q * MM_TO_PT  # PDF y axis points up, like ours
    a, c1, c2, b = catmull_rom_beziers(Q)
    r, g, bl = (v / 255 for v in _hex_to_rgb(s.line_color))
    ops = [f"{r:.3f} {g:.3f} {bl:.3f} RG", f"{s.stroke_mm * MM_TO_PT:.3f} w", "1 J 1 j",
           f"{a[0, 0]:.3f} {a[0, 1]:.3f} m"]
    for i in range(len(a)):
        ops.append(f"{c1[i, 0]:.3f} {c1[i, 1]:.3f} {c2[i, 0]:.3f} {c2[i, 1]:.3f} {b[i, 0]:.3f} {b[i, 1]:.3f} c")
    ops.append("h S")
    stream = zlib.compress("\n".join(ops).encode("ascii"))
    W, H = w * MM_TO_PT, h * MM_TO_PT

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W:.3f} {H:.3f}] /Contents 4 0 R /Resources << >> >>".encode(),
        f"<< /Length {len(stream)} /Filter /FlateDecode >>\nstream\n".encode() + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def export_pdf(P_mm: np.ndarray, path: str, s: ExportSettings):
    with open(path, "wb") as f:
        f.write(pdf_bytes(P_mm, s))


def dxf_string(P_mm: np.ndarray, s: ExportSettings) -> str:
    """AutoCAD R12 DXF with one closed POLYLINE, units = millimetres."""
    Q = smooth_points(P_mm, s)
    Q = Q - Q.min(axis=0)
    lo, hi = Q.min(axis=0), Q.max(axis=0)
    lines = [
        "0", "SECTION", "2", "HEADER",
        "9", "$ACADVER", "1", "AC1009",
        "9", "$INSUNITS", "70", "4",
        "9", "$EXTMIN", "10", f"{lo[0]:.4f}", "20", f"{lo[1]:.4f}", "30", "0.0",
        "9", "$EXTMAX", "10", f"{hi[0]:.4f}", "20", f"{hi[1]:.4f}", "30", "0.0",
        "0", "ENDSEC",
        "0", "SECTION", "2", "ENTITIES",
        "0", "POLYLINE", "8", "PATTERN", "66", "1", "70", "1",
        "10", "0.0", "20", "0.0", "30", "0.0",
    ]
    for x, y in Q:
        lines += ["0", "VERTEX", "8", "PATTERN", "10", f"{x:.4f}", "20", f"{y:.4f}", "30", "0.0"]
    lines += ["0", "SEQEND", "8", "PATTERN", "0", "ENDSEC", "0", "EOF"]
    return "\n".join(lines) + "\n"


def export_dxf(P_mm: np.ndarray, path: str, s: ExportSettings):
    with open(path, "w", encoding="ascii", newline="\r\n") as f:
        f.write(dxf_string(P_mm, s))


def render_image(P_mm: np.ndarray, s: ExportSettings, dpi: int | None = None) -> Image.Image:
    dpi = dpi or s.png_dpi
    Q, w, h = _layout(smooth_points(P_mm, s), s.margin_mm)
    out_w, out_h = w * dpi / 25.4, h * dpi / 25.4
    if out_w * out_h > MAX_PNG_PIXELS:
        raise ValueError(f"A {w:.0f} x {h:.0f} mm PNG at {dpi} DPI would be {out_w:,.0f} x {out_h:,.0f} pixels, "
                         f"which is too big. Choose a lower PNG DPI, or use SVG / PDF, which stay sharp at any size.")
    # supersample for smooth edges, but less for big images so memory stays reasonable
    ss = next(k for k in (3, 2, 1) if out_w * out_h * k * k <= MAX_WORK_PIXELS or k == 1)
    px = dpi / 25.4 * ss
    W, H = max(1, int(round(w * px))), max(1, int(round(h * px)))
    img = Image.new("RGB", (W, H), _hex_to_rgb(s.background))
    draw = ImageDraw.Draw(img)
    pts = [(x * px, (h - y) * px) for x, y in Q]
    if s.png_style == "filled":
        draw.polygon(pts, fill=_hex_to_rgb(s.fill_color))
    width = max(1, int(round(s.stroke_mm * px)))
    draw.line(pts + pts[:1], fill=_hex_to_rgb(s.line_color), width=width, joint="curve")
    return img.resize((max(1, W // ss), max(1, H // ss)), Image.LANCZOS)


def export_png(P_mm: np.ndarray, path: str, s: ExportSettings):
    render_image(P_mm, s).save(path, dpi=(s.png_dpi, s.png_dpi))


EXPORTERS = {"svg": export_svg, "pdf": export_pdf, "dxf": export_dxf, "png": export_png}


def export(P_mm: np.ndarray, path: str, fmt: str, s: ExportSettings):
    EXPORTERS[fmt.lower()](P_mm, path, s)
