"""Coral Pattern Generator - desktop app (Tkinter).

Run:  python app.py
"""
from __future__ import annotations

import colorsys
import json
import os
import random
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from exporters import FORMATS, ExportSettings, export, smooth_points
from geometry import closed_length, count_self_intersections
from growth import BOUNDARIES, START_SHAPES, DifferentialGrowth, GrowthParams
from linetest import run_line_test

APP_TITLE = "Coral Pattern Generator"
BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
PRESET_DIR = os.path.join(BASE_DIR, "presets")

# (key, label, kind, options) - kind "scale": (min, max, step); kind "combo": list of values
GROWTH_CONTROLS = [
    ("Size & shape", [
        ("diameter_mm", "Pattern width (mm)", "scale", (40, 600, 5)),
        ("spacing_mm", "Line spacing (mm)", "scale", (1.0, 20.0, 0.1)),
        ("boundary", "Outer shape", "combo", BOUNDARIES),
        ("aspect", "Height / width", "scale", (0.4, 2.0, 0.05)),
        ("rotation_deg", "Shape rotation (deg)", "scale", (0, 180, 1)),
    ]),
    ("Growth", [
        ("fill", "Fill amount", "scale", (0.2, 1.0, 0.01)),
        ("growth_speed", "Growth speed", "scale", (0.05, 1.0, 0.05)),
        ("branchiness", "Branchiness", "scale", (0.0, 1.0, 0.05)),
        ("smoothness", "Smoothness", "scale", (0.0, 1.0, 0.05)),
        ("wiggle", "Wiggle / randomness", "scale", (0.0, 1.0, 0.05)),
        ("max_iterations", "Max iterations", "scale", (500, 30000, 100)),
    ]),
    ("Start & structure", [
        ("start_shape", "Start shape", "combo", START_SHAPES),
        ("start_size", "Start size", "scale", (0.02, 0.6, 0.01)),
        ("star_arms", "Star arms", "scale", (2, 12, 1)),
        ("hole", "Centre hole", "scale", (0.0, 0.6, 0.01)),
        ("channels", "Channels (arms)", "scale", (0, 12, 1)),
        ("channel_width", "Channel width (x spacing)", "scale", (0.5, 6.0, 0.1)),
        ("channel_length", "Channel length", "scale", (0.1, 1.0, 0.01)),
        ("channel_rotation_deg", "Channel rotation (deg)", "scale", (0, 360, 1)),
    ]),
]
EXPORT_CONTROLS = [
    ("Output", [
        ("smoothing", "Output smoothing", "scale", (0, 6, 1)),
        ("stroke_mm", "Line width (mm)", "scale", (0.1, 5.0, 0.1)),
        ("point_spacing_mm", "Point spacing (mm)", "scale", (0.2, 5.0, 0.1)),
        ("margin_mm", "Page margin (mm)", "scale", (0, 50, 1)),
        ("png_dpi", "PNG DPI", "combo", [150, 300, 600, 1200]),
        ("png_style", "PNG style", "combo", ["line", "filled"]),
    ]),
]

TOOLTIP_DELAY_MS = 2000  # hover this long before the explanation appears
TRACE_SECONDS = 8.0      # how long the line test takes to trace the whole line
TEST_LABEL = "Test: is it one continuous line?"


def _tip(what: str, high: str | None = None, low: str | None = None, note: str | None = None) -> str:
    parts = [what]
    if high:
        parts.append("▲ Higher: " + high)
    if low:
        parts.append("▼ Lower: " + low)
    if note:
        parts.append(note)
    return "\n\n".join(parts)


TIPS = {
    # --- size & shape
    "diameter_mm": _tip(
        "The overall width of the pattern in millimetres (the size of the outer shape).",
        "Bigger pattern with more lines and more fingers. Takes longer to grow.",
        "Smaller pattern with fewer fingers. Grows faster.",
        "The finger width stays the same; that is set by Line spacing."),
    "spacing_mm": _tip(
        "The distance between neighbouring lines. This is the width of every finger and of every gap.",
        "Fat, bold fingers and fewer of them. Simpler look, grows faster.",
        "Thin fingers and many more of them. Finer detail, but slower "
        "(half the spacing = about 4x more work).",
        "For 3D printing keep it well above your wall thickness."),
    "boundary": _tip(
        "The outer shape the pattern grows inside.\n"
        "- circle: round (ignores Height / width)\n"
        "- ellipse: oval, use Height / width to stretch it\n"
        "- triangle / square / hexagon / octagon: straight-sided shapes"),
    "aspect": _tip(
        "Stretches the outer shape. 1.00 = not stretched. Has no effect when Outer shape is 'circle' "
        "(pick 'ellipse' for an oval).",
        "Above 1: the shape gets taller than it is wide.",
        "Below 1: the shape gets wider and flatter."),
    "rotation_deg": _tip(
        "Turns the outer shape around its centre, in degrees. Has no effect on a circle.",
        "Turns the shape further anticlockwise.",
        "Turns it back. Example: a hexagon at 0 has a pointed top, at 30 it has a flat top."),
    # --- growth
    "fill": _tip(
        "How much of the shape gets filled with line before growing stops (1.00 = packed full).",
        "Denser pattern that reaches all the way to the edges. Longer total line.",
        "Stops earlier and leaves open space: around the outside with a circle start, "
        "or in the middle with a ring start."),
    "growth_speed": _tip(
        "How many new points are added to the line on every step.",
        "Finishes faster, but the spacing can get a bit uneven and rougher.",
        "Slower, but calmer, tidier and more evenly spaced."),
    "branchiness": _tip(
        "Where the line prefers to grow: at the tips of the fingers, or everywhere along the line.",
        "Growth happens at the tips, giving long coral fingers that radiate outward and split into branches.",
        "Growth happens everywhere, giving a tight, wavy brain-coral maze."),
    "smoothness": _tip(
        "How strongly the line straightens itself out while it grows.",
        "Round, calm, flowing curves with bigger rounded tips.",
        "Crinkly, busy curves with many small bumps and kinks."),
    "wiggle": _tip(
        "The amount of random jitter added on every step. This is what makes every pattern unique.",
        "Wilder, more irregular and unpredictable shapes (also a bit rougher).",
        "Calmer, more regular and orderly patterns."),
    "max_iterations": _tip(
        "A safety limit on how many growth steps can run.",
        "Big or dense patterns have enough time to finish.",
        "Growth may stop early and leave the pattern unfinished.",
        "Usually leave it alone. Raise it if the status bar says 'iteration limit'."),
    # --- start & structure
    "start_shape": _tip(
        "The shape the line starts from.\n"
        "- circle: a small circle in the centre that grows outward\n"
        "- ring: starts along the outer edge and grows inward, leaving an open glowing centre\n"
        "- star: a star in the centre (see Star arms) that grows outward"),
    "start_size": _tip(
        "The size of the starting circle or star, as a fraction of the pattern's radius. "
        "Not used by the 'ring' start.",
        "A bigger solid blob in the middle before the fingers begin.",
        "Starts from a tiny spot, so the fingers radiate right from the centre."),
    "star_arms": _tip(
        "The number of points on the starting star. Only used when Start shape is 'star'.",
        "More arms, so more main branches radiating from the centre.",
        "Fewer, bigger arms."),
    "hole": _tip(
        "Keeps a round empty area in the middle, e.g. room for the bulb. "
        "Fraction of the pattern's radius, 0 = no hole.",
        "A bigger empty centre with more open light.",
        "A smaller empty centre. 0 turns it off."),
    "channels": _tip(
        "Straight empty gaps running from the centre outward, like the cross in the reference photo. "
        "0 = no channels.",
        "More channels, so the pattern is cut into more slices, like a star or wheel.",
        "Fewer slices (4 makes a cross, 3 a Y shape)."),
    "channel_width": _tip(
        "The width of each channel, measured in line spacings (2.0 = twice the finger width).",
        "Wider, more obvious gaps that let more light through.",
        "Narrow slits."),
    "channel_length": _tip(
        "How far the channels reach from the centre, as a fraction of the radius. "
        "1.00 = all the way to the edge, which splits the pattern into separate sections.",
        "Longer channels, up to cutting right through to the edge.",
        "Short stubs near the centre only.",
        "With a 'ring' start the channels only show where the growth reaches them."),
    "channel_rotation_deg": _tip(
        "Turns all channels (and the star arms) around the centre, in degrees.",
        "Turns them further anticlockwise.",
        "Turns them back. Example with 4 channels: 90 makes a '+', 45 makes an 'x'."),
    # --- output
    "smoothing": _tip(
        "How much the finished line is rounded off for the preview and the exported files. "
        "It does not re-grow the pattern, so you can change it after generating.",
        "Softer, rounder lines like the reference photos (3-4 works well). Very high values slightly "
        "shrink the finger tips.",
        "Follows the raw growth more closely: more detail and small bumps. 0 = raw line."),
    "stroke_mm": _tip(
        "The thickness of the drawn line in SVG, PDF and PNG files. The DXF is always a pure centre line.",
        "A bolder, thicker line.",
        "A thinner, finer line.",
        "For 3D work this usually doesn't matter; set the wall thickness in Blender / Fusion."),
    "point_spacing_mm": _tip(
        "The distance between the points that make up the exported line.",
        "Fewer points: smaller files, and faster to work with in Fusion / Blender, but slightly less precise.",
        "More points: a more precise line, but bigger files. Very low values can make CAD programs slow."),
    "margin_mm": _tip(
        "The empty border around the pattern on the SVG / PDF / PNG page.",
        "More white space around the pattern.",
        "The page hugs the pattern tightly. 0 = no border."),
    "png_dpi": _tip(
        "The resolution of the PNG picture (dots per inch).",
        "A sharper picture, but a larger file.",
        "A smaller file that looks blurrier when zoomed in."),
    "png_style": _tip(
        "How the PNG picture is drawn.\n"
        "- line: just the line on white\n"
        "- filled: the inside of the line is coloured in, so the shape is easier to see"),
}

BUTTON_TIPS = {
    "generate": "Grows a brand-new pattern with a new random seed and the current settings.",
    "regenerate": "Grows the pattern again with the same seed. The same seed and settings always "
                  "give exactly the same pattern.",
    "stop": "Stops growing now. You can still export what has grown so far.",
    "seed": "The pattern's ID number. Note it down to get the same pattern again later, "
            "or type a number and press 'Regenerate same seed'.",
    "lock_seed": "When ticked, 'Generate new pattern' keeps this seed, so you can see how the "
                 "settings change the same pattern.",
    "preset": "Ready-made settings. Picking one loads it and grows a new pattern.",
    "load": "Loads settings from a preset file (.json).",
    "save": "Saves all current settings (and the seed) to a preset file to use again later.",
    "export": "Saves the current pattern to a file in real millimetres.\n"
              "- SVG: for Blender, Illustrator, Inkscape\n"
              "- PDF: for printing\n"
              "- DXF: for Fusion 360 and other CAD programs\n"
              "- PNG: a picture\n"
              "- All: all four files plus the settings, into one folder",
    "preview_style": "Preview only: 'line' shows the line, 'filled' colours in the inside of the line.",
    "show_boundary": "Shows the outer shape as a dotted line in the preview. It is not exported.",
    "test": "Proves the pattern is ONE continuous line.\n\n"
            "1. A pen traces the whole line, starting at the green START dot, without ever lifting, "
            "and ends back at START. The colour changes as it goes, so you can follow the path.\n\n"
            "2. A report then checks: no gaps, closed loop, no crossings, no touching, and that the "
            "exported SVG / PDF / DXF files really contain exactly one line.\n\n"
            "Click again while it traces to skip straight to the result.",
}


def _decimals(step: float) -> int:
    s = f"{step:g}"
    return len(s.split(".")[1]) if "." in s else 0


class Tooltip:
    """Explanation box that appears after the mouse rests on any of `widgets` for TOOLTIP_DELAY_MS."""

    def __init__(self, widgets, text, title=None):
        self.widgets = widgets if isinstance(widgets, (list, tuple)) else [widgets]
        self.text, self.title = text, title
        self.tip = None
        self.job = None
        for w in self.widgets:
            w.bind("<Enter>", self._schedule, add="+")
            w.bind("<Leave>", self.hide, add="+")
            w.bind("<ButtonPress>", self.hide, add="+")

    def _schedule(self, event):
        self._cancel()
        self._pos = (event.x_root, event.y_root)
        # schedule and cancel on the same widget, or Tk complains when the window closes
        self.job = self.widgets[0].after(TOOLTIP_DELAY_MS, self.show)

    def _cancel(self):
        if self.job is not None:
            self.widgets[0].after_cancel(self.job)
            self.job = None

    def show(self):
        self.job = None
        if self.tip is not None:
            return
        root = self.widgets[0]
        self.tip = tk.Toplevel(root)
        self.tip.wm_overrideredirect(True)
        self.tip.attributes("-topmost", True)
        frame = tk.Frame(self.tip, background="#fffbe6", highlightbackground="#b09a50",
                         highlightthickness=1, padx=8, pady=6)
        frame.pack()
        if self.title:
            tk.Label(frame, text=self.title, background="#fffbe6", font=("Segoe UI", 10, "bold"),
                     justify="left", anchor="w").pack(fill="x")
        tk.Label(frame, text=self.text, background="#fffbe6", font=("Segoe UI", 9),
                 wraplength=340, justify="left", anchor="w").pack(fill="x")
        # place next to the cursor, but keep it on screen
        self.tip.update_idletasks()
        w, h = self.tip.winfo_reqwidth(), self.tip.winfo_reqheight()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        x, y = self._pos[0] + 16, self._pos[1] + 18
        if x + w > sw:
            x = max(0, sw - w - 4)
        if y + h > sh - 40:
            y = max(0, self._pos[1] - h - 10)
        self.tip.wm_geometry(f"+{x}+{y}")

    def hide(self, _event=None):
        self._cancel()
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1380x900")
        self.minsize(1000, 650)

        self.vars: dict[str, tk.Variable] = {}
        self.steps: dict[str, float] = {}
        self.seed_var = tk.StringVar(value="1")
        self.lock_seed = tk.BooleanVar(value=False)
        self.show_boundary = tk.BooleanVar(value=True)
        self.preview_style = tk.StringVar(value="line")
        self.status = tk.StringVar(value="Press 'Generate new pattern' to start.   Tip: rest the mouse on "
                                         "any setting or button for 2 seconds to see what it does.")
        self.preset_var = tk.StringVar()

        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.run_id = 0
        self.snapshot = None       # (run_id, points_mm, stats) from the worker
        self.drawn_snapshot = None
        self.sim: DifferentialGrowth | None = None
        self.result_mm: np.ndarray | None = None
        self.boundary_mm: np.ndarray | None = None
        self._redraw_job = None
        self.tooltips: list[Tooltip] = []
        self._view = None          # (cx, cy, scale) of the last preview drawing
        self._trace = None         # state of the running line-test animation
        self._trace_job = None
        self._report = None

        self._build_ui()
        self._load_values(GrowthParams(), ExportSettings())
        self.after(60, self._poll)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=8)

        left = ttk.Frame(self, padding=(8, 8, 4, 8))
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=(4, 8, 8, 8))
        right.pack(side="left", fill="both", expand=True)

        # --- top actions
        tip = self._add_tip
        b = ttk.Button(left, text="Generate new pattern", style="Big.TButton", command=self.generate_new)
        b.pack(fill="x")
        tip(b, BUTTON_TIPS["generate"])
        row = ttk.Frame(left)
        row.pack(fill="x", pady=(6, 0))
        b = ttk.Button(row, text="Regenerate same seed", command=self.regenerate)
        b.pack(side="left", expand=True, fill="x")
        tip(b, BUTTON_TIPS["regenerate"])
        b = ttk.Button(row, text="Stop", command=self.stop)
        b.pack(side="left", padx=(6, 0))
        tip(b, BUTTON_TIPS["stop"])
        row = ttk.Frame(left)
        row.pack(fill="x", pady=(6, 0))
        lab = ttk.Label(row, text="Seed")
        lab.pack(side="left")
        ent = ttk.Entry(row, textvariable=self.seed_var, width=10)
        ent.pack(side="left", padx=4)
        tip([lab, ent], BUTTON_TIPS["seed"], "Seed")
        cb = ttk.Checkbutton(row, text="Keep this seed", variable=self.lock_seed)
        cb.pack(side="left")
        tip(cb, BUTTON_TIPS["lock_seed"], "Keep this seed")

        row = ttk.Frame(left)
        row.pack(fill="x", pady=(6, 0))
        lab = ttk.Label(row, text="Preset")
        lab.pack(side="left")
        self.preset_box = ttk.Combobox(row, textvariable=self.preset_var, state="readonly", width=22,
                                       values=self._builtin_presets())
        self.preset_box.pack(side="left", padx=4)
        self.preset_box.bind("<<ComboboxSelected>>", lambda e: self._apply_builtin_preset())
        tip([lab, self.preset_box], BUTTON_TIPS["preset"], "Preset")
        b = ttk.Button(row, text="Load...", width=7, command=self.load_preset)
        b.pack(side="left")
        tip(b, BUTTON_TIPS["load"])
        b = ttk.Button(row, text="Save...", width=7, command=self.save_preset)
        b.pack(side="left", padx=(4, 0))
        tip(b, BUTTON_TIPS["save"])

        # --- scrollable settings
        holder = ttk.Frame(left)
        holder.pack(fill="both", expand=True, pady=(8, 0))
        canvas = tk.Canvas(holder, width=390, highlightthickness=0)
        sb = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._settings_canvas = canvas
        self.bind_all("<MouseWheel>", self._on_wheel)

        for title, controls in GROWTH_CONTROLS + EXPORT_CONTROLS:
            self._build_group(inner, title, controls)
        for key, _, _, _ in EXPORT_CONTROLS[0][1]:
            self.vars[key].trace_add("write", lambda *a: self._schedule_redraw())

        # --- continuity test
        self.test_btn = ttk.Button(left, text=TEST_LABEL, command=self.run_test)
        self.test_btn.pack(fill="x", pady=(8, 0))
        tip(self.test_btn, BUTTON_TIPS["test"], "Line test")

        # --- export buttons
        box = ttk.LabelFrame(left, text="Export", padding=6)
        box.pack(fill="x", pady=(8, 0))
        export_buttons = []
        for fmt in FORMATS:
            b = ttk.Button(box, text=fmt.upper(), width=6, command=lambda f=fmt: self.export_one(f))
            b.pack(side="left", padx=2)
            export_buttons.append(b)
        b = ttk.Button(box, text="All...", width=6, command=self.export_all)
        b.pack(side="left", padx=2)
        export_buttons.append(b)
        tip(export_buttons, BUTTON_TIPS["export"], "Export")

        # --- preview
        bar = ttk.Frame(right)
        bar.pack(fill="x")
        preview_widgets = [ttk.Label(bar, text="Preview:")]
        preview_widgets[0].pack(side="left")
        for val in ("line", "filled"):
            rb = ttk.Radiobutton(bar, text=val, value=val, variable=self.preview_style,
                                 command=self._schedule_redraw)
            rb.pack(side="left", padx=4)
            preview_widgets.append(rb)
        tip(preview_widgets, BUTTON_TIPS["preview_style"], "Preview")
        cb = ttk.Checkbutton(bar, text="Show outer shape", variable=self.show_boundary,
                             command=self._schedule_redraw)
        cb.pack(side="left", padx=12)
        tip(cb, BUTTON_TIPS["show_boundary"], "Show outer shape")
        self.progress = ttk.Progressbar(bar, length=220, maximum=1.0)
        self.progress.pack(side="right")

        self.canvas = tk.Canvas(right, background="#ffffff", highlightthickness=1,
                                highlightbackground="#cccccc")
        self.canvas.pack(fill="both", expand=True, pady=6)
        self.canvas.bind("<Configure>", lambda e: self._schedule_redraw())
        ttk.Label(right, textvariable=self.status, anchor="w").pack(fill="x")

    def _build_group(self, parent, title, controls):
        box = ttk.LabelFrame(parent, text=title, padding=(6, 4))
        box.pack(fill="x", pady=4, padx=(0, 4))
        box.columnconfigure(1, weight=1)
        for r, (key, label, kind, opts) in enumerate(controls):
            lab = ttk.Label(box, text=label, width=24)
            lab.grid(row=r, column=0, sticky="w", pady=1)
            row_widgets = [lab]
            if kind == "combo":
                var = tk.StringVar()
                combo = ttk.Combobox(box, textvariable=var, values=opts, state="readonly", width=12)
                combo.grid(row=r, column=1, columnspan=2, sticky="w")
                row_widgets.append(combo)
            else:
                lo, hi, step = opts
                var = tk.StringVar()
                dec = _decimals(step)
                self.steps[key] = step

                busy = [False]  # stops slider <-> text box feedback loops

                def on_slide(v, var=var, step=step, dec=dec, busy=busy):
                    if busy[0]:
                        return
                    x = round(round(float(v) / step) * step, dec)
                    text = f"{x:.{dec}f}"
                    if var.get() != text:
                        var.set(text)

                sc = ttk.Scale(box, from_=lo, to=hi, orient="horizontal", length=150, command=on_slide)
                sc.grid(row=r, column=1, sticky="ew", padx=4)
                spin = ttk.Spinbox(box, textvariable=var, from_=lo, to=hi, increment=step, width=7)
                spin.grid(row=r, column=2, sticky="e")
                row_widgets += [sc, spin]

                def sync(*_a, var=var, sc=sc, busy=busy):
                    busy[0] = True
                    try:
                        sc.set(float(var.get()))
                    except (ValueError, tk.TclError):
                        pass
                    finally:
                        busy[0] = False

                var.trace_add("write", sync)
            self.vars[key] = var
            if key in TIPS:
                self._add_tip(row_widgets, TIPS[key], label)

    def _add_tip(self, widgets, text, title=None):
        self.tooltips.append(Tooltip(widgets, text, title))

    def _on_wheel(self, event):
        w = self.winfo_containing(event.x_root, event.y_root)
        while w is not None:
            if w is self._settings_canvas or str(w).startswith(str(self._settings_canvas)):
                self._settings_canvas.yview_scroll(int(-event.delta / 120), "units")
                return
            w = w.master

    # ------------------------------------------------------------ settings
    def _load_values(self, gp: GrowthParams, es: ExportSettings):
        for src in (gp.to_dict(), es.to_dict()):
            for k, v in src.items():
                if k in self.vars:
                    if k in self.steps:
                        dec = _decimals(self.steps[k])
                        self.vars[k].set(f"{float(v):.{dec}f}")
                    else:
                        self.vars[k].set(str(v))
        self.seed_var.set(str(gp.seed))

    def _read(self, cls):
        base = cls().to_dict()
        vals = {}
        for k, default in base.items():
            if k not in self.vars:
                continue
            raw = self.vars[k].get()
            try:
                vals[k] = type(default)(float(raw)) if isinstance(default, (int, float)) else type(default)(raw)
            except (ValueError, TypeError):
                vals[k] = default
        return cls.from_dict(vals)

    def growth_params(self) -> GrowthParams:
        gp = self._read(GrowthParams)
        try:
            gp.seed = int(self.seed_var.get())
        except ValueError:
            gp.seed = 0
        return gp

    def export_settings(self) -> ExportSettings:
        return self._read(ExportSettings)

    def _builtin_presets(self):
        if not os.path.isdir(PRESET_DIR):
            return []
        return sorted(f[:-5] for f in os.listdir(PRESET_DIR) if f.endswith(".json"))

    def _apply_preset_file(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        gp = GrowthParams.from_dict(data.get("growth", {}))
        es = ExportSettings.from_dict(data.get("export", {}))
        seed = self.seed_var.get()
        self._load_values(gp, es)
        if "seed" not in data.get("growth", {}):
            self.seed_var.set(seed)

    def _apply_builtin_preset(self):
        self._apply_preset_file(os.path.join(PRESET_DIR, self.preset_var.get() + ".json"))
        self.generate_new()

    def load_preset(self):
        path = filedialog.askopenfilename(title="Load preset", initialdir=PRESET_DIR,
                                          filetypes=[("Preset", "*.json")])
        if path:
            self._apply_preset_file(path)

    def save_preset(self):
        path = filedialog.asksaveasfilename(title="Save preset", defaultextension=".json",
                                            filetypes=[("Preset", "*.json")])
        if path:
            data = {"growth": self.growth_params().to_dict(), "export": self.export_settings().to_dict()}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.status.set(f"Preset saved: {path}")

    # ---------------------------------------------------------- simulation
    def generate_new(self):
        if not self.lock_seed.get():
            self.seed_var.set(str(random.randint(1, 999_999)))
        self._start()

    def regenerate(self):
        self._start()

    def stop(self):
        self.stop_event.set()

    def _start(self):
        self._cancel_trace()
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        gp = self.growth_params()
        try:
            sim = DifferentialGrowth(gp)
        except Exception as exc:  # bad combination of settings
            messagebox.showerror(APP_TITLE, f"Could not start with these settings:\n{exc}")
            return
        self.run_id += 1
        self.stop_event = threading.Event()
        self.sim = sim
        self.result_mm = None
        self.boundary_mm = sim.boundary_mm()
        self.status.set(f"Growing... seed {gp.seed}")
        self.thread = threading.Thread(target=self._worker, args=(sim, self.run_id, self.stop_event), daemon=True)
        self.thread.start()

    def _worker(self, sim: DifferentialGrowth, run_id: int, stop_event: threading.Event):
        def publish(s: DifferentialGrowth):
            stats = {"iteration": s.iteration, "nodes": len(s.P), "progress": s.progress,
                     "done": s.done, "reason": s.stop_reason}
            with self.lock:
                self.snapshot = (run_id, s.points_mm().copy(), stats)
        try:
            sim.run(stop_event=stop_event, callback=publish, callback_every=4)
        except Exception as exc:
            with self.lock:
                self.snapshot = (run_id, sim.points_mm().copy(),
                                 {"done": True, "reason": f"error: {exc}", "iteration": sim.iteration,
                                  "nodes": len(sim.P), "progress": 0})

    def _poll(self):
        with self.lock:
            snap = self.snapshot
        if snap is not None and snap is not self.drawn_snapshot and snap[0] == self.run_id:
            self.drawn_snapshot = snap
            _, pts, st = snap
            self.progress["value"] = st["progress"]
            if st["done"]:
                self._finish(pts, st)
            else:
                self.status.set(f"Growing... seed {self.seed_var.get()}  |  step {st['iteration']:,}  |  "
                                f"{st['nodes']:,} points  |  {st['progress'] * 100:.0f}%")
                self._draw(pts, final=False)
        self.after(60, self._poll)

    def _finish(self, pts, st):
        self.result_mm = pts
        crossings = count_self_intersections(pts)
        length_m = closed_length(smooth_points(pts, self.export_settings())) / 1000
        check = "one closed line, no crossings" if crossings == 0 else f"WARNING: {crossings} crossings"
        self.status.set(f"Done ({st['reason']})  |  seed {self.seed_var.get()}  |  {st['nodes']:,} points  |  "
                        f"line length {length_m:.2f} m  |  {check}")
        self._draw(pts, final=True)

    # ------------------------------------------------------------- drawing
    def _schedule_redraw(self):
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(80, self._redraw)

    def _redraw(self):
        self._redraw_job = None
        if self.result_mm is not None:
            self._draw(self.result_mm, final=True)
        elif self.drawn_snapshot is not None:
            self._draw(self.drawn_snapshot[1], final=False)

    def _draw(self, pts_mm: np.ndarray, final: bool):
        self._cancel_trace()
        c = self.canvas
        c.delete("all")
        W, H = c.winfo_width(), c.winfo_height()
        if W < 10 or H < 10 or self.boundary_mm is None:
            return
        B = self.boundary_mm
        ext = max(np.abs(B).max(), np.abs(pts_mm).max()) * 1.04
        s = min(W, H) / (2 * ext)
        cx, cy = W / 2, H / 2
        self._view = (cx, cy, s)

        def coords(Q):
            out = np.empty(2 * len(Q))
            out[0::2] = cx + Q[:, 0] * s
            out[1::2] = cy - Q[:, 1] * s
            return out.tolist()

        if self.show_boundary.get():
            c.create_polygon(coords(B), outline="#b8c4e0", fill="", dash=(4, 4))
        if final:
            try:
                pts_mm = smooth_points(pts_mm, self.export_settings())
            except Exception:
                pass
        xy = coords(pts_mm)
        if self.preview_style.get() == "filled":
            c.create_polygon(xy, fill="#f2c46d", outline="#3a2a10", width=1.5, tags="pattern")
        else:
            c.create_polygon(xy, fill="", outline="#111111", width=1.5, tags="pattern")

    # ------------------------------------------------------------ line test
    def run_test(self):
        if self._trace is not None:  # clicked again while tracing: jump to the result
            self._trace["skip"] = True
            return
        if not self._need_result():
            return
        es = self.export_settings()
        gp = self.sim.p if self.sim is not None else self.growth_params()
        self.status.set("Testing the line...")
        self.update_idletasks()
        try:
            P, checks = run_line_test(self.result_mm, es, gp.spacing_mm)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"The line test could not run:\n{exc}")
            return
        self._draw(self.result_mm, final=True)
        if self._view is None:
            self._show_report(checks, gp.seed)
            return
        c = self.canvas
        c.itemconfigure("pattern", outline="#d6d6d6")
        cx, cy, s = self._view
        xy = np.column_stack([cx + P[:, 0] * s, cy - P[:, 1] * s])
        xy = np.vstack([xy, xy[:1]])  # the pen finishes back on the start point
        x0, y0 = xy[0]
        c.create_oval(x0 - 7, y0 - 7, x0 + 7, y0 + 7, fill="#22aa44", outline="white", width=2, tags="start")
        label_bg = c.create_rectangle(0, 0, 0, 0, fill="white", outline="#1a7f34", tags="start")
        label = c.create_text(x0, y0 - 19, text="START", fill="#1a7f34", font=("Segoe UI", 10, "bold"),
                              tags="start")
        self._fit_label_bg(label_bg, label)
        pen = c.create_oval(x0 - 5, y0 - 5, x0 + 5, y0 + 5, fill="#e0312f", outline="white", width=1.5)
        self._trace = {"xy": xy, "i": 0, "t0": time.time(), "pen": pen, "label": label, "label_bg": label_bg,
                       "checks": checks, "seed": gp.seed, "skip": False}
        self.test_btn.configure(text="Skip to result")
        self._trace_step()

    def _trace_step(self):
        tr = self._trace
        if tr is None:
            return
        c = self.canvas
        xy = tr["xy"]
        n = len(xy) - 1
        frac = 1.0 if tr["skip"] else min(1.0, (time.time() - tr["t0"]) / TRACE_SECONDS)
        target = int(round(frac * n))
        if target > tr["i"]:
            band = max(1, n // 400)  # colour changes every `band` points
            for a in range(tr["i"], target, band):
                b = min(a + band, target)
                r, g, bl = colorsys.hsv_to_rgb(0.8 * a / n, 0.85, 0.85)
                c.create_line(xy[a:b + 1].ravel().tolist(), width=2.5, capstyle="round", joinstyle="round",
                              fill=f"#{int(r * 255):02x}{int(g * 255):02x}{int(bl * 255):02x}")
            tr["i"] = target
            x, y = xy[target]
            c.coords(tr["pen"], x - 5, y - 5, x + 5, y + 5)
            c.tag_raise("start")
            c.tag_raise(tr["pen"])
        self.status.set(f"Tracing the whole line in one stroke... {frac * 100:.0f}%   (the pen never lifts)")
        if frac >= 1.0:
            self._finish_trace()
        else:
            self._trace_job = self.after(16, self._trace_step)

    def _finish_trace(self):
        tr = self._trace
        self._trace, self._trace_job = None, None
        self.test_btn.configure(text=TEST_LABEL)
        ok = all(ch.passed for ch in tr["checks"])
        self.canvas.itemconfigure(tr["label"], text="START = END" if ok else "START")
        self._fit_label_bg(tr["label_bg"], tr["label"])
        self.status.set("Line test PASSED: the pattern is one continuous closed line." if ok
                        else "Line test NOT PASSED: see the report.")
        self._show_report(tr["checks"], tr["seed"])

    def _fit_label_bg(self, bg, label):
        x1, y1, x2, y2 = self.canvas.bbox(label)
        self.canvas.coords(bg, x1 - 4, y1 - 1, x2 + 4, y2 + 1)

    def _cancel_trace(self):
        if self._trace_job is not None:
            self.after_cancel(self._trace_job)
            self._trace_job = None
        if self._trace is not None:
            self._trace = None
            self.test_btn.configure(text=TEST_LABEL)

    def _show_report(self, checks, seed):
        if self._report is not None and self._report.winfo_exists():
            self._report.destroy()
        ok = all(ch.passed for ch in checks)
        green, red = "#1a7f34", "#c62828"
        win = self._report = tk.Toplevel(self)
        win.title("Line test report")
        win.transient(self)
        win.resizable(False, False)
        body = ttk.Frame(win, padding=16)
        body.pack(fill="both", expand=True)
        head = ("\u2714  PASSED: this pattern is ONE continuous closed line" if ok
                else "\u2718  NOT PASSED: see the red items below")
        ttk.Label(body, text=head, foreground=green if ok else red,
                  font=("Segoe UI", 13, "bold")).pack(anchor="w")
        ttk.Label(body, text=f"Seed {seed}. Checked on exactly the line that gets exported "
                             f"(current output settings).", foreground="#555555").pack(anchor="w", pady=(2, 10))
        for ch in checks:
            row = ttk.Frame(body)
            row.pack(fill="x", pady=5)
            ttk.Label(row, text="\u2714" if ch.passed else "\u2718", width=3,
                      foreground=green if ch.passed else red, font=("Segoe UI", 14, "bold")).pack(side="left", anchor="n")
            col = ttk.Frame(row)
            col.pack(side="left", fill="x", expand=True)
            ttk.Label(col, text=("PASS: " if ch.passed else "FAIL: ") + ch.title,
                      font=("Segoe UI", 10, "bold")).pack(anchor="w")
            ttk.Label(col, text=ch.detail, wraplength=520, justify="left").pack(anchor="w")
        ttk.Button(body, text="Close", command=win.destroy).pack(anchor="e", pady=(12, 0))
        win.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - win.winfo_reqwidth()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - win.winfo_reqheight()) // 2
        win.geometry(f"+{max(0, x)}+{max(0, y)}")

    # -------------------------------------------------------------- export
    def _need_result(self) -> bool:
        if self.result_mm is None:
            if self.drawn_snapshot is not None and self.drawn_snapshot[0] == self.run_id:
                if messagebox.askyesno(APP_TITLE, "The pattern is still growing. Stop and export what's there now?"):
                    self.stop()
                    if self.thread is not None:
                        self.thread.join(timeout=5)
                    self._poll_once()
                    return self.result_mm is not None
            else:
                messagebox.showinfo(APP_TITLE, "Generate a pattern first.")
            return False
        return True

    def _poll_once(self):
        with self.lock:
            snap = self.snapshot
        if snap is not None and snap[0] == self.run_id:
            self.drawn_snapshot = snap
            self._finish(snap[1], snap[2])

    def _default_name(self):
        return f"coral_seed{self.seed_var.get()}"

    def export_one(self, fmt: str):
        if not self._need_result():
            return
        path = filedialog.asksaveasfilename(title=f"Export {fmt.upper()}", defaultextension=f".{fmt}",
                                            initialfile=f"{self._default_name()}.{fmt}",
                                            filetypes=[(fmt.upper(), f"*.{fmt}")])
        if not path:
            return
        self._do_export([(fmt, path)])

    def export_all(self):
        if not self._need_result():
            return
        folder = filedialog.askdirectory(title="Export all formats to folder")
        if not folder:
            return
        name = self._default_name()
        jobs = [(fmt, os.path.join(folder, f"{name}.{fmt}")) for fmt in FORMATS]
        self._do_export(jobs)
        with open(os.path.join(folder, f"{name}_settings.json"), "w", encoding="utf-8") as f:
            json.dump({"growth": self.growth_params().to_dict(), "export": self.export_settings().to_dict()},
                      f, indent=2)

    def _do_export(self, jobs):
        es = self.export_settings()
        try:
            for fmt, path in jobs:
                export(self.result_mm, path, fmt, es)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Export failed:\n{exc}")
            return
        where = jobs[0][1] if len(jobs) == 1 else os.path.dirname(jobs[0][1])
        self.status.set(f"Exported {', '.join(f.upper() for f, _ in jobs)} -> {where}")


if __name__ == "__main__":
    App().mainloop()
