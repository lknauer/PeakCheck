"""
The Tkinter graphical interface (`PeakCheckGUI`).

A single window with three control bars, an embedded matplotlib plot, four
parameter sliders and a status line at the bottom. The user opens a folder of
data files, picks one as the reference and any number as components, tunes
the search interactively (sliders + mouse-add/remove on the plot) and clicks
`Fit & Save` to dispatch to `run_analysis`. All run state lives on the
shared Config — what the user does here is functionally identical to what a
TOML-driven headless run does.
"""
from __future__ import annotations

import os

import numpy as np

from .background import subtract_background, x_range_mask
from .config import (Config, X_CONVERSIONS, apply_conversion, load_config,
                     write_template, _coerce_nullable)
from .fit import search_peaks
from .io import load_xy
from .pipeline import find_component_files, run_analysis
from .plots import _xlabel


# --------------------------------------------------------------------------
# Tooltip helper for entry fields, sliders and buttons
# --------------------------------------------------------------------------

class _Tooltip:
    """Lightweight hover tooltip for any Tk widget.

    On <Enter> a small Toplevel pops up below the widget with the help text;
    on <Leave> it is destroyed. No external dependency.
    """

    def __init__(self, widget, text, delay_ms=400, wraplength=320):
        import tkinter as tk
        self.tk = tk
        self.widget = widget
        self.text = text
        self.delay = delay_ms
        self.wraplength = wraplength
        self._after_id = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _ev=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self._tip or not self.widget.winfo_exists():
            return
        x = self.widget.winfo_rootx() + 8
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        tw = self.tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = self.tk.Label(
            tw, text=self.text, justify="left",
            background="#ffffe0", relief="solid", borderwidth=1,
            wraplength=self.wraplength, padx=6, pady=4,
            font=("TkDefaultFont", 8),
        )
        lbl.pack()
        self._tip = tw

    def _hide(self, _ev=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


# Tooltip texts for the fields, sliders and main buttons in the GUI.
# Kept here (rather than inline) so the documentation lives in one place and
# stays in sync with the docs.
_TOOLTIPS = {
    # first bar
    "open_folder":  "Open a folder and pick the reference + components from a list. "
                    "Shortcut: Ctrl+O.",
    "output_folder":"Choose where plots, Excel and CSV are written. "
                    "Empty = next to the reference file.",
    "load_cfg":     "Load a TOML configuration file into the GUI. Shortcut: Ctrl+L.",
    "save_cfg":     "Save the current GUI settings as a TOML file (for reproducible reruns). "
                    "Shortcut: Ctrl+S.",
    # parameter fields
    "profile":      "Line profile: voigt (true Gaussian (x) Lorentzian convolution), "
                    "gauss or lorentz.",
    "wg":           "Gaussian FWHM in x-units — the instrumental resolution. "
                    "For NIS at BL09XU (SPring-8) ~0.8 meV. Kept fixed during the fit.",
    "wl":           "Lorentzian FWHM in x-units — natural linewidth plus thermal "
                    "contributions. Kept fixed during the fit.",
    "x_min":        "Lower bound of the analysis window in x-units. Empty = open. "
                    "Pressing Enter zooms the plot to the window.",
    "x_max":        "Upper bound of the analysis window in x-units. Empty = open.",
    "tolerance":    "+/- position window (in x-units) within which a component peak "
                    "counts as the same peak as the reference.",
    "snr_thresh":   "Signal/error threshold above which a peak in a component is "
                    "counted as present. Raise for noisy spectra.",
    "x_label":      "x-axis label used in plots and result files.",
    "x_unit":       "x-axis unit; shown in parentheses behind the label.",
    "y_label":      "y-axis label used in plots and result files.",
    "x_conversion": "Named axis preset (e.g. meV -> cm^-1). Sets scale, offset, "
                    "label and unit at once.",
    "x_scale":      "Manual scale factor (x' = x*scale + offset). Ignored when "
                    "an x_conversion preset is active.",
    "x_offset":     "Manual additive offset.",
    # sliders
    "prominence":   "Minimum peak prominence (scipy.signal.find_peaks). Higher = "
                    "fewer false positives but missed weak peaks.",
    "distance":     "Minimum distance between two detected peaks, in points.",
    "min_snr":      "Minimum signal/error ratio for the auto-search (corrected signal).",
    "bg_window":    "Window for the rolling-minimum background, in points. "
                    "0 disables the baseline subtraction.",
    # baseline method + optional position refinement
    "baseline_method": "Baseline estimator used by the fit, the statistics and "
                       "the presence check: rolling_min (default), polynomial "
                       "or als (asymmetric least squares). The live preview "
                       "always uses the rolling minimum.",
    "baseline_poly_order": "Polynomial order for the 'polynomial' baseline.",
    "baseline_als_lambda": "ALS smoothness (larger = stiffer baseline).",
    "baseline_als_p":      "ALS asymmetry (smaller = baseline hugs the lower envelope).",
    "refine":       "Refine the peak positions after the fit by a small bounded "
                    "step (amplitudes stay non-negative). Off by default; the "
                    "result is unchanged unless enabled.",
    "refine_window":"Maximum position shift (in x-units) allowed when refining.",
    "refine_widths":"Also refine a per-peak width after the fit (bounded; amplitudes "
                    "stay non-negative). Off by default. Note: extra free widths "
                    "lower the reduced chi-square mechanically.",
    "width_mode":   "What broadens: 'fwhm' scales the whole Voigt (default), "
                    "'sigma' scales only the Gaussian (inhomogeneous) part.",
    "width_min_factor": "Lower width bound as a factor of the instrumental width "
                        "(1.0 = never narrower than the instrument).",
    "width_max_factor": "Upper width bound as a factor of the instrumental width "
                        "(e.g. 2.0 = up to twice the instrumental width).",
    # bottom buttons
    "clear_peaks":  "Remove all peaks from the plot.",
    "auto_search":  "Run the automatic peak search with the current sliders. Shortcut: Ctrl+R.",
    "fit_save":     "Fit, run the presence check on every component, and write "
                    "plots, Excel and CSV. Shortcut: F5.",
}


class PeakCheckGUI:
    """
    Tkinter front end. Mirrors the original interactive picker:
      * sliders for prominence / min distance / min SNR / background window
      * LEFT click on the plot  : add a peak at the nearest data point
      * RIGHT click on the plot : remove the nearest peak marker
    plus a file dialog, parameter fields, TOML load/save and a 'Fit & Save'
    button that produces the same plots, Excel workbook and ASCII CSV files as
    the headless run.
    """

    _MARKER_OFFSET_FRAC = 0.06

    def __init__(self, cfg=None):
        import tkinter as tk
        from tkinter import ttk
        import matplotlib
        matplotlib.use("TkAgg", force=True)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)

        self.tk = tk
        self.cfg = cfg or Config()
        self.x = self.y = self.yerr = None
        self.had_errors = False
        self.peaks = []                 # list of integer indices into self.x
        self._mask = None
        self._yrange = 1.0
        self.selected_components = None  # list of paths chosen via "Open folder…",
                                         # or None to fall back to the glob pattern

        self.root = tk.Tk()
        self.root.title("PeakCheck")
        self.root.geometry("1180x860")

        # subtle, professional colour accents via ttk styles
        self._accent = "#2c6fb5"          # muted blue
        try:
            style = ttk.Style(self.root)
            # group headings: small, accent-coloured, slightly bold
            style.configure("Group.TLabelframe.Label",
                            foreground=self._accent,
                            font=("TkDefaultFont", 9, "bold"))
            style.configure("Group.TLabelframe", borderwidth=1, relief="groove")
            # primary action button (Fit & Save) gets a touch of accent text
            style.configure("Accent.TButton", font=("TkDefaultFont", 9, "bold"))
            # hint labels in a quiet grey
            style.configure("Hint.TLabel", foreground="#777")
            style.configure("Group.TLabel", foreground=self._accent)
        except Exception:
            pass

        # ---- controls: grouped into labelled frames -------------------
        # A horizontal container holds the first groups side by side; the
        # remaining groups stack below. Each group is a ttk.LabelFrame.
        self.vars = {}
        self._field_widgets = {}        # key -> (label, entry) for enable/disable

        def add_field(parent, label, key, width=7, on_return=None, tip_key=None):
            lbl = ttk.Label(parent, text=label)
            lbl.pack(side=tk.LEFT, padx=(8, 2))
            v = tk.StringVar(value=self._cfg_str(key))
            e = ttk.Entry(parent, textvariable=v, width=width)
            e.pack(side=tk.LEFT)
            cb = on_return or (lambda _ev: self._pull_fields_and_redraw())
            e.bind("<Return>", cb)
            self.vars[key] = v
            self._field_widgets[key] = (lbl, e)
            tip = _TOOLTIPS.get(tip_key or key)
            if tip:
                _Tooltip(lbl, tip)
                _Tooltip(e, tip)
            return v

        def group(title):
            f = ttk.LabelFrame(self.root, text=title, padding=(8, 4, 8, 6),
                               style="Group.TLabelframe")
            f.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(4, 0))
            return f

        # --- group: Files & configuration ------------------------------
        g_files = group("Files & configuration")
        b_open = ttk.Button(g_files, text="Open folder...",
                            command=self.on_open_folder)
        b_open.pack(side=tk.LEFT)
        b_out = ttk.Button(g_files, text="Output folder...",
                           command=self.on_pick_output_folder)
        b_out.pack(side=tk.LEFT, padx=3)
        self.out_label = ttk.Label(g_files, text="", style="Hint.TLabel",
                                   font=("TkDefaultFont", 8))
        self.out_label.pack(side=tk.LEFT, padx=(0, 6))
        b_load = ttk.Button(g_files, text="Load config", command=self.on_load_cfg)
        b_load.pack(side=tk.LEFT, padx=3)
        b_save = ttk.Button(g_files, text="Save config", command=self.on_save_cfg)
        b_save.pack(side=tk.LEFT)
        _Tooltip(b_open,  _TOOLTIPS["open_folder"])
        _Tooltip(b_out,   _TOOLTIPS["output_folder"])
        _Tooltip(b_load,  _TOOLTIPS["load_cfg"])
        _Tooltip(b_save,  _TOOLTIPS["save_cfg"])

        # --- group: Profile & widths -----------------------------------
        g_prof = group("Profile & widths")
        prof_lbl = ttk.Label(g_prof, text="Profile")
        prof_lbl.pack(side=tk.LEFT, padx=(2, 2))
        self.profile_var = tk.StringVar(value=str(self.cfg.profile))
        prof = ttk.Combobox(g_prof, textvariable=self.profile_var, width=8,
                            values=["voigt", "gauss", "lorentz"], state="readonly")
        prof.pack(side=tk.LEFT)
        prof.bind("<<ComboboxSelected>>", lambda _ev: self._on_profile_change())
        _Tooltip(prof_lbl, _TOOLTIPS["profile"])
        _Tooltip(prof,     _TOOLTIPS["profile"])
        self._wg_field = add_field(g_prof, "WG (Gauss)", "wg")
        self._wl_field = add_field(g_prof, "WL (Lorentz)", "wl")
        self._profile_combo = prof
        # read-only effective FWHM (not an input; updates with WG/WL/profile)
        ttk.Label(g_prof, text="FWHM").pack(side=tk.LEFT, padx=(12, 2))
        self.fwhm_var = tk.StringVar(value="")
        fwhm_lbl = ttk.Label(g_prof, textvariable=self.fwhm_var, width=8,
                             style="Group.TLabel")
        fwhm_lbl.pack(side=tk.LEFT)
        _Tooltip(fwhm_lbl, "Effective full width at half maximum of the chosen "
                           "profile (computed from WG/WL; not editable).")
        self._update_fwhm_label()

        # --- group: x-axis (window, labels, conversion) ----------------
        g_x = group("x-axis: window, labels & conversion")
        add_field(g_x, "x_min", "x_min", width=6)
        add_field(g_x, "x_max", "x_max", width=6)
        add_field(g_x, "x label", "x_label", width=11)
        add_field(g_x, "x unit", "x_unit", width=7)
        add_field(g_x, "y label", "y_label", width=11)
        # conversion on its own line within the same group
        g_xc = ttk.Frame(g_x)
        g_xc.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        conv_lbl = ttk.Label(g_xc, text="x conversion")
        conv_lbl.pack(side=tk.LEFT, padx=(0, 2))
        self.conv_var = tk.StringVar(value=self._match_conversion())
        _affine = [k for k, v in X_CONVERSIONS.items() if v[2] == "affine"]
        conv = ttk.Combobox(g_xc, textvariable=self.conv_var, width=18,
                            values=_affine + ["custom"], state="readonly")
        conv.pack(side=tk.LEFT)
        conv.bind("<<ComboboxSelected>>", lambda _ev: self._on_conversion_preset())
        _Tooltip(conv_lbl, _TOOLTIPS["x_conversion"])
        _Tooltip(conv,     _TOOLTIPS["x_conversion"])
        add_field(g_xc, "x_scale", "x_scale", width=11,
                  on_return=lambda _ev: self._reload_on_scale_change())
        add_field(g_xc, "x_offset", "x_offset", width=9,
                  on_return=lambda _ev: self._reload_on_scale_change())
        ttk.Label(g_xc, text="(x -> x*scale + offset, applied on load)",
                  style="Hint.TLabel").pack(side=tk.LEFT, padx=(8, 2))

        # --- group: Background & peak presence -------------------------
        g_bg = group("Background & peak presence")
        base_lbl = ttk.Label(g_bg, text="Baseline")
        base_lbl.pack(side=tk.LEFT, padx=(2, 2))
        self.baseline_var = tk.StringVar(value=str(self.cfg.baseline_method))
        base = ttk.Combobox(g_bg, textvariable=self.baseline_var, width=12,
                            values=["rolling_min", "polynomial", "als"],
                            state="readonly")
        base.pack(side=tk.LEFT)
        base.bind("<<ComboboxSelected>>", lambda _ev: self._on_baseline_change())
        _Tooltip(base_lbl, _TOOLTIPS["baseline_method"])
        _Tooltip(base,     _TOOLTIPS["baseline_method"])
        add_field(g_bg, "poly order", "baseline_poly_order", width=5)
        add_field(g_bg, "ALS lambda", "baseline_als_lambda", width=10)
        add_field(g_bg, "ALS p", "baseline_als_p", width=7)
        add_field(g_bg, "tolerance", "tolerance", width=6)
        add_field(g_bg, "SNR thresh", "snr_thresh", width=6)
        self._baseline_combo = base

        # --- group: Refinement (advanced) ------------------------------
        g_ref = group("Refinement (optional)")
        self.refine_var = tk.BooleanVar(value=bool(self.cfg.refine))
        refine_chk = ttk.Checkbutton(g_ref, text="Refine positions",
                                     variable=self.refine_var,
                                     command=self._on_refine_toggle)
        refine_chk.pack(side=tk.LEFT, padx=(2, 2))
        _Tooltip(refine_chk, _TOOLTIPS["refine"])
        add_field(g_ref, "refine window", "refine_window", width=7)
        ttk.Separator(g_ref, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.refine_widths_var = tk.BooleanVar(value=bool(self.cfg.refine_widths))
        rw_chk = ttk.Checkbutton(g_ref, text="Refine widths",
                                 variable=self.refine_widths_var,
                                 command=self._on_refine_toggle)
        rw_chk.pack(side=tk.LEFT, padx=(2, 2))
        _Tooltip(rw_chk, _TOOLTIPS["refine_widths"])
        wm_lbl = ttk.Label(g_ref, text="width mode")
        wm_lbl.pack(side=tk.LEFT, padx=(12, 2))
        self.width_mode_var = tk.StringVar(value=str(self.cfg.width_mode))
        wmode = ttk.Combobox(g_ref, textvariable=self.width_mode_var, width=7,
                             values=["fwhm", "sigma"], state="readonly")
        wmode.pack(side=tk.LEFT)
        wmode.bind("<<ComboboxSelected>>", lambda _ev: self._pull_fields())
        _Tooltip(wm_lbl, _TOOLTIPS["width_mode"])
        _Tooltip(wmode,  _TOOLTIPS["width_mode"])
        add_field(g_ref, "min x", "width_min_factor", width=5)
        add_field(g_ref, "max x", "width_max_factor", width=5)
        self._refine_chk = refine_chk
        self._rw_chk = rw_chk
        self._wmode_combo = wmode

        # Pack order matters: reserve the fixed bottom strips FIRST (from the
        # bottom up), then let the canvas expand into whatever space is left.
        # Otherwise an expanding canvas packed first can push the sliders and
        # the action bar off the bottom of the window.

        # ---- bottom bar (packed first, pinned to the bottom) ----------
        bottom = ttk.Frame(self.root, padding=6)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.status = tk.StringVar(value="Open a data file to begin.")
        ttk.Label(bottom, textvariable=self.status,
                  style="Group.TLabel").pack(side=tk.LEFT)
        b_fit = ttk.Button(bottom, text="Fit & Save", command=self.on_fit,
                           style="Accent.TButton")
        b_fit.pack(side=tk.RIGHT)
        b_auto = ttk.Button(bottom, text="Auto-search peaks", command=self._auto_search)
        b_auto.pack(side=tk.RIGHT, padx=4)
        b_clr = ttk.Button(bottom, text="Clear peaks", command=self._clear_peaks)
        b_clr.pack(side=tk.RIGHT)
        _Tooltip(b_fit,  _TOOLTIPS["fit_save"])
        _Tooltip(b_auto, _TOOLTIPS["auto_search"])
        _Tooltip(b_clr,  _TOOLTIPS["clear_peaks"])

        # ---- sliders (packed next, pinned just above the bottom bar) ---
        sl = ttk.Frame(self.root, padding=6)
        sl.pack(side=tk.BOTTOM, fill=tk.X)
        self.sliders = {}
        self._make_slider(sl, "prominence", "Prominence", 0.1, 1000.0,
                          self.cfg.prominence_init, 0)
        self._make_slider(sl, "distance", "Min dist (pts)", 1, 50,
                          self.cfg.distance_init, 1)
        self._make_slider(sl, "min_snr", "Min SNR", 0.0, 20.0,
                          self.cfg.min_snr_init, 2)
        self._make_slider(sl, "bg_window", "BG window (pts)", 0, 100,
                          self.cfg.bg_window, 3)

        # ---- matplotlib canvas (packed last; expands into remaining space)
        self.fig = Figure(figsize=(11, 5.2))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)  # zoom/pan toolbar
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        # keyboard shortcuts (mirror the buttons; no conflict with field Enter)
        self.root.bind("<Control-o>", lambda _e: self.on_open_folder())
        self.root.bind("<Control-O>", lambda _e: self.on_open_folder())
        self.root.bind("<Control-l>", lambda _e: self.on_load_cfg())
        self.root.bind("<Control-s>", lambda _e: self.on_save_cfg())
        self.root.bind("<Control-r>", lambda _e: self._auto_search())
        self.root.bind("<F5>",        lambda _e: self.on_fit())

        # If the config already points at an existing file, load it.
        if cfg and os.path.isfile(self.cfg.reference_file):
            self._load_reference(self.cfg.reference_file)

        # initial output-folder label
        self._update_out_label()
        # grey out fields that don't apply to the current selections
        self._update_enabled_states()

    # ---- helpers ------------------------------------------------------

    def _set_field_state(self, key, enabled):
        """Enable/disable a label+entry pair created by add_field."""
        pair = getattr(self, "_field_widgets", {}).get(key)
        if not pair:
            return
        state = ("!disabled",) if enabled else ("disabled",)
        for w in pair:
            try:
                w.state(state)
            except Exception:
                try:
                    w.configure(state="normal" if enabled else "disabled")
                except Exception:
                    pass

    def _update_enabled_states(self):
        """Grey out parameters that don't apply to the current method/toggles."""
        method = str(self.baseline_var.get()).lower()
        self._set_field_state("baseline_poly_order", method == "polynomial")
        self._set_field_state("baseline_als_lambda", method == "als")
        self._set_field_state("baseline_als_p",      method == "als")
        on_pos = bool(self.refine_var.get())
        self._set_field_state("refine_window", on_pos)
        on_w = bool(self.refine_widths_var.get())
        self._set_field_state("width_min_factor", on_w)
        self._set_field_state("width_max_factor", on_w)
        for w in (getattr(self, "_wmode_combo", None),):
            if w is not None:
                try:
                    w.configure(state="readonly" if on_w else "disabled")
                except Exception:
                    pass

    def _on_profile_change(self):
        self._update_fwhm_label()
        self._pull_fields_and_redraw()

    def _update_fwhm_label(self):
        """Recompute the read-only effective-FWHM label from the WG/WL/profile
        widgets. Falls back silently while widgets are mid-edit."""
        if not hasattr(self, "fwhm_var"):
            return
        try:
            wg = float(self.vars["wg"].get())
            wl = float(self.vars["wl"].get())
            prof = str(self.profile_var.get())
            tmp = type(self.cfg)(wg=wg, wl=wl, profile=prof)
            self.fwhm_var.set(f"{tmp.effective_fwhm():.4g}")
        except Exception:
            self.fwhm_var.set("--")

    def _on_baseline_change(self):
        self._pull_fields()
        self._update_enabled_states()

    def _on_refine_toggle(self):
        self._pull_fields()
        self._update_enabled_states()

    def _cfg_str(self, key):
        v = getattr(self.cfg, key)
        return "" if v is None else str(v)

    def _match_conversion(self):
        """Return the affine preset label whose scale/offset match the current
        cfg, or 'custom' if none does. Reciprocal presets are TOML-only and are
        not matched here."""
        if str(self.cfg.x_transform).lower() != "affine":
            return "custom"
        for label, (sc, off, tr, _lab, _unit) in X_CONVERSIONS.items():
            if (tr == "affine" and abs(self.cfg.x_scale - sc) < 1e-9
                    and abs(self.cfg.x_offset - off) < 1e-12):
                return label
        return "custom"

    def _on_conversion_preset(self):
        """Apply a chosen x-conversion preset, then reload the reference so the
        transform is applied to the original data (never to transformed values)."""
        label = self.conv_var.get()
        if label == "custom":
            return
        apply_conversion(self.cfg, label)
        self.vars["x_scale"].set(str(self.cfg.x_scale))
        self.vars["x_offset"].set(str(self.cfg.x_offset))
        self.vars["x_label"].set(self.cfg.x_label)
        self.vars["x_unit"].set(self.cfg.x_unit)
        if self.x is not None and os.path.isfile(self.cfg.reference_file):
            self._load_reference(self.cfg.reference_file)  # re-read & re-transform

    def _make_slider(self, parent, key, label, lo, hi, init, col):
        tk = self.tk
        frame = self.tk.Frame(parent)
        frame.grid(row=0, column=col, padx=8, sticky="we")
        parent.grid_columnconfigure(col, weight=1)
        lbl = self.tk.Label(frame, text=label)
        lbl.pack(side=tk.TOP, anchor="w")
        var = tk.DoubleVar(value=init)
        scale = tk.Scale(frame, from_=lo, to=hi, orient=tk.HORIZONTAL,
                         variable=var, resolution=(1 if key in ("distance", "bg_window") else 0.1),
                         length=240, command=lambda _v: self._on_slider())
        scale.pack(side=tk.TOP, fill=tk.X)
        self.sliders[key] = var
        tip = _TOOLTIPS.get(key)
        if tip:
            _Tooltip(lbl, tip)
            _Tooltip(scale, tip)

    def _slider_vals(self):
        return (float(self.sliders["prominence"].get()),
                int(self.sliders["distance"].get()),
                float(self.sliders["min_snr"].get()),
                int(self.sliders["bg_window"].get()))

    def _pull_fields(self):
        """Read the text fields / dropdown into self.cfg (silently ignore bad input)."""
        def fnum(key, cast=float):
            s = self.vars[key].get().strip()
            if s == "":
                return None
            try:
                return cast(s)
            except ValueError:
                return getattr(self.cfg, key)
        self.cfg.profile = self.profile_var.get()
        for key in ("wg", "wl", "tolerance", "snr_thresh"):
            val = fnum(key)
            if val is not None:
                setattr(self.cfg, key, val)
        self.cfg.x_min = fnum("x_min")
        self.cfg.x_max = fnum("x_max")
        sc = fnum("x_scale"); self.cfg.x_scale = 1.0 if sc is None else sc
        off = fnum("x_offset"); self.cfg.x_offset = 0.0 if off is None else off
        for key in ("x_label", "y_label", "x_unit"):
            setattr(self.cfg, key, self.vars[key].get())
        self.cfg.baseline_method = self.baseline_var.get()
        po = fnum("baseline_poly_order", int)
        if po is not None:
            self.cfg.baseline_poly_order = po
        for key in ("baseline_als_lambda", "baseline_als_p", "refine_window",
                    "width_min_factor", "width_max_factor"):
            val = fnum(key)
            if val is not None:
                setattr(self.cfg, key, val)
        self.cfg.refine = bool(self.refine_var.get())
        self.cfg.refine_widths = bool(self.refine_widths_var.get())
        self.cfg.width_mode = self.width_mode_var.get()
        self.conv_var.set(self._match_conversion())

    def _push_fields(self):
        """Write self.cfg back into the text fields / dropdown."""
        self.profile_var.set(str(self.cfg.profile))
        self.baseline_var.set(str(self.cfg.baseline_method))
        self.refine_var.set(bool(self.cfg.refine))
        self.refine_widths_var.set(bool(self.cfg.refine_widths))
        self.width_mode_var.set(str(self.cfg.width_mode))
        for key in ("wg", "wl", "x_min", "x_max", "tolerance", "snr_thresh",
                    "x_label", "y_label", "x_unit", "x_scale", "x_offset",
                    "baseline_poly_order", "baseline_als_lambda",
                    "baseline_als_p", "refine_window",
                    "width_min_factor", "width_max_factor"):
            if key in self.vars:
                self.vars[key].set(self._cfg_str(key))
        self.conv_var.set(self._match_conversion())
        if hasattr(self, "_update_enabled_states"):
            self._update_enabled_states()
        if hasattr(self, "_update_fwhm_label"):
            self._update_fwhm_label()

    def _reload_on_scale_change(self):
        """x_scale / x_offset edited by hand: re-read the original file so the
        transform is applied once, to the raw data (not to transformed x)."""
        self._pull_fields()
        if self.x is not None and os.path.isfile(self.cfg.reference_file):
            self._load_reference(self.cfg.reference_file)

    def _pull_fields_and_redraw(self):
        self._pull_fields()
        self._update_fwhm_label()
        if self.x is not None:
            self._recompute_mask()
            # Reset the matplotlib toolbar's zoom/pan history so a new
            # x_min/x_max actually takes effect on the displayed view.
            if hasattr(self, "toolbar") and self.toolbar is not None:
                try:
                    self.toolbar.update()
                except Exception:
                    pass
            self._redraw()

    # ---- data loading -------------------------------------------------

    def on_open_folder(self):
        """Open a folder, then a chooser dialog: pick the reference file (one)
        and the component files (many) from the data files in that folder."""
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        folder = filedialog.askdirectory(title="Open folder with data files")
        if not folder:
            return

        # collect candidate data files (common spectroscopy extensions)
        exts = (".txt", ".dat", ".csv", ".nis", ".xy", ".tsv", ".asc",
                ".fio", ".prn", ".chi", ".dataset")
        try:
            files = sorted(
                f for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith(exts)
            )
        except OSError as exc:
            messagebox.showerror("Folder error", str(exc))
            return

        if not files:
            messagebox.showinfo(
                "No data files",
                f"No data files (extensions {', '.join(exts)}) found in:\n{folder}")
            return

        # build the chooser dialog
        dlg = tk.Toplevel(self.root)
        dlg.title("Choose reference and components")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.geometry("640x480")

        ttk.Label(
            dlg, padding=(10, 8),
            text=(f"Folder: {folder}\n"
                  "Select ONE reference (radio) and any number of components "
                  "(checkboxes).")
        ).pack(side=tk.TOP, fill=tk.X)

        # scrollable frame with one row per file
        frame_outer = ttk.Frame(dlg, padding=(8, 0))
        frame_outer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(frame_outer, highlightthickness=0)
        scroll = ttk.Scrollbar(frame_outer, orient="vertical", command=canvas.yview)
        rows = ttk.Frame(canvas)
        rows.bind("<Configure>",
                  lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=rows, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # header
        hdr = ttk.Frame(rows); hdr.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(hdr, text="Ref", width=5, anchor="center").pack(side=tk.LEFT)
        ttk.Label(hdr, text="Cmp", width=5, anchor="center").pack(side=tk.LEFT)
        ttk.Label(hdr, text="File", anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        ref_var = tk.StringVar(value="")
        comp_vars = {}
        # if the first file looks like a sum-/total-/sample-spectrum, preselect it
        likely_ref = next(
            (f for f in files if any(k in f.lower()
                                     for k in ("sum", "total", "sample", "ref"))),
            files[0])
        ref_var.set(likely_ref)
        for f in files:
            row = ttk.Frame(rows); row.pack(side=tk.TOP, fill=tk.X, pady=1)
            ttk.Radiobutton(row, variable=ref_var, value=f).pack(side=tk.LEFT, padx=(8, 18))
            v = tk.BooleanVar(value=(f != likely_ref))   # everything else preselected as component
            comp_vars[f] = v
            ttk.Checkbutton(row, variable=v).pack(side=tk.LEFT, padx=(0, 14))
            ttk.Label(row, text=f, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        # buttons
        result = {"ok": False}
        btns = ttk.Frame(dlg, padding=10); btns.pack(side=tk.BOTTOM, fill=tk.X)

        def on_select_all():
            ref = ref_var.get()
            for f, v in comp_vars.items():
                v.set(f != ref)

        def on_clear():
            for v in comp_vars.values():
                v.set(False)

        def on_ok():
            if not ref_var.get():
                messagebox.showwarning("No reference", "Pick a reference file.")
                return
            # a file cannot be both reference and component
            if comp_vars.get(ref_var.get()) and comp_vars[ref_var.get()].get():
                comp_vars[ref_var.get()].set(False)
            result["ok"] = True
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        ttk.Button(btns, text="Select all components", command=on_select_all).pack(side=tk.LEFT)
        ttk.Button(btns, text="Clear components", command=on_clear).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side=tk.RIGHT)
        ttk.Button(btns, text="OK", command=on_ok).pack(side=tk.RIGHT, padx=(0, 6))

        self.root.wait_window(dlg)
        if not result["ok"]:
            return

        # apply the user's choice
        ref_path = os.path.join(folder, ref_var.get())
        comps = [os.path.join(folder, f) for f, v in comp_vars.items()
                 if v.get() and f != ref_var.get()]
        self.selected_components = comps
        self._load_reference(ref_path)
        self.status.set(
            f"{os.path.basename(ref_path)}: {len(self.x)} points, "
            f"errors {'from file' if self.had_errors else 'estimated'}, "
            f"{len(comps)} component(s) chosen.")

    def _load_reference(self, fn):
        from tkinter import messagebox
        self._pull_fields()
        self.cfg.reference_file = fn
        try:
            self.x, self.y, self.yerr, self.had_errors = load_xy(fn, self.cfg)
        except Exception as exc:
            messagebox.showerror("Load error", f"Could not read file:\n{exc}")
            return
        self._recompute_mask()
        self._auto_search(redraw=False)
        self._redraw()
        if self.selected_components is None:
            ncomp = len(find_component_files(self.cfg))
            comp_msg = f"{ncomp} component file(s) auto-detected."
        else:
            comp_msg = f"{len(self.selected_components)} component(s) chosen."
        self.status.set(
            f"{os.path.basename(fn)}: {len(self.x)} points, "
            f"errors {'from file' if self.had_errors else 'estimated'}, "
            f"{comp_msg}")

    def _recompute_mask(self):
        self._mask = x_range_mask(self.x, self.cfg.x_min, self.cfg.x_max)
        vis_y = self.y[self._mask] if np.any(self._mask) else self.y
        self._yrange = (vis_y.max() - vis_y.min()) if vis_y.max() != vis_y.min() else 1.0

    # ---- peak operations ----------------------------------------------

    def _auto_search(self, redraw=True):
        if self.x is None:
            return
        prom, dist, min_snr, bgw = self._slider_vals()
        idx = search_peaks(self.y, self.yerr, self.x, self._mask,
                           prom, dist, min_snr, bgw)
        self.peaks = sorted(idx.tolist())
        if redraw:
            self._redraw()

    def _clear_peaks(self):
        self.peaks = []
        self._redraw()

    def _on_slider(self):
        if self.x is None:
            return
        self._auto_search(redraw=True)

    def _on_click(self, event):
        if self.x is None or event.inaxes != self.ax or event.xdata is None:
            return
        idx = int(np.argmin(np.abs(self.x - event.xdata)))
        if event.button == 1:           # add
            if idx not in self.peaks:
                self.peaks.append(idx)
                self.peaks.sort()
                self._redraw()
        elif event.button == 3:         # remove nearest
            if self.peaks:
                clicked_x = self.x[idx]
                peak_xs = self.x[self.peaks]
                nearest = self.peaks[int(np.argmin(np.abs(peak_xs - clicked_x)))]
                if abs(self.x[nearest] - clicked_x) < (self.x[-1] - self.x[0]) * 0.04:
                    self.peaks.remove(nearest)
                    self._redraw()

    # ---- drawing ------------------------------------------------------

    def _redraw(self):
        ax = self.ax
        ax.clear()
        prom, dist, min_snr, bgw = self._slider_vals()

        ax.errorbar(self.x, self.y, yerr=self.yerr, fmt="o", ms=3.5, color="black",
                    ecolor="#999999", capsize=2, lw=0.7, label="Data", zorder=2)
        y_corr = subtract_background(self.y, bgw)
        ax.plot(self.x, y_corr, "-", color="#4488cc", lw=1.0, alpha=0.6,
                label="BG-corrected", zorder=1)

        if self.cfg.x_min is not None or self.cfg.x_max is not None:
            xlo = self.cfg.x_min if self.cfg.x_min is not None else self.x[0]
            xhi = self.cfg.x_max if self.cfg.x_max is not None else self.x[-1]
            ax.axvspan(xlo, xhi, alpha=0.07, color="orange")

        if self.peaks:
            px = self.x[self.peaks]
            py = self.y[self.peaks]
            off = self._yrange * self._MARKER_OFFSET_FRAC
            ax.scatter(px, py + off, marker="v", s=110, color="red", zorder=5)
            for xi, yi in zip(px, py):
                ax.annotate(f"{xi:.1f}", (xi, yi + off * 2.2), ha="center",
                            fontsize=7, color="red", fontweight="bold")

        x_lo = self.cfg.x_min if self.cfg.x_min is not None else self.x[0]
        x_hi = self.cfg.x_max if self.cfg.x_max is not None else self.x[-1]
        ax.set_xlim(x_lo, x_hi)
        vis = (self.x >= x_lo) & (self.x <= x_hi)
        if np.any(vis):
            y_lo, y_hi = self.y[vis].min(), self.y[vis].max()
            span = y_hi - y_lo if y_hi != y_lo else 1.0
            ax.set_ylim(y_lo - 0.05 * span, y_hi + 0.30 * span)

        ax.set_xlabel(_xlabel(self.cfg))
        ax.set_ylabel(self.cfg.y_label)
        ax.set_title(f"{len(self.peaks)} peaks  |  left-click add, right-click remove  |  "
                     f"prom={prom:.0f} dist={dist} SNR={min_snr:.1f} BG={bgw}",
                     fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw_idle()

    # ---- fit ----------------------------------------------------------

    def on_fit(self):
        """Run the fit and presence check on the current peaks and save all outputs."""
        from tkinter import messagebox
        if self.x is None:
            messagebox.showinfo("No data", "Open a data file first.")
            return
        if not self.peaks:
            messagebox.showinfo("No peaks", "Select at least one peak "
                                "(auto-search or left-click).")
            return
        self._pull_fields()
        prom, dist, min_snr, bgw = self._slider_vals()
        self.cfg.bg_window = bgw
        problems = self.cfg.validate()
        if problems:
            messagebox.showerror("Invalid settings", "\n".join(problems))
            return

        peak_centers = self.x[np.array(self.peaks, dtype=int)]
        picker_settings = {"prominence": prom, "distance": dist,
                           "min_snr": min_snr, "bg_window": bgw}
        self.status.set("Fitting and saving ...")
        self.root.update_idletasks()
        try:
            res = run_analysis(self.cfg, self.x, self.y, self.yerr, peak_centers,
                               picker_settings, self.had_errors, verbose=True,
                               component_files=self.selected_components)
        except Exception as exc:
            messagebox.showerror("Run error", f"{type(exc).__name__}: {exc}")
            self.status.set("Error during fit — see console.")
            return
        st = res["stats"]
        msg = (f"Done. {st['n_active']}/{len(peak_centers)} active peaks, "
               f"chi2_red={st['chi2_red']:.3g}, R2={st['r2']:.4f}.")
        if res["xlsx"]:
            msg += f"\nExcel: {os.path.basename(res['xlsx'])}"
        hint = res.get("hint")
        self.status.set(msg.replace("\n", "   ")
                        + ("   [!] check WG/WL & x conversion" if hint else ""))
        out_dir = self.cfg.output_dir.strip() if self.cfg.output_dir else None
        where = out_dir if out_dir else "next to the data file"
        full = msg + f"\n\nPlots and CSV written: {where}"
        if hint:
            messagebox.showwarning("Finished — but the fit looks poor",
                                   full + "\n\n[!] " + hint)
        else:
            messagebox.showinfo("Finished", full)

    # ---- output folder -----------------------------------------------

    def on_pick_output_folder(self):
        """File dialog: choose where the results (plots, Excel, CSV) are written.
        Empty path = next to the reference file (the default)."""
        from tkinter import filedialog
        initial = self.cfg.output_dir or (
            os.path.dirname(os.path.abspath(self.cfg.reference_file))
            if self.cfg.reference_file else os.getcwd())
        chosen = filedialog.askdirectory(title="Choose output folder",
                                         initialdir=initial)
        if chosen:
            self.cfg.output_dir = chosen
            self._update_out_label()

    def _update_out_label(self):
        """Update the 'output:' display next to the Output folder button."""
        d = self.cfg.output_dir.strip() if self.cfg.output_dir else ""
        if not d:
            txt = "(next to data file)"
        else:
            # show only the last two path components, to keep the bar compact
            parts = os.path.normpath(d).split(os.sep)
            txt = (os.sep.join(parts[-2:]) if len(parts) >= 2 else d)
            txt = "out: " + txt
        if hasattr(self, "out_label"):
            self.out_label.config(text=txt)

    # ---- config load/save --------------------------------------------

    def on_load_cfg(self):
        """File dialog: load a TOML configuration into the GUI."""
        from tkinter import filedialog, messagebox
        fn = filedialog.askopenfilename(title="Load TOML config",
                                        filetypes=[("TOML", "*.toml"), ("All", "*.*")])
        if not fn:
            return
        try:
            cfg = load_config(fn)
            _coerce_nullable(cfg)
        except Exception as exc:
            messagebox.showerror("Config error", str(exc))
            return
        cfg.peaks = []
        self.cfg = cfg
        self._push_fields()
        for key, var in (("prominence", "prominence_init"), ("distance", "distance_init"),
                         ("min_snr", "min_snr_init"), ("bg_window", "bg_window")):
            self.sliders[key].set(getattr(self.cfg, var))
        if os.path.isfile(self.cfg.reference_file):
            self._load_reference(self.cfg.reference_file)
        else:
            self.status.set(f"Config loaded: {os.path.basename(fn)} "
                            "(reference file not found — open one).")
        self._update_out_label()

    def on_save_cfg(self):
        """File dialog: save the current GUI settings to a TOML file."""
        from tkinter import filedialog, messagebox
        fn = filedialog.asksaveasfilename(title="Save TOML config", defaultextension=".toml",
                                          filetypes=[("TOML", "*.toml")])
        if not fn:
            return
        self._pull_fields()
        prom, dist, min_snr, bgw = self._slider_vals()
        self.cfg.prominence_init = prom
        self.cfg.distance_init = dist
        self.cfg.min_snr_init = min_snr
        self.cfg.bg_window = bgw
        try:
            write_template(fn, self.cfg)
            messagebox.showinfo("Saved", f"Configuration written to:\n{fn}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()
