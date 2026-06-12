"""
Configuration: the Config dataclass, TOML I/O, and named axis conversions.

This module collects everything the user is normally allowed to tune:
  * `Config` — a `@dataclass` holding every run parameter; instances are passed
    around the whole pipeline (read by load_xy, the fitter, the plotters, the
    output writers and the GUI), so a single struct fully describes a run.
  * `X_CONVERSIONS` — named axis presets (meV → cm⁻¹ etc.); affine for the GUI,
    affine + reciprocal via the TOML.
  * `load_config` / `write_template` — TOML I/O with comments per parameter.
  * `apply_conversion` / `apply_x_transform` — apply a preset / map an x array.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

import numpy as np


# ============================================================
#  CONFIGURATION
# ============================================================

@dataclass
class Config:
    """All run parameters. Loaded from / saved to a TOML file, or edited live
    in the GUI. Every output records the full set for reproducibility."""

    # --- input -------------------------------------------------------------
    reference_file: str = "data.txt"   # reference data set (peaks defined here)
    component_glob: str = ""           # glob for component sets; "" = auto:
                                       #   "<reference-stem>_*.<ext>"
    error_column: str = "auto"         # "auto" | "none" | column index (0-based)
    fio_x_column: str = ""             # FIO files: x column name or 1-based index; "" = first column
    fio_y_column: str = "nisp"         # FIO files: y column name or 1-based index (default NIS signal)

    # --- x-axis transform (applied on load) --------------------------------
    #   affine:     x -> x*x_scale + x_offset
    #   reciprocal: x -> x_scale/x + x_offset   (wavelength <-> wavenumber etc.)
    x_conversion: str = ""             # optional named preset (see X_CONVERSIONS);
                                       # if set, it overrides the four fields below
    x_transform: str = "affine"        # "affine" | "reciprocal"
    x_scale: float = 1.0               # e.g. 8.065544 to convert meV -> cm^-1
    x_offset: float = 0.0

    # --- labels (purely cosmetic, used in plots / sheets) ------------------
    x_label: str = "x"
    y_label: str = "y"
    x_unit: str = ""                   # short unit shown in tables, e.g. "cm-1"

    # --- line profile ------------------------------------------------------
    profile: str = "voigt"             # "voigt" | "gauss" | "lorentz"
    wg: float = 7.8701                 # Gaussian FWHM (width of G part / pure G)
    wl: float = 1.094                  # Lorentzian FWHM (width of L part / pure L)

    # --- search window -----------------------------------------------------
    x_min: float | None = None         # None = full range
    x_max: float | None = None

    # --- background & peak search -----------------------------------------
    bg_window: int = 30                # rolling-minimum window (points); 0 = off
    baseline_method: str = "rolling_min"   # "rolling_min" | "polynomial" | "als"
    baseline_poly_order: int = 3       # order for the polynomial baseline
    baseline_als_lambda: float = 1.0e5  # smoothness for the ALS baseline
    baseline_als_p: float = 0.01       # asymmetry for the ALS baseline
    prominence_init: float = 200.0
    distance_init: int = 3
    min_snr_init: float = 2.0

    # --- optional position refinement -------------------------------------
    refine: bool = False               # refine peak positions by a bounded fit
    refine_window: float = 3.0         # max position shift (x units) when refining
    # --- optional per-peak width refinement -------------------------------
    refine_widths: bool = False        # also refine per-peak widths (bounded)
    width_mode: str = "fwhm"           # "fwhm" (scale whole Voigt) | "sigma" (Gaussian part only)
    width_min_factor: float = 1.0      # lower width bound = factor x instrumental width
    width_max_factor: float = 2.0      # upper width bound = factor x instrumental width

    # --- presence check in component sets ---------------------------------
    tolerance: float = 5.0             # +/- position window (x units)
    snr_thresh: float = 2.0            # min signal/error to count as present
    presence_baseline_corrected: bool = True
    #   True  -> SNR = y_corrected / yerr  (signal *above the background*;
    #            physically the right measure of peak presence)
    #   False -> SNR = y_raw / yerr        (legacy behaviour; raw counts)

    # --- error treatment ---------------------------------------------------
    error_mode: str = "statistical"    # "statistical" | "scaled"
                                       # forced to "scaled" when no errors exist

    # --- outputs -----------------------------------------------------------
    write_csv: bool = True
    write_excel: bool = True
    plot_dpi: int = 150

    # --- output --------------------------------------------------------------
    output_dir: str = ""   # empty = use the reference file's directory
    # --- not persisted: peaks set interactively or listed in TOML ----------
    peaks: list = field(default_factory=list)   # explicit peak x-positions

    # ---- profile helpers --------------------------------------------------
    def sigma(self) -> float:
        """Gaussian FWHM -> standard deviation."""
        return self.wg / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    def gamma(self) -> float:
        """Lorentzian FWHM -> half width at half maximum."""
        return self.wl / 2.0

    def effective_fwhm(self) -> float:
        r"""Total full width at half maximum (FWHM) of the selected profile.

        For the Voigt profile the FWHM has no closed form; the widely used
        empirical approximation of Olivero & Longbothum [1]_ is applied
        (accurate to about 0.02 %):

        .. math:: f_V \approx 0.5346\,f_L + \sqrt{0.2166\,f_L^2 + f_G^2},

        with :math:`f_G = \texttt{wg}` and :math:`f_L = \texttt{wl}`. For the
        pure profiles the FWHM is ``wg`` (Gaussian) or ``wl`` (Lorentzian).

        Returns
        -------
        float
            Effective FWHM in x units.

        References
        ----------
        .. [1] J. J. Olivero, R. L. Longbothum, J. Quant. Spectrosc. Radiat.
           Transfer 17 (1977) 233-236.
        """
        p = str(self.profile).lower()
        if p in ("gauss", "gaussian"):
            return self.wg
        if p in ("lorentz", "lorentzian"):
            return self.wl
        return 0.5346 * self.wl + np.sqrt(0.2166 * self.wl ** 2 + self.wg ** 2)

    def range_suffix(self) -> str:
        """Filename suffix reflecting the active x-range, e.g. '_100_200'."""
        if self.x_min is None and self.x_max is None:
            return ""
        lo = f"{int(self.x_min)}" if self.x_min is not None else "min"
        hi = f"{int(self.x_max)}" if self.x_max is not None else "max"
        return f"_{lo}_{hi}"

    def validate(self) -> list:
        """Return a list of human-readable problems (empty = OK)."""
        problems = []
        if str(self.profile).lower() not in (
            "voigt", "gauss", "gaussian", "lorentz", "lorentzian"
        ):
            problems.append(f"unknown profile '{self.profile}'")
        if self.wg <= 0 and str(self.profile).lower() in ("voigt", "gauss", "gaussian"):
            problems.append("wg must be > 0 for a Gaussian/Voigt profile")
        if self.wl <= 0 and str(self.profile).lower() in ("voigt", "lorentz", "lorentzian"):
            problems.append("wl must be > 0 for a Lorentzian/Voigt profile")
        if self.x_min is not None and self.x_max is not None and self.x_min >= self.x_max:
            problems.append("x_min must be < x_max")
        if self.bg_window < 0:
            problems.append("bg_window must be >= 0")
        if self.tolerance <= 0:
            problems.append("tolerance must be > 0")
        if str(self.baseline_method).lower() not in (
            "rolling_min", "rolling", "min", "polynomial", "poly",
            "als", "arpls", "asymmetric"
        ):
            problems.append(f"unknown baseline_method '{self.baseline_method}'")
        if self.refine and self.refine_window <= 0:
            problems.append("refine_window must be > 0 when refine = true")
        if str(self.width_mode).lower() not in ("fwhm", "sigma"):
            problems.append(f"unknown width_mode '{self.width_mode}' (use 'fwhm' or 'sigma')")
        if self.refine_widths:
            if self.width_min_factor <= 0:
                problems.append("width_min_factor must be > 0 when refine_widths = true")
            if self.width_max_factor <= self.width_min_factor:
                problems.append("width_max_factor must be > width_min_factor when refine_widths = true")
        return problems


# ---- TOML load / save ------------------------------------------------------

# Keys that live in the TOML file (peaks excluded — those are run-specific).
_TOML_KEYS = [f.name for f in fields(Config) if f.name != "peaks"]


def _load_tomllib():
    """Return a TOML-reading module (tomllib on 3.11+, else tomli)."""
    try:
        import tomllib            # Python >= 3.11
        return tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
            return tomllib
        except ModuleNotFoundError:
            return None


def load_config(path: str) -> Config:
    """Load a Config from a TOML file. Unknown keys are ignored with a warning;
    missing keys keep their defaults."""
    tomllib = _load_tomllib()
    cfg = Config()
    if tomllib is None:
        raise RuntimeError(
            "Reading TOML needs Python 3.11+ (tomllib) or the 'tomli' package "
            "(pip install tomli)."
        )
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    valid = set(_TOML_KEYS)

    def apply(key, val):
        if key == "peaks":
            cfg.peaks = list(val)
        elif key in valid:
            setattr(cfg, key, val)
        else:
            print(f"  [warn] unknown config key '{key}' ignored")

    for key, val in data.items():
        if isinstance(val, dict):
            # a [section] table -> flatten its entries into the flat Config
            for sub_key, sub_val in val.items():
                apply(sub_key, sub_val)
        else:
            apply(key, val)

    # A named conversion preset overrides scale/offset/transform/label/unit.
    if cfg.x_conversion:
        if not apply_conversion(cfg, cfg.x_conversion):
            print(f"  [warn] unknown x_conversion '{cfg.x_conversion}' ignored; "
                  "using x_scale/x_offset/x_transform as given.")

    # Make the config portable: a *relative* reference_file is resolved against
    # the directory of the config file, not the current working directory. This
    # means a job.toml and its data can live together and be run from anywhere
    # (e.g. batch pipelines). Absolute paths and an empty value are left as-is;
    # a command-line --reference (handled in cli.py) still wins and keeps the
    # usual shell semantics (relative to the working directory).
    if cfg.reference_file and not os.path.isabs(cfg.reference_file):
        base = os.path.dirname(os.path.abspath(path))
        cfg.reference_file = os.path.normpath(os.path.join(base, cfg.reference_file))
    # Same portability rule for a relative output_dir set *in the config*. A
    # command-line --output-dir (applied later in cli.py) overrides this and
    # keeps the usual working-directory semantics. Empty = next to the reference.
    if cfg.output_dir and cfg.output_dir.strip() and not os.path.isabs(cfg.output_dir):
        base = os.path.dirname(os.path.abspath(path))
        cfg.output_dir = os.path.normpath(os.path.join(base, cfg.output_dir.strip()))
    # Normalise nullable numeric sentinels ("" -> None) so that every entry
    # point (CLI, GUI, direct API) gets the dataclass invariant x_min/x_max is
    # None-or-float. Without this, make_fit_mask() would receive a bare "".
    _coerce_nullable(cfg)
    return cfg


def _toml_value(v) -> str:
    """Serialise a Python value as a TOML scalar."""
    if v is None:
        # TOML has no null; we represent "unset" as an empty string and
        # convert back on load for the nullable numeric fields.
        return '""'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(i) for i in v) + "]"
    return '"' + str(v) + '"'


_TOML_SECTIONS = [
    ("input", ["reference_file", "component_glob", "error_column",
               "fio_x_column", "fio_y_column"]),
    ("xaxis", ["x_conversion", "x_transform", "x_scale", "x_offset",
               "x_label", "y_label", "x_unit"]),
    ("profile", ["profile", "wg", "wl"]),
    ("window", ["x_min", "x_max"]),
    ("search", ["bg_window", "baseline_method", "baseline_poly_order",
                "baseline_als_lambda", "baseline_als_p",
                "prominence_init", "distance_init", "min_snr_init"]),
    ("refine", ["refine", "refine_window", "refine_widths", "width_mode",
                "width_min_factor", "width_max_factor"]),
    ("presence", ["tolerance", "snr_thresh", "presence_baseline_corrected"]),
    ("errors", ["error_mode"]),
    ("output", ["output_dir", "write_csv", "write_excel", "plot_dpi"]),
]

_TOML_COMMENTS = {
    "reference_file": "data set in which the peaks are defined",
    "component_glob": 'glob for component sets; "" = auto "<stem>_*.<ext>"',
    "error_column":   '"auto" | "none" | 0-based column index',
    "fio_x_column":   'FIO only: x column name or 1-based index; "" = first column',
    "fio_y_column":   'FIO only: y column name or 1-based index (e.g. "nisp", "nfsp")',
    "x_conversion":   'named preset, e.g. "meV -> cm^-1" or "nm -> eV (reciprocal)"; "" = none',
    "x_transform":    '"affine" (x*scale+offset) | "reciprocal" (scale/x+offset)',
    "x_scale":        "affine: factor (8.065544 = meV -> cm^-1); reciprocal: numerator",
    "profile":        '"voigt" | "gauss" | "lorentz" (Voigt is a true Voigt)',
    "wg":             "Gaussian FWHM (width of the G part / of a pure Gaussian)",
    "wl":             "Lorentzian FWHM (width of the L part / of a pure Lorentzian)",
    "x_min":          'lower search bound; "" = open',
    "x_max":          'upper search bound; "" = open',
    "bg_window":      "rolling-minimum window in points; 0 = off",
    "baseline_method": '"rolling_min" | "polynomial" | "als" (asymmetric least squares)',
    "baseline_poly_order": "order for the polynomial baseline",
    "baseline_als_lambda": "ALS smoothness (larger = stiffer baseline)",
    "baseline_als_p": "ALS asymmetry (smaller = baseline hugs the lower envelope)",
    "refine":         "refine peak positions by a bounded nonlinear fit after the NNLS",
    "refine_window":  "maximum position shift in x units when refine = true",
    "refine_widths":  "also refine per-peak widths (bounded) after the NNLS",
    "width_mode":     '"fwhm" (scale whole Voigt) | "sigma" (Gaussian part only)',
    "width_min_factor": "lower width bound as a factor of the instrumental width (1.0 = instrument)",
    "width_max_factor": "upper width bound as a factor of the instrumental width (e.g. 2.0 = up to 2x)",
    "tolerance":      "+/- position window for the presence check (x units)",
    "snr_thresh":     "min signal/error ratio to count a peak as present",
    "presence_baseline_corrected": "true: SNR on background-corrected signal (recommended); false: on raw signal",
    "error_mode":     '"statistical" (true 1-sigma) | "scaled" (x sqrt(chi2_red))',
    "output_dir":     'where to write results; "" = next to the reference file',
}


# Named x-axis conversion presets, selectable in the TOML (`x_conversion`) and,
# for the linear ones, in the GUI drop-down. Each preset sets the scale, the
# offset, the transform type, and the cosmetic axis label / unit.
#
# Transform types:
#   "affine"     -> x_new = x * scale + offset           (linear; GUI + TOML)
#   "reciprocal" -> x_new = scale / x  + offset           (TOML only)
#
# Linear conversions relate energy / frequency / wavenumber units, which are
# proportional to one another. Reciprocal conversions relate a wavelength to a
# wavenumber/energy/frequency (e.g. cm^-1 = 1e7 / lambda[nm]); these are *not*
# linear and therefore use the "reciprocal" transform. Physical constants are
# CODATA-based: 1 meV = 8.065544 cm^-1; hc = 1239.84198 eV*nm;
# c = 2.99792458e5 nm*THz.
X_CONVERSIONS = {
    # label                       scale         offset  transform      x_label        x_unit
    "none (x unchanged)":        (1.0,          0.0,    "affine",      "x",           ""),
    "meV -> cm^-1":              (8.065544,     0.0,    "affine",      "Wavenumber",  "cm-1"),
    "cm^-1 -> meV":              (0.12398419,   0.0,    "affine",      "Energy",      "meV"),
    "meV -> THz":                (0.24179893,   0.0,    "affine",      "Frequency",   "THz"),
    "cm^-1 -> THz":              (0.029979246,  0.0,    "affine",      "Frequency",   "THz"),
    "THz -> cm^-1":              (33.356410,    0.0,    "affine",      "Wavenumber",  "cm-1"),
    "eV -> cm^-1":               (8065.544,     0.0,    "affine",      "Wavenumber",  "cm-1"),
    "meV -> K (k_B)":            (11.604518,    0.0,    "affine",      "Temperature", "K"),
    "minutes -> seconds":        (60.0,         0.0,    "affine",      "Time",        "s"),
    # reciprocal (TOML only) -- wavelength <-> wavenumber / energy / frequency
    "nm -> cm^-1 (reciprocal)":  (1.0e7,        0.0,    "reciprocal",  "Wavenumber",  "cm-1"),
    "cm^-1 -> nm (reciprocal)":  (1.0e7,        0.0,    "reciprocal",  "Wavelength",  "nm"),
    "nm -> eV (reciprocal)":     (1239.84198,   0.0,    "reciprocal",  "Energy",      "eV"),
    "eV -> nm (reciprocal)":     (1239.84198,   0.0,    "reciprocal",  "Wavelength",  "nm"),
    "nm -> THz (reciprocal)":    (2.99792458e5, 0.0,    "reciprocal",  "Frequency",   "THz"),
}


def apply_conversion(cfg, label) -> bool:
    """Apply a named X_CONVERSIONS preset to `cfg` (scale/offset/transform and
    the cosmetic label/unit). Returns True if the label was a known preset."""
    if label not in X_CONVERSIONS:
        return False
    scale, offset, transform, xlabel, xunit = X_CONVERSIONS[label]
    cfg.x_scale, cfg.x_offset = scale, offset
    cfg.x_transform = transform
    cfg.x_label, cfg.x_unit = xlabel, xunit
    return True


def apply_x_transform(x, cfg):
    """Map the raw abscissa to the working units according to cfg.x_transform.

    "affine"     -> x*x_scale + x_offset
    "reciprocal" -> x_scale/x + x_offset   (for wavelength <-> wavenumber etc.;
                    zeros are guarded to avoid division by zero)
    """
    if str(cfg.x_transform).lower() == "reciprocal":
        x = np.asarray(x, dtype=float)
        safe = np.where(x == 0.0, np.nan, x)
        return cfg.x_scale / safe + cfg.x_offset
    return x * cfg.x_scale + cfg.x_offset


def write_template(path: str, cfg: Config | None = None) -> None:
    """Write a commented TOML template (current defaults, or `cfg`)."""
    cfg = cfg or Config()
    lines = [
        "# PeakCheck configuration",
        "# Generated template - edit and pass with:  peakcheck.py --config "
        + os.path.basename(path),
        "#",
        '# Tip: an empty string "" means "unset" for x_min / x_max.',
        "",
    ]
    for section, keys in _TOML_SECTIONS:
        lines.append(f"[{section}]")
        for key in keys:
            val = getattr(cfg, key)
            comment = _TOML_COMMENTS.get(key, "")
            entry = f"{key} = {_toml_value(val)}"
            if comment:
                entry += f"    # {comment}"
            lines.append(entry)
        lines.append("")
    lines.append("# Optional: fix the peak positions (x values) instead of")
    lines.append("# picking them interactively. Leave empty to search/pick.")
    lines.append("# peaks = [120.0, 150.0, 175.0]")
    lines.append("")
    with open(path, "w", encoding="ascii", errors="replace") as fh:
        fh.write("\n".join(lines))
    print(f"  Template written: {path}")


def _coerce_nullable(cfg: Config) -> None:
    """Convert empty-string sentinels for nullable numeric fields back to None."""
    for key in ("x_min", "x_max"):
        v = getattr(cfg, key)
        if v == "" or v is None:
            setattr(cfg, key, None)
        else:
            setattr(cfg, key, float(v))
