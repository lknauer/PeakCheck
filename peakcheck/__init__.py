"""
PeakCheck
=========
Interactive multi-peak fitting and peak-presence checking for generic x/y data.

PeakCheck determines peaks in a *reference* data set (e.g. a summed spectrum)
and then checks whether those same peaks are present in one or more *component*
data sets (e.g. sub-spectra). It works with any x/y data — spectra, diffraction
patterns, chromatograms, any 2-column or 3-column numeric table.

Two ways to run
---------------
  * Graphical (default):     ``python -m peakcheck``  or  ``peakcheck``
  * Headless / scriptable:   ``peakcheck --config job.toml --no-gui``
  * Write a template:        ``peakcheck --write-template job.toml``

Core method
-----------
  * area-normalised line profiles (true Voigt via the Faddeeva function)
  * rolling-minimum background subtraction
  * weighted non-negative least squares (NNLS) for the amplitudes at fixed
    positions and widths -> amplitude is the integrated intensity
  * goodness of fit (reduced chi^2, R^2) and per-amplitude standard errors
  * three-stage peak-presence search (strict max, shoulder, duplicate
    resolution) in the component data sets

License
-------
  MIT License (c) 2026 Lukas Knauer, RPTU Kaiserslautern-Landau,
  AG Schuenemann. SPDX-License-Identifier: MIT.

References
----------
The methods used here rest on the following standard works; individual
functions cite the relevant entry in their own docstrings.

[1] J. J. Olivero and R. L. Longbothum, "Empirical fits to the Voigt line
    width: A brief review", J. Quant. Spectrosc. Radiat. Transfer 17 (1977)
    233-236. doi:10.1016/0022-4073(77)90161-3.
[2] M. Abramowitz and I. A. Stegun, "Handbook of Mathematical Functions",
    Dover (1972), Sec. 7.1.
[3] C. L. Lawson and R. J. Hanson, "Solving Least Squares Problems",
    Prentice-Hall (1974); reprinted SIAM, Classics in Applied Mathematics
    (1995), Chapter 23.
[4] P. R. Bevington and D. K. Robinson, "Data Reduction and Error Analysis
    for the Physical Sciences", 3rd ed., McGraw-Hill (2003).
[5] W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery,
    "Numerical Recipes", 3rd ed., Cambridge University Press (2007), Ch. 15.
[6] P. J. Rousseeuw and C. Croux, "Alternatives to the median absolute
    deviation", J. Am. Stat. Assoc. 88 (1993) 1273-1283.
[7] P. Virtanen et al., "SciPy 1.0", Nature Methods 17 (2020) 261-272.
[8] scipy.optimize.curve_fit documentation, parameter `absolute_sigma`.
"""
from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Lukas Knauer"

# Public API — re-exported so users can write `from peakcheck import voigt`
# without having to know the internal module layout. Modules that need a
# function importing it from the appropriate submodule directly (e.g. the
# fit module uses `from .profiles import profile_shape`); the names here are
# for downstream users and the test suite.
from .config import (
    Config,
    X_CONVERSIONS,
    apply_conversion,
    apply_x_transform,
    load_config,
    write_template,
    _coerce_nullable,
)
from .profiles import voigt, gaussian, lorentzian, profile_shape
from .io import estimate_noise, load_xy
from .background import subtract_background, subtract_background_cfg, x_range_mask
from .fit import (
    search_peaks, make_fit_mask, compute_model, refine_peaks,
    fit_amplitudes, fit_statistics, refine_widths, per_peak_widths,
    fwhm_from_sigma_gamma,
)
from .presence import check_peaks_in_component
from .plots import plot_fit_result, plot_component
from .output import gather_parameters, save_excel, save_csv
from .pipeline import find_component_files, run_analysis, run_headless
from .cli import build_arg_parser, main

# `PeakCheckGUI` is imported lazily so a headless install without tkinter
# does not fail on `import peakcheck` (the GUI is only needed when starting it).
def __getattr__(name):
    if name == "PeakCheckGUI":
        from .gui import PeakCheckGUI
        return PeakCheckGUI
    raise AttributeError(f"module 'peakcheck' has no attribute {name!r}")


__all__ = [
    "__version__", "__author__",
    "Config", "X_CONVERSIONS", "apply_conversion", "apply_x_transform",
    "load_config", "write_template", "_coerce_nullable",
    "voigt", "gaussian", "lorentzian", "profile_shape",
    "estimate_noise", "load_xy",
    "subtract_background", "subtract_background_cfg", "x_range_mask",
    "search_peaks", "make_fit_mask", "compute_model",
    "fit_amplitudes", "fit_statistics", "refine_peaks",
    "refine_widths", "per_peak_widths", "fwhm_from_sigma_gamma",
    "check_peaks_in_component",
    "plot_fit_result", "plot_component",
    "gather_parameters", "save_excel", "save_csv",
    "find_component_files", "run_analysis", "run_headless",
    "build_arg_parser", "main",
    "PeakCheckGUI",
]
