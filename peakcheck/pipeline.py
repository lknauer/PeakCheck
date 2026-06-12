"""
The single analysis pipeline shared by the GUI and the headless run.

`run_analysis` is the core: given data + peak centres, it fits, scans every
component for presence, makes the plots and writes Excel + CSV. The GUI calls
it from `on_fit`; `run_headless` is a thin wrapper that also loads the data,
runs the automatic peak search (or takes the peaks listed in the TOML) and
then dispatches to `run_analysis`. Both entry points share one Config, so a
given configuration reproduces the same result regardless of how it was
launched.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from .background import x_range_mask
from .fit import (fit_amplitudes, fit_statistics, make_fit_mask, refine_peaks,
                  refine_widths, per_peak_widths, fwhm_from_sigma_gamma,
                  search_peaks)
from .io import load_xy
from .output import gather_parameters, save_csv, save_excel
from .plots import plot_component, plot_fit_result
from .presence import check_peaks_in_component


def find_component_files(cfg):
    """Return the sorted list of component files for the reference file,
    using cfg.component_glob or the auto pattern '<stem>_*.<ext>'.
    The reference file itself is excluded."""
    ref = cfg.reference_file
    if cfg.component_glob:
        pattern = cfg.component_glob
        if not os.path.isabs(pattern):
            pattern = os.path.join(os.path.dirname(os.path.abspath(ref)) or ".", pattern)
    else:
        stem, ext = os.path.splitext(ref)
        pattern = f"{stem}_*{ext}"
    files = sorted(f for f in glob.glob(pattern) if os.path.abspath(f) != os.path.abspath(ref))
    return files


def _fit_quality_hint(stats, n_peaks):
    """Return a short diagnostic hint when the fit looks poor, else ``""``.

    Fires when peaks were dropped (NNLS set their amplitude to zero), the fit is
    worse than a flat line (R^2 < 0) or the reduced chi-square is far above one.
    The most common cause for generic data is a profile width (``WG``/``WL``)
    that does not match the data's peak width, or a missing/incorrect x-axis
    conversion, so the hint points the user there.
    """
    n_active = stats.get("n_active", n_peaks)
    r2 = stats.get("r2", 0.0)
    dropped = n_peaks - n_active
    # Only the two unambiguous signals: a peak the user asked for was zeroed by
    # NNLS, or the model is worse than a flat line. A merely large reduced
    # chi-square is not used here -- with estimated errors it is often >1 for a
    # perfectly good fit, so it would cry wolf.
    poor = (dropped > 0) or (np.isfinite(r2) and r2 < 0)
    if not poor:
        return ""
    parts = []
    if dropped > 0:
        parts.append(f"{dropped} of {n_peaks} peaks were dropped (amplitude 0)")
    if np.isfinite(r2) and r2 < 0:
        parts.append("the fit is worse than a flat line (R^2 < 0)")
    lead = "; ".join(parts) if parts else "the fit looks poor"
    return (f"{lead}. This usually means the profile width WG/WL does not match "
            "the data's peak width, or the x-axis conversion/units are not set "
            "as the widths expect. Check WG/WL (and the x conversion) against the "
            "spacing of your peaks.")


def run_analysis(cfg, x, y, yerr, peak_centers, picker_settings, had_errors,
                 verbose=True, component_files=None):
    """
    Fit the reference peaks, scan the component data sets, make plots and write
    Excel + CSV. Returns a results dict. This is the headless core shared with
    the GUI (the GUI calls it after the user confirms the peaks).
    """
    peak_centers = np.asarray(peak_centers, dtype=float)
    force_scaled = not had_errors          # no real errors -> scale by sqrt(chi2_red)

    fit_mask   = make_fit_mask(x, cfg)
    n_extra = 0
    if getattr(cfg, "refine", False) and len(peak_centers) > 0:
        refined = refine_peaks(x, y, yerr, peak_centers, cfg, fit_mask)
        if verbose:
            shift = float(np.max(np.abs(refined - peak_centers))) if len(refined) else 0.0
            print(f"  Position refinement on (max shift {shift:.3g}):")
            for c0, c1 in zip(peak_centers, refined):
                print(f"    {c0:10.3f}  ->  {c1:10.3f}")
        peak_centers = refined
        n_extra += len(peak_centers)

    # optional per-peak width refinement (after positions, before amplitudes)
    width_scales = None
    if getattr(cfg, "refine_widths", False) and len(peak_centers) > 0:
        width_scales = refine_widths(x, y, yerr, peak_centers, cfg, fit_mask)
        n_extra += len(peak_centers)
    sigmas, gammas = per_peak_widths(cfg, len(peak_centers), width_scales)
    fwhms = fwhm_from_sigma_gamma(sigmas, gammas, cfg.profile)
    if width_scales is not None and verbose:
        f0 = float(cfg.effective_fwhm())
        print(f"  Width refinement on (mode '{cfg.width_mode}', "
              f"instrumental FWHM {f0:.3g}):")
        for c, fw, sc in zip(peak_centers, np.atleast_1d(fwhms),
                             np.atleast_1d(width_scales)):
            print(f"    {c:10.3f}  ->  FWHM {fw:8.3f}  ({sc:.3f}x)")

    amplitudes = fit_amplitudes(x, y, yerr, peak_centers, cfg, fit_mask,
                                sigmas=sigmas, gammas=gammas)
    stats      = fit_statistics(x, y, yerr, amplitudes, peak_centers, cfg,
                                fit_mask, force_scaled=force_scaled,
                                sigmas=sigmas, gammas=gammas,
                                n_extra_params=n_extra)

    hint = _fit_quality_hint(stats, len(peak_centers))
    if verbose:
        emode = stats["error_mode"]
        elabel = "statistical 1 sigma" if emode == "statistical" else "1 sigma, chi^2-scaled"
        print(f"  Fit results (amplitude +/- {elabel}):")
        for c, a, se in zip(peak_centers, amplitudes, stats["stderr"]):
            se_str = f"+/- {se:.3f}" if np.isfinite(se) else "+/-   -  "
            print(f"    {c:10.2f}  ->  A = {a:12.4f}  {se_str}")
        print(f"\n  Goodness of fit ({stats['n_points']} pts in fit window):")
        print(f"    reduced chi^2 = {stats['chi2_red']:.4g}   "
              f"R^2 = {stats['r2']:.4f}   "
              f"active = {stats['n_active']}/{len(peak_centers)}   "
              f"error mode = {emode}")
        if hint:
            print(f"    [hint] {hint}")

    # fit plot
    plot_fit_result(x, y, yerr, amplitudes, peak_centers, cfg, cfg.reference_file,
                    stats=stats, sigmas=sigmas, gammas=gammas,
                    fwhms=(fwhms if width_scales is not None else None))

    # components
    if component_files is not None:
        comp_files = list(component_files)
    else:
        comp_files = find_component_files(cfg)
    component_results = []
    if verbose:
        if comp_files:
            print(f"\n  {len(comp_files)} component file(s) found:")
        else:
            src = "explicit list" if component_files is not None else (
                cfg.component_glob or '<stem>_*<ext>')
            print(f"\n  No component files (source: {src}).")
    for cf in comp_files:
        try:
            x_s, y_s, e_s, _ = load_xy(cf, cfg)
        except Exception as exc:
            print(f"     [skip] could not read '{cf}': {exc}")
            continue
        fmask, fint, fpos = check_peaks_in_component(
            x_s, y_s, e_s, peak_centers, cfg.tolerance, cfg.snr_thresh,
            cfg.bg_window, cfg.presence_baseline_corrected,
            method=cfg.baseline_method, poly_order=cfg.baseline_poly_order,
            als_lambda=cfg.baseline_als_lambda, als_p=cfg.baseline_als_p)
        component_results.append((cf, fmask, fint, fpos))
        if verbose:
            print(f"    {os.path.basename(cf)}: {sum(fmask)}/{len(peak_centers)} present")
        plot_component(x_s, y_s, e_s, peak_centers, fmask, fpos, cfg, cf)

    # outputs
    fit_data = (x, y, yerr, amplitudes, peak_centers)
    params   = gather_parameters(cfg, peak_centers, picker_settings, stats, had_errors,
                                 component_files=[cf for cf, _, _, _ in component_results],
                                 width_scales=width_scales, fwhms=fwhms)
    xlsx = csvs = None
    if cfg.write_excel:
        xlsx = save_excel(cfg, peak_centers, amplitudes, component_results,
                          fit_data=fit_data, stats=stats, params=params,
                          sigmas=sigmas, gammas=gammas)
    if cfg.write_csv:
        csvs = save_csv(cfg, peak_centers, amplitudes, component_results,
                        fit_data=fit_data, stats=stats, params=params,
                        sigmas=sigmas, gammas=gammas)

    return {"amplitudes": amplitudes, "stats": stats,
            "component_results": component_results,
            "params": params, "xlsx": xlsx, "csv": csvs,
            "sigmas": sigmas, "gammas": gammas, "fwhms": fwhms,
            "width_scales": width_scales, "hint": hint}


def run_headless(cfg, verbose=True):
    """Headless run: load reference, take peaks from cfg.peaks or the automatic
    search, then run the full analysis. No window is opened."""
    import matplotlib
    matplotlib.use("Agg", force=True)

    problems = cfg.validate()
    if problems:
        raise ValueError("Invalid configuration: " + "; ".join(problems))
    if not os.path.isfile(cfg.reference_file):
        raise FileNotFoundError(f"Reference file '{cfg.reference_file}' not found.")

    x, y, yerr, had_errors = load_xy(cfg.reference_file, cfg)
    if verbose:
        print(f"  Loaded {len(x)} points from {cfg.reference_file} "
              f"(x: {x[0]:.3g} .. {x[-1]:.3g}); "
              f"errors {'from file' if had_errors else 'estimated'}.")

    if cfg.peaks:
        peak_centers = np.array(sorted(float(p) for p in cfg.peaks))
        picker_settings = None
        if verbose:
            print(f"  Using {len(peak_centers)} peak position(s) from config.")
    else:
        mask = x_range_mask(x, cfg.x_min, cfg.x_max)
        idx  = search_peaks(y, yerr, x, mask, cfg.prominence_init,
                            cfg.distance_init, cfg.min_snr_init, cfg.bg_window)
        peak_centers = x[idx]
        picker_settings = {"prominence": cfg.prominence_init,
                           "distance": cfg.distance_init,
                           "min_snr": cfg.min_snr_init,
                           "bg_window": cfg.bg_window}
        if verbose:
            print(f"  Automatic search found {len(peak_centers)} peak(s).")

    if len(peak_centers) == 0:
        print("  No peaks to fit — aborting.")
        return None
    return run_analysis(cfg, x, y, yerr, peak_centers, picker_settings,
                        had_errors, verbose=verbose)
