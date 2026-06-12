"""
Plot helpers and figure producers.

Two figures per run:
  * `plot_fit_result` — the fit window with raw + background-corrected data,
    sum, components and a residual panel below; saved as `*_fit.png`.
  * `plot_component` — one marked plot per component data set, with present
    peaks green and absent peaks red; saved as `<comp>_*_peaks.png`.

Output paths run through `_outpath`, which honours `cfg.output_dir` if set and
otherwise drops the file next to the reference data.
"""
from __future__ import annotations

import os

import numpy as np

from .background import subtract_background_cfg
from .fit import compute_model


def _xlabel(cfg):
    return f"{cfg.x_label} ({cfg.x_unit})" if cfg.x_unit else cfg.x_label


def _apply_xlim(ax, cfg):
    if cfg.x_min is not None or cfg.x_max is not None:
        lo = cfg.x_min if cfg.x_min is not None else ax.get_xlim()[0]
        hi = cfg.x_max if cfg.x_max is not None else ax.get_xlim()[1]
        ax.set_xlim(lo, hi)


def _outpath(cfg, filename, suffix):
    """Return the absolute path for an output file.

    The base name is the stem of ``filename`` plus the active search range
    suffix (e.g. ``_100_200``) and ``suffix`` (e.g. ``"_fit.png"``). The
    directory is ``cfg.output_dir`` when set, otherwise the directory of
    ``filename`` -- so by default outputs land next to the reference file.
    """
    base = os.path.splitext(os.path.basename(filename))[0] + cfg.range_suffix() + suffix
    out_dir = cfg.output_dir.strip() if cfg.output_dir else ""
    if not out_dir:
        out_dir = os.path.dirname(os.path.abspath(filename))
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, base)


def plot_fit_result(x, y, yerr, amplitudes, centers, cfg, filename,
                    stats=None, fig=None, sigmas=None, gammas=None, fwhms=None):
    """Fit plot: raw + BG-corrected data, sum, components and residuals.
    If `fig` is given it is drawn into (for the GUI); otherwise a new figure is
    created, saved to '<stem><suffix>_fit.png' and returned.

    `sigmas`/`gammas` give per-peak widths (default: the single instrumental
    width for all peaks). When `fwhms` is provided the per-peak FWHM is appended
    to each component's legend label (used when width refinement is active)."""
    import matplotlib.pyplot as plt
    if sigmas is None or gammas is None:
        sigmas, gammas = cfg.sigma(), cfg.gamma()
    y_corr = subtract_background_cfg(y, cfg)
    total, components = compute_model(x, amplitudes, centers, sigmas, gammas, cfg.profile)
    residuals = y_corr - total

    created = False
    if fig is None:
        fig = plt.figure(figsize=(13, 8))
        created = True
    else:
        fig.clear()
    ax1 = fig.add_subplot(4, 1, (1, 3))
    ax2 = fig.add_subplot(4, 1, 4, sharex=ax1)

    title = f"Multi-{str(cfg.profile).title()} fit: {os.path.basename(filename)}"
    if stats is not None:
        title += (f"   |   chi2_red = {stats['chi2_red']:.3g}"
                  f"   R2 = {stats['r2']:.4f}"
                  f"   ({stats['n_active']}/{len(centers)} active)")
    fig.suptitle(title, fontsize=11)

    ax1.errorbar(x, y, yerr=yerr, fmt="o", ms=3, color="#aaaaaa",
                 ecolor="#cccccc", capsize=1.5, lw=0.5, label="Raw data", zorder=2)
    ax1.errorbar(x, y_corr, yerr=yerr, fmt="o", ms=4, color="black",
                 ecolor="#888888", capsize=2, lw=0.8,
                 label="BG-corrected (fitted)", zorder=3)
    ax1.plot(x, total, "r-", lw=2, label="Sum", zorder=4)

    import matplotlib
    cmap = matplotlib.colormaps["tab20"]
    n_peaks = len(centers)
    fwhm_arr = np.atleast_1d(fwhms) if fwhms is not None else None
    for i, (comp, c, amp) in enumerate(zip(components, centers, amplitudes)):
        label = f"{c:.1f} (A={amp:.0f})"
        if fwhm_arr is not None and i < len(fwhm_arr):
            label = f"{c:.1f} (A={amp:.0f}, FWHM={fwhm_arr[i]:.2f})"
        ax1.plot(x, comp, "-", color=cmap(i / max(n_peaks, 1)), lw=1.2, alpha=0.85,
                 label=label)
    ax1.set_ylabel(cfg.y_label)
    ax1.legend(fontsize=7, ncol=max(1, n_peaks // 8 + 1), loc="upper left")
    _apply_xlim(ax1, cfg)

    x_lo = cfg.x_min if cfg.x_min is not None else x[0]
    x_hi = cfg.x_max if cfg.x_max is not None else x[-1]
    vis  = (x >= x_lo) & (x <= x_hi)
    if np.any(vis):
        y_vis  = np.concatenate([y_corr[vis], total[vis]])
        margin = 0.08 * np.ptp(y_vis) if np.ptp(y_vis) > 0 else 1.0
        ax1.set_ylim(min(y_vis) - margin, max(y_vis) + margin)

    ax2.errorbar(x, residuals, yerr=yerr, fmt="o", ms=3, color="#555555",
                 ecolor="#aaaaaa", capsize=2, lw=0.6)
    ax2.axhline(0, color="red", lw=1, ls="--")
    ax2.set_ylabel("Residuals")
    ax2.set_xlabel(_xlabel(cfg))
    if np.any(vis):
        rv = residuals[vis]
        rs = np.ptp(rv) if np.ptp(rv) > 0 else 1.0
        ax2.set_ylim(rv.min() - 0.15 * rs, rv.max() + 0.15 * rs)

    fig.tight_layout()
    outname = None
    if created:
        outname = _outpath(cfg, filename, "_fit.png")
        fig.savefig(outname, dpi=cfg.plot_dpi, bbox_inches="tight")
        print(f"    -> fit plot saved: {outname}")
        plt.close(fig)
    return outname


def plot_component(x, y, yerr, peak_centers, found_mask, found_pos, cfg, filename):
    """Marked plot of one component data set; saved to '<stem><suffix>_peaks.png'."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_title(f"Component: {os.path.basename(filename)}", fontsize=11)
    ax.errorbar(x, y, yerr=yerr, fmt="o", ms=4, color="black",
                ecolor="#888888", capsize=2, lw=0.8, label="Data")

    x_lo = cfg.x_min if cfg.x_min is not None else x[0]
    x_hi = cfg.x_max if cfg.x_max is not None else x[-1]
    vis  = (x >= x_lo) & (x <= x_hi)
    if np.any(vis):
        y_vis  = y[vis]
        y_lo, y_hi = min(y_vis), max(y_vis)
        margin = 0.08 * (y_hi - y_lo) if y_hi != y_lo else 1.0
        ax.set_ylim(y_lo - margin, y_hi + margin)
    else:
        y_hi = max(y)

    for i, (center, found, pos) in enumerate(zip(peak_centers, found_mask, found_pos)):
        color = "#22aa44" if found else "#cc3333"
        sym   = "v" if found else "x"
        ax.axvline(center, color=color, lw=1.0, alpha=0.4, ls="--")
        if found and not np.isnan(pos):
            ax.axvline(pos, color=color, lw=1.5, alpha=0.9, ls="-")
            label_x = pos
        else:
            label_x = center
        y_off = y_hi * 0.96 if i % 2 == 0 else y_hi * 0.88
        ax.text(label_x, y_off, f"{label_x:.1f}\n{sym}", ha="center",
                fontsize=7, color=color, fontweight="bold", va="top")

    ax.legend(handles=[
        Line2D([0], [0], color="#22aa44", lw=1.5, ls="-", label="present (measured)"),
        Line2D([0], [0], color="#22aa44", lw=1.0, ls="--", label="present (reference)"),
        Line2D([0], [0], color="#cc3333", lw=1.0, ls="--", label="absent"),
    ], fontsize=8)
    ax.set_xlabel(_xlabel(cfg))
    ax.set_ylabel(cfg.y_label)
    _apply_xlim(ax, cfg)

    fig.tight_layout()
    outname = _outpath(cfg, filename, "_peaks.png")
    fig.savefig(outname, dpi=cfg.plot_dpi, bbox_inches="tight")
    print(f"    -> component plot saved: {outname}")
    plt.close(fig)
    return outname
