"""
Background subtraction and search-window masks.

`subtract_background` estimates a baseline and subtracts it (clipped at zero).
Three methods are available via ``method``:

  * ``"rolling_min"`` (default) -- a rolling minimum over ``window`` points; fast
    and robust for narrow peaks on a broad pedestal. This is the original
    behaviour and the default everywhere.
  * ``"polynomial"``            -- an iterative low-order polynomial (ModPoly
    style) refit to the running lower envelope; good for smooth, gently curved
    baselines.
  * ``"als"``                   -- asymmetric least squares (Eilers 2005); a
    smooth baseline pulled toward points below it and away from peaks above.

The configured method is applied identically to the data fed to the amplitude
fit, the goodness-of-fit statistics, the component presence check and the
fit plot / outputs, so all of them see one consistent baseline. (The automatic
peak *search* and the GUI live preview always use the rolling minimum, which is
only a candidate-detection / display aid.)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d


def _poly_baseline(y, order, niter=24):
    """Iterative polynomial (ModPoly) baseline: refit to the running lower
    envelope so peaks are progressively excluded; the last fit is the baseline."""
    n = len(y)
    if n <= order + 1:
        return np.full(n, float(np.min(y)) if n else 0.0)
    t = np.linspace(-1.0, 1.0, n)
    z = np.asarray(y, dtype=float).copy()
    fit = z
    for _ in range(niter):
        coef = np.polyfit(t, z, order)
        fit = np.polyval(coef, t)
        z = np.minimum(z, fit)
    return fit


def _als_baseline(y, lam=1.0e5, p=0.01, niter=10):
    """Asymmetric-least-squares baseline (Eilers 2005): a smooth curve weighted
    1-p toward points below it and p toward points above (the peaks)."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    y = np.asarray(y, dtype=float)
    L = len(y)
    if L < 3:
        return np.full(L, float(np.min(y)) if L else 0.0)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    DTD = lam * (D @ D.transpose())
    w = np.ones(L)
    z = y.copy()
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve((W + DTD).tocsc(), w * y)
        w = p * (y > z) + (1.0 - p) * (y < z)
    return z


def subtract_background(y, window, method="rolling_min", poly_order=3,
                        als_lambda=1.0e5, als_p=0.01):
    r"""Subtract an estimated baseline and clip the result at zero.

    .. math:: y_\text{corr}(i) = \max\big(y(i) - \text{baseline}(i),\; 0\big).

    Parameters
    ----------
    y : numpy.ndarray
        Intensity values.
    window : int
        Rolling-minimum width in samples (``method="rolling_min"``).
        ``window <= 0`` disables the rolling-minimum correction and returns a
        copy of ``y``.
    method : str
        ``"rolling_min"`` (default), ``"polynomial"`` or ``"als"``.
    poly_order : int
        Polynomial order for ``method="polynomial"``.
    als_lambda, als_p : float
        Smoothness and asymmetry for ``method="als"``.

    Returns
    -------
    numpy.ndarray
        Background-corrected, non-negative intensities.
    """
    y = np.asarray(y, dtype=float)
    m = str(method).lower()
    if m in ("rolling_min", "rolling", "min", ""):
        if window <= 0:
            return y.copy()
        bg = minimum_filter1d(y, size=int(window), mode="reflect")
    elif m in ("polynomial", "poly"):
        bg = _poly_baseline(y, int(poly_order))
    elif m in ("als", "arpls", "asymmetric"):
        bg = _als_baseline(y, lam=float(als_lambda), p=float(als_p))
    else:
        bg = minimum_filter1d(y, size=max(1, int(window)), mode="reflect")
    return np.maximum(y - bg, 0.0)


def subtract_background_cfg(y, cfg):
    """Convenience wrapper: subtract the baseline configured in ``cfg``
    (method and parameters), so every quantitative step uses one baseline."""
    return subtract_background(
        y, cfg.bg_window,
        method=getattr(cfg, "baseline_method", "rolling_min"),
        poly_order=getattr(cfg, "baseline_poly_order", 3),
        als_lambda=getattr(cfg, "baseline_als_lambda", 1.0e5),
        als_p=getattr(cfg, "baseline_als_p", 0.01),
    )


def x_range_mask(x, x_min, x_max):
    """Boolean mask for the desired x range."""
    mask = np.ones(len(x), dtype=bool)
    if x_min is not None:
        mask &= x >= x_min
    if x_max is not None:
        mask &= x <= x_max
    return mask
