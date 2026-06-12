"""
Peak search and weighted non-negative least-squares fit.

The fit at fixed positions and widths is *linear* in the amplitudes, so it
reduces to a weighted NNLS problem (Lawson & Hanson 1995): non-negativity is
the right physical constraint (no negative intensities), and a peak with no
support in the data gets exactly A_k=0 rather than a noisy small value.
Standard errors come from the analytic covariance (B^T W^2 B)^-1; the
`statistical` mode reports them unscaled, the `scaled` mode multiplies by
sqrt(chi2_red) for the curve_fit-style absolute_sigma=False convention.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls, least_squares
from scipy.signal import find_peaks

from .background import subtract_background, subtract_background_cfg
from .profiles import profile_shape


def search_peaks(y, yerr, x, mask, prominence, distance, min_snr, bg_window):
    r"""Automatic peak search on the background-corrected data.

    Local maxima are located with :func:`scipy.signal.find_peaks` [7]_ on the
    background-corrected signal (so a sloping baseline does not hide weak
    peaks), filtered by topographic *prominence* and a minimum *distance*, and
    then by a signal-to-noise criterion evaluated on the **raw** data:

    .. math:: \frac{y_i}{\sigma_i} \ge \texttt{min\_snr}.

    Parameters
    ----------
    y, yerr : numpy.ndarray
        Raw intensities and their 1-sigma uncertainties.
    x : numpy.ndarray
        Abscissa (only its length / masking is used here).
    mask : numpy.ndarray of bool
        Restricts the search to the chosen x range.
    prominence : float
        Minimum topographic prominence of a peak (``find_peaks`` argument).
    distance : int
        Minimum separation between peaks in samples.
    min_snr : float
        Minimum signal-to-noise ratio; ``0`` disables the SNR filter.
    bg_window : int
        Window for :func:`subtract_background`.

    Returns
    -------
    numpy.ndarray of int
        Indices of the accepted peaks into the full (unmasked) array.

    References
    ----------
    .. [7] P. Virtanen et al., "SciPy 1.0", Nature Methods 17 (2020) 261-272.
    """
    y_sub    = y[mask]
    yerr_sub = yerr[mask]
    y_corr   = subtract_background(y_sub, bg_window)

    local_idx, _ = find_peaks(y_corr, prominence=prominence,
                              distance=max(1, int(distance)))
    if min_snr > 0 and len(local_idx) > 0:
        snr = y_sub[local_idx] / yerr_sub[local_idx]
        local_idx = local_idx[snr >= min_snr]

    return np.where(mask)[0][local_idx]


def make_fit_mask(x, cfg):
    """Fit region = search window widened by 3*(wg+wl) so profile wings are
    captured but distant structure cannot bias the amplitudes."""
    margin = 3.0 * (cfg.wg + cfg.wl)
    lo = (cfg.x_min - margin) if cfg.x_min is not None else x[0]
    hi = (cfg.x_max + margin) if cfg.x_max is not None else x[-1]
    return (x >= lo) & (x <= hi)


def _fwhm_const():
    """2*sqrt(2*ln2): Gaussian sigma -> FWHM factor."""
    return 2.0 * np.sqrt(2.0 * np.log(2.0))


def fwhm_from_sigma_gamma(sigma, gamma, profile):
    """Effective FWHM of a profile from its sigma/gamma (scalar or array).

    Uses the Olivero-Longbothum approximation for the Voigt; the pure limits
    return the Gaussian (``2.3548*sigma``) or Lorentzian (``2*gamma``) FWHM.
    """
    sigma = np.asarray(sigma, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    fG = sigma * _fwhm_const()
    fL = 2.0 * gamma
    p = str(profile).lower()
    if p in ("gauss", "gaussian"):
        return fG
    if p in ("lorentz", "lorentzian"):
        return fL
    return 0.5346 * fL + np.sqrt(0.2166 * fL ** 2 + fG ** 2)


def per_peak_widths(cfg, n, width_scales=None):
    """Return per-peak ``(sigmas, gammas)`` arrays of length ``n``.

    With ``width_scales is None`` every peak uses the instrumental
    ``cfg.sigma()``/``cfg.gamma()`` (the fixed-width default, unchanged). With a
    per-peak scale vector the widths are scaled according to ``cfg.width_mode``:
    ``"fwhm"`` scales the whole Voigt (both sigma and gamma, so the effective
    FWHM scales with the factor), ``"sigma"`` scales only the Gaussian part
    (inhomogeneous broadening; the Lorentzian lifetime width stays fixed).
    """
    s0, g0 = cfg.sigma(), cfg.gamma()
    if width_scales is None:
        return np.full(n, s0), np.full(n, g0)
    w = np.asarray(width_scales, dtype=float)
    if str(cfg.width_mode).lower() == "sigma":
        return s0 * w, np.full(n, g0)
    return s0 * w, g0 * w


def compute_model(x, amplitudes, centers, sigma, gamma, profile):
    """Total profile sum and individual components.

    ``sigma`` and ``gamma`` may be scalars (one width for all peaks, the
    default) or per-peak arrays of the same length as ``centers``.
    """
    n = len(centers)
    sig = np.broadcast_to(np.asarray(sigma, dtype=float), (n,))
    gam = np.broadcast_to(np.asarray(gamma, dtype=float), (n,))
    components = [amp * profile_shape(x, c, s, g, profile)
                  for amp, c, s, g in zip(amplitudes, centers, sig, gam)]
    total = np.sum(components, axis=0) if components else np.zeros_like(x)
    return total, components


def fit_amplitudes(x, y, yerr, centers, cfg, fit_mask=None, sigmas=None, gammas=None):
    r"""Weighted non-negative least-squares fit of the peak amplitudes.

    With the peak positions and widths held fixed, the model is *linear* in the
    amplitudes :math:`A_k \ge 0`, so the fit reduces to a non-negative least
    squares (NNLS) problem on the background-corrected data:

    .. math::

        \min_{\mathbf{A}\ge 0}\;
        \big\lVert W\,(B\,\mathbf{A} - \mathbf{y}_\text{corr})\big\rVert_2^2,
        \qquad B_{ik} = P_k(x_i),\quad W = \operatorname{diag}(1/\sigma_i),

    where :math:`P_k` is the (area-normalised) profile of peak :math:`k`. The
    non-negativity constraint is physically motivated (intensities cannot be
    negative) and is solved with the active-set algorithm of Lawson & Hanson
    [3]_ as implemented in :func:`scipy.optimize.nnls` [7]_. Because the
    amplitudes are areas of unit-area profiles, ``A_k`` is the integrated
    intensity of peak :math:`k`.

    Parameters
    ----------
    x, y, yerr : numpy.ndarray
        Abscissa, raw intensities and 1-sigma uncertainties.
    centers : array_like
        Peak positions :math:`x_0` (fixed).
    cfg : Config
        Provides the profile, the widths (via ``sigma()``/``gamma()``) and the
        background window.
    fit_mask : numpy.ndarray of bool, optional
        Restricts the least-squares region (see :func:`make_fit_mask`). If
        omitted, the whole spectrum is used.
    sigmas, gammas : numpy.ndarray, optional
        Per-peak widths (same length as ``centers``). If omitted, the single
        instrumental ``cfg.sigma()``/``cfg.gamma()`` is used for all peaks (the
        default fixed-width behaviour).

    Returns
    -------
    numpy.ndarray
        The fitted amplitudes (one per centre); empty if ``centers`` is empty.

    References
    ----------
    .. [3] C. L. Lawson, R. J. Hanson, "Solving Least Squares Problems",
       Prentice-Hall (1974) / SIAM (1995), Chapter 23.
    .. [7] P. Virtanen et al., "SciPy 1.0", Nature Methods 17 (2020) 261-272.
    """
    if len(centers) == 0:
        return np.array([])
    if sigmas is None or gammas is None:
        s0, g0 = cfg.sigma(), cfg.gamma()
        sigmas = np.full(len(centers), s0)
        gammas = np.full(len(centers), g0)
    y_corr = subtract_background_cfg(y, cfg)
    basis  = np.column_stack(
        [profile_shape(x, c, s, g, cfg.profile)
         for c, s, g in zip(centers, sigmas, gammas)])
    if fit_mask is None:
        fit_mask = np.ones(len(x), dtype=bool)
    W = 1.0 / yerr
    A = basis[fit_mask] * W[fit_mask, np.newaxis]
    b = (y_corr * W)[fit_mask]
    amplitudes, _ = nnls(A, b)
    return amplitudes


def fit_statistics(x, y, yerr, amplitudes, centers, cfg, fit_mask=None,
                   force_scaled=False, sigmas=None, gammas=None,
                   n_extra_params=0):
    r"""Goodness-of-fit metrics and amplitude standard errors.

    Evaluated on the background-corrected data over ``fit_mask``. With residuals
    :math:`r_i = y_{\text{corr},i} - \sum_k A_k P_k(x_i)` and weights
    :math:`1/\sigma_i`,

    .. math::

        \chi^2 = \sum_i (r_i/\sigma_i)^2, \quad
        \chi^2_\nu = \frac{\chi^2}{N - N_\text{active} - N_\text{extra}}, \quad
        R^2 = 1 - \frac{\sum_i r_i^2}{\sum_i (y_i-\bar y)^2}.

    The amplitude covariance for the active set (:math:`A_k > 0`) is
    :math:`\operatorname{Cov} = (B_\text{act}^\top W^2 B_\text{act})^{-1}` and
    the statistical 1-sigma errors are its square-rooted diagonal [4]_ [5]_.

    Two reporting conventions are available (``error_mode``):

    * ``'statistical'`` -- report the covariance error unscaled. Correct when
      ``yerr`` is a genuine 1-sigma (e.g. photon counting): the random error is
      reported as is and chi2_red / R^2 remain *separate* model-adequacy
      diagnostics. A :math:`\chi^2_\nu > 1` then signals systematic model error
      (one fixed width for all peaks, approximate background, missing peaks),
      which should be addressed in the model rather than folded into the random
      error bar.
    * ``'scaled'`` -- multiply the error by :math:`\sqrt{\chi^2_\nu}`, the
      convention of :func:`scipy.optimize.curve_fit` with
      ``absolute_sigma=False`` [8]_. Appropriate when the errors are only
      relative weights of unknown absolute scale. ``force_scaled`` selects this
      automatically when the errors were merely estimated from the data.

    Parameters
    ----------
    x, y, yerr : numpy.ndarray
        Abscissa, raw intensities and 1-sigma uncertainties.
    amplitudes : numpy.ndarray
        Fitted amplitudes from :func:`fit_amplitudes`.
    centers : array_like
        Peak positions.
    cfg : Config
        Profile, widths, background window and ``error_mode``.
    fit_mask : numpy.ndarray of bool, optional
        Fit region; whole spectrum if omitted.
    force_scaled : bool, optional
        Force ``error_mode='scaled'`` (used when ``yerr`` was estimated).
    sigmas, gammas : numpy.ndarray, optional
        Per-peak widths; the single instrumental width is used for all peaks if
        omitted (the default).
    n_extra_params : int, optional
        Number of additional free parameters beyond the amplitudes (e.g. refined
        positions and/or widths). Subtracted from the degrees of freedom so the
        reduced chi-square reflects them; defaults to ``0``.

    Returns
    -------
    dict
        Keys: ``chi2``, ``chi2_red``, ``r2``, ``dof``, ``n_active``,
        ``n_points``, ``stderr`` (reported 1-sigma per amplitude, NaN for
        inactive peaks), ``stderr_stat`` (the unscaled statistical error),
        ``error_scale`` and ``error_mode``.

    Notes
    -----
    NNLS constrains :math:`A_k \ge 0`. For a peak pinned at the boundary
    (:math:`A_k = 0`) the linear covariance does not apply; its error is
    reported as ``NaN``.

    References
    ----------
    .. [4] P. R. Bevington, D. K. Robinson, "Data Reduction and Error Analysis
       for the Physical Sciences", 3rd ed., McGraw-Hill (2003).
    .. [5] W. H. Press et al., "Numerical Recipes", 3rd ed., CUP (2007), Ch. 15.
    .. [8] scipy.optimize.curve_fit documentation, parameter ``absolute_sigma``.
    """
    n = len(centers)
    eff_mode = "scaled" if force_scaled else str(cfg.error_mode).lower()
    if n == 0:
        z = np.full(0, np.nan)
        return {"chi2": float("nan"), "chi2_red": float("nan"),
                "r2": float("nan"), "dof": 0, "n_active": 0, "n_points": 0,
                "stderr": z, "stderr_stat": z, "error_scale": 1.0,
                "error_mode": eff_mode}

    if fit_mask is None:
        fit_mask = np.ones(len(x), dtype=bool)
    if sigmas is None or gammas is None:
        s0, g0 = cfg.sigma(), cfg.gamma()
        sigmas = np.full(n, s0)
        gammas = np.full(n, g0)
    y_corr = subtract_background_cfg(y, cfg)
    basis  = np.column_stack(
        [profile_shape(x, c, s, g, cfg.profile)
         for c, s, g in zip(centers, sigmas, gammas)])

    yf = y_corr[fit_mask]
    ef = yerr[fit_mask]
    Bf = basis[fit_mask]
    w  = 1.0 / ef

    model = Bf @ amplitudes
    resid = yf - model

    chi2     = float(np.sum((resid * w) ** 2))
    n_active = int(np.sum(amplitudes > 0))
    n_points = int(len(yf))
    dof      = max(n_points - n_active - int(max(n_extra_params, 0)), 1)
    chi2_red = chi2 / dof

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((yf - np.mean(yf)) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    active = np.where(amplitudes > 0)[0]
    stderr_stat = np.full(n, np.nan)
    if len(active) > 0:
        Ba = Bf[:, active] * w[:, np.newaxis]
        try:
            cov = np.linalg.inv(Ba.T @ Ba)
            stderr_stat[active] = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        except np.linalg.LinAlgError:
            pass

    if eff_mode == "scaled":
        scale = np.sqrt(chi2_red) if np.isfinite(chi2_red) and chi2_red > 0 else 1.0
    else:
        scale = 1.0
    stderr = stderr_stat * scale

    return {"chi2": chi2, "chi2_red": chi2_red, "r2": r2, "dof": dof,
            "n_active": n_active, "n_points": n_points,
            "stderr": stderr, "stderr_stat": stderr_stat,
            "error_scale": float(scale), "error_mode": eff_mode}



def refine_peaks(x, y, yerr, centers, cfg, fit_mask=None):
    r"""Optionally refine the peak *positions* by a bounded nonlinear fit.

    With the global widths held fixed the model is still nonlinear in the
    centres. Using variable projection -- the amplitudes are obtained by NNLS
    for any trial set of centres -- the centres are refined with a bounded
    least-squares solve (``scipy.optimize.least_squares``), each centre
    constrained to :math:`\pm` ``cfg.refine_window`` of its start value so it
    cannot wander onto a neighbouring peak. Amplitudes stay non-negative
    throughout. Returns the (possibly) refined centres in the original order;
    a no-op returning ``centers`` unchanged if ``cfg.refine`` is false, if there
    are no peaks, or if the solve fails.
    """
    centers = np.asarray(centers, dtype=float)
    if not getattr(cfg, "refine", False) or len(centers) == 0:
        return centers
    sigma, gamma = cfg.sigma(), cfg.gamma()
    y_corr = subtract_background_cfg(y, cfg)
    if fit_mask is None:
        fit_mask = np.ones(len(x), dtype=bool)
    xf = x[fit_mask]
    wf = 1.0 / yerr[fit_mask]
    target = y_corr[fit_mask] * wf

    def residual(c):
        B = np.column_stack(
            [profile_shape(xf, ci, sigma, gamma, cfg.profile) for ci in c])
        amp, _ = nnls(B * wf[:, np.newaxis], target)
        return (B @ amp) * wf - target

    lo = centers - cfg.refine_window
    hi = centers + cfg.refine_window
    try:
        res = least_squares(residual, centers, bounds=(lo, hi),
                            method="trf", max_nfev=200)
        return np.asarray(res.x, dtype=float)
    except Exception:
        return centers


def refine_widths(x, y, yerr, centers, cfg, fit_mask=None):
    r"""Optionally refine **per-peak widths** by a bounded nonlinear fit.

    Parallel to :func:`refine_peaks`, but the free parameters are one width
    *scale* per peak (relative to the instrumental width). Using variable
    projection -- the amplitudes are obtained by NNLS for any trial set of
    widths -- the scales are refined with a bounded least-squares solve
    (:func:`scipy.optimize.least_squares`), each scale constrained to
    ``[cfg.width_min_factor, cfg.width_max_factor]``. ``cfg.width_mode`` selects
    whether the whole Voigt is scaled (``"fwhm"``) or only the Gaussian part
    (``"sigma"``). Amplitudes stay non-negative throughout.

    Returns a per-peak scale vector (all ones when refinement is off, there are
    no peaks, or the solve fails) -- multiply the instrumental widths by it (see
    :func:`per_peak_widths`) to obtain the per-peak ``(sigma, gamma)``.
    """
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    if not getattr(cfg, "refine_widths", False) or n == 0:
        return np.ones(n)
    lo_f = float(cfg.width_min_factor)
    hi_f = float(cfg.width_max_factor)
    if not (hi_f > lo_f > 0):
        return np.ones(n)
    y_corr = subtract_background_cfg(y, cfg)
    if fit_mask is None:
        fit_mask = np.ones(len(x), dtype=bool)
    xf = x[fit_mask]
    wf = 1.0 / yerr[fit_mask]
    target = y_corr[fit_mask] * wf

    def residual(scales):
        sig, gam = per_peak_widths(cfg, n, scales)
        B = np.column_stack(
            [profile_shape(xf, c, s, g, cfg.profile)
             for c, s, g in zip(centers, sig, gam)])
        amp, _ = nnls(B * wf[:, np.newaxis], target)
        return (B @ amp) * wf - target

    start = np.clip(np.ones(n), lo_f, hi_f)
    lo = np.full(n, lo_f)
    hi = np.full(n, hi_f)
    try:
        res = least_squares(residual, start, bounds=(lo, hi),
                            method="trf", max_nfev=300)
        return np.clip(np.asarray(res.x, dtype=float), lo_f, hi_f)
    except Exception:
        return np.ones(n)
