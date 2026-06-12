"""
Peak-presence check across the component data sets.

For each reference peak the routine decides whether the component carries a
peak inside `± tolerance`. The search has three stages:
  (1) strict local maximum in both corrected AND raw data (the raw condition
      prevents a point merely lifted by the background subtraction from
      passing as a peak);
  (2) shoulder fallback via inflection points of dy/dx (only if stage 1 found
      nothing);
  (3) duplicate resolution if two close references claim the same data point
      (the nearer reference wins; the other searches the next free maximum).
Acceptance is decided by an SNR threshold; by default this is taken on the
background-corrected signal (peak height above the background, which is the
physically meaningful measure).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .background import subtract_background


def check_peaks_in_component(x_sub, y_sub, yerr_sub, peak_centers,
                             tolerance, snr_thresh, bg_window,
                             baseline_corrected=True, method="rolling_min",
                             poly_order=3, als_lambda=1.0e5, als_p=0.01):
    r"""Decide, for each reference peak, whether a component data set contains it.

    A reference peak is considered *present* if, within :math:`\pm` ``tolerance``
    of its position, the component shows either a genuine local maximum or a
    significant shoulder that also clears the SNR threshold. The search has
    three stages:

    1. **Strict maximum.** A local maximum of the background-corrected data that
       is *also* a local maximum of the raw data (rejecting points lifted only
       by the background subtraction), located with
       :func:`scipy.signal.find_peaks` [7]_ on the window extended by
       ``EDGE_PAD`` samples so boundary peaks are not missed. Among valid
       candidates the one with the highest corrected intensity wins.
    2. **Shoulder fallback.** If stage 1 finds nothing, inflection points of
       :math:`\mathrm{d}y/\mathrm{d}x` (via :func:`numpy.gradient`) are tested;
       a shoulder must reach at least ``SHOULDER_MIN_FRAC`` of the window
       maximum and must not sit in a valley.
    3. **Duplicate resolution.** If two references claim the same sample, the
       closer reference keeps it and the other searches for the next free local
       maximum in its window.

    A candidate is finally accepted only if its signal-to-noise ratio clears
    ``snr_thresh`` and :math:`y_{\text{corr},i} > 0`. The SNR is evaluated on the
    background-corrected signal when ``baseline_corrected`` is true (default),
    i.e. :math:`y_{\text{corr},i}/\sigma_i` -- the peak height *above* the
    background, which is the physically meaningful measure of presence. With
    ``baseline_corrected=False`` the legacy raw-signal ratio
    :math:`y_i/\sigma_i` is used instead.
    No re-fitting is done here: only presence (and the measured position and
    raw intensity) is determined, not an amplitude.

    Parameters
    ----------
    x_sub, y_sub, yerr_sub : numpy.ndarray
        Component abscissa, intensities and 1-sigma uncertainties.
    peak_centers : array_like
        Reference peak positions to look for.
    tolerance : float
        Half-width of the search window around each reference (x units).
    snr_thresh : float
        Minimum signal-to-noise ratio to accept a candidate.
    bg_window : int
        Background window for :func:`subtract_background`.
    baseline_corrected : bool, optional
        If True (default), test the SNR on the background-corrected signal
        (peak height above background); if False, on the raw signal.

    Returns
    -------
    found_mask : list of bool
        ``True`` where the peak is present.
    found_intensities : list of float
        Raw intensity at the accepted sample (``NaN`` if absent).
    found_positions : list of float
        Abscissa of the accepted sample (``NaN`` if absent).

    References
    ----------
    .. [7] P. Virtanen et al., "SciPy 1.0", Nature Methods 17 (2020) 261-272.
    """
    SHOULDER_MIN_FRAC = 0.50
    EDGE_PAD          = 2

    y_corr_sub = subtract_background(y_sub, bg_window, method, poly_order,
                                     als_lambda, als_p)
    # Signal used for the SNR test: the background-corrected signal measures the
    # peak height *above* the background (physically the right presence measure);
    # the raw signal is the legacy choice.
    y_snr = y_corr_sub if baseline_corrected else y_sub

    found_mask, found_intensities, found_positions, _imax_list = [], [], [], []

    for center in peak_centers:
        window_mask = np.abs(x_sub - center) <= tolerance
        window_idx  = np.where(window_mask)[0]

        if len(window_idx) == 0:
            found_mask.append(False)
            found_intensities.append(float("nan"))
            found_positions.append(float("nan"))
            _imax_list.append(-1)
            continue

        y_win_corr = y_corr_sub[window_idx]
        y_win_raw  = y_sub[window_idx]
        imax       = None

        # Step 1: local maximum in extended window
        ext_lo     = max(0, window_idx[0] - EDGE_PAD)
        ext_hi     = min(len(x_sub), window_idx[-1] + EDGE_PAD + 1)
        ext_idx_ar = np.arange(ext_lo, ext_hi)
        y_ext_corr = y_corr_sub[ext_idx_ar]
        y_ext_raw  = y_sub[ext_idx_ar]

        ext_peaks, _ = find_peaks(y_ext_corr)
        for edge_ep in [0, len(ext_idx_ar) - 1]:
            gi = ext_idx_ar[edge_ep]
            if 0 < gi < len(y_sub) - 1:
                if (y_corr_sub[gi] > y_corr_sub[gi - 1] and
                        y_corr_sub[gi] > y_corr_sub[gi + 1] and
                        y_sub[gi]      >= y_sub[gi - 1] and
                        y_sub[gi]      >= y_sub[gi + 1]):
                    if edge_ep not in ext_peaks.tolist():
                        ext_peaks = np.append(ext_peaks, edge_ep)

        valid_ext = [ep for ep in ext_peaks
                     if 0 < ep < len(y_ext_raw) - 1
                     and y_ext_raw[ep] >= y_ext_raw[ep - 1]
                     and y_ext_raw[ep] >= y_ext_raw[ep + 1]
                     and abs(x_sub[ext_idx_ar[ep]] - center) <= tolerance]

        if len(valid_ext) > 0:
            best_ep = valid_ext[int(np.argmax(y_ext_corr[valid_ext]))]
            imax    = int(ext_idx_ar[best_ep])

        # Step 2: shoulder fallback
        if imax is None and len(window_idx) >= 3:
            dy = np.gradient(y_win_corr.astype(float))
            rising_cands,  _ = find_peaks(dy)
            falling_cands, _ = find_peaks(-dy)
            all_cands = np.concatenate([rising_cands, falling_cands])

            win_max_corr = np.max(y_win_corr)
            valid_shoulders = []
            for c in all_cands:
                if c == 0 or c == len(y_win_raw) - 1:
                    continue
                if y_win_corr[c] < SHOULDER_MIN_FRAC * win_max_corr:
                    continue
                if y_win_raw[c] < y_win_raw[c - 1] and y_win_raw[c] < y_win_raw[c + 1]:
                    continue
                valid_shoulders.append(c)

            if len(valid_shoulders) > 0:
                shoulders_arr = np.array(valid_shoulders)
                best_local    = shoulders_arr[np.argmax(y_win_corr[shoulders_arr])]
                imax          = int(window_idx[best_local])

        if imax is None:
            found_mask.append(False)
            found_intensities.append(float("nan"))
            found_positions.append(float("nan"))
            _imax_list.append(-1)
            continue

        snr   = y_snr[imax] / yerr_sub[imax]
        found = (snr >= snr_thresh) and (y_corr_sub[imax] > 0)

        found_mask.append(found)
        found_intensities.append(float(y_sub[imax]))
        found_positions.append(float(x_sub[imax]))
        _imax_list.append(imax if found else -1)

    # Step 3: duplicate resolution
    taken = {}
    for i, imax_i in enumerate(_imax_list):
        if imax_i < 0:
            continue
        if imax_i not in taken:
            taken[imax_i] = i
        else:
            j      = taken[imax_i]
            dist_i = abs(x_sub[imax_i] - peak_centers[i])
            dist_j = abs(x_sub[imax_i] - peak_centers[j])
            winner, loser = (i, j) if dist_i <= dist_j else (j, i)
            taken[imax_i] = winner

            center_l  = peak_centers[loser]
            win_idx_l = np.where(np.abs(x_sub - center_l) <= tolerance)[0]
            if len(win_idx_l) == 0:
                continue
            ext_lo_l = max(0, win_idx_l[0] - EDGE_PAD)
            ext_hi_l = min(len(x_sub), win_idx_l[-1] + EDGE_PAD + 1)
            ext_l    = np.arange(ext_lo_l, ext_hi_l)

            alt_peaks, _ = find_peaks(y_corr_sub[ext_l])
            alt_valid = [ep for ep in alt_peaks
                         if 0 < ep < len(ext_l) - 1
                         and y_sub[ext_l[ep]] >= y_sub[ext_l[ep] - 1]
                         and y_sub[ext_l[ep]] >= y_sub[ext_l[ep] + 1]
                         and abs(x_sub[ext_l[ep]] - center_l) <= tolerance
                         and int(ext_l[ep]) not in taken
                         and int(ext_l[ep]) != imax_i]

            if alt_valid:
                best_alt = alt_valid[int(np.argmax(y_corr_sub[ext_l[alt_valid]]))]
                new_imax = int(ext_l[best_alt])
                snr_alt  = y_snr[new_imax] / yerr_sub[new_imax]
                found_alt = (snr_alt >= snr_thresh) and (y_corr_sub[new_imax] > 0)
                found_mask[loser]        = found_alt
                found_intensities[loser] = float(y_sub[new_imax])
                found_positions[loser]   = float(x_sub[new_imax])
                _imax_list[loser]        = new_imax if found_alt else -1
                if found_alt:
                    taken[new_imax] = loser

    return found_mask, found_intensities, found_positions
