"""
Reading data files and estimating noise.

Provides a tolerant text reader (`load_xy`) that handles header lines, common
comment characters (# ! %) and comma decimals, applies the x-axis transform
(affine or reciprocal) on the way, and re-sorts the data when the map
reverses the order. `estimate_noise` recovers a robust 1-sigma from the first
differences when no error column is present (factor 1.4826/sqrt(2) on the
MAD; Rousseeuw & Croux 1993).
"""
from __future__ import annotations

import os

import numpy as np

from .config import apply_x_transform


def estimate_noise(y):
    r"""Robust constant noise estimate from the first differences of ``y``.

    Used when the data file has no error column, so that the SNR filter and
    the weighted fit still have a sensible 1-sigma. The estimator is based on
    the median absolute deviation (MAD) of the first differences:

    .. math::

        \sigma = 1.4826 \cdot \frac{\operatorname{MAD}(\Delta y)}{\sqrt{2}},
        \qquad \Delta y_i = y_{i+1} - y_i .

    The factor 1.4826 converts the MAD into a consistent estimator of the
    standard deviation of normally distributed data [6]_; the factor
    :math:`1/\sqrt2` removes the variance doubling caused by differencing
    (:math:`\operatorname{Var}(\Delta y) = 2\sigma^2` for independent noise).
    Differencing first suppresses smooth signal, so the estimate reflects the
    high-frequency noise rather than the peaks.

    Parameters
    ----------
    y : numpy.ndarray
        Intensity values.

    Returns
    -------
    float
        A single positive sigma applied uniformly across the spectrum.
        Falls back to ``std(y)`` (then 1.0) if the MAD vanishes.

    References
    ----------
    .. [6] P. J. Rousseeuw, C. Croux, J. Am. Stat. Assoc. 88 (1993) 1273-1283.
    """
    d = np.diff(np.asarray(y, dtype=float))
    if len(d) == 0:
        return 1.0
    mad = np.median(np.abs(d - np.median(d)))
    sigma = 1.4826 * mad / np.sqrt(2.0)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.std(y) if np.std(y) > 0 else 1.0
    return float(sigma)


def _fio_columns(filename):
    """Parse a DESY/PETRA-III FIO file: return (col_names, data_array).

    FIO layout: a ``%c`` comment block, a ``%p`` parameter block (``key = value``)
    and a ``%d`` data block. The data block starts with ``Col <n> <name> <type>``
    declarations followed by whitespace-separated numeric rows. Only the data
    block is read here; the parameter block is ignored.
    """
    names, rows, in_data = [], [], False
    with open(filename, "r", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("%d"):
                in_data = True
                continue
            if s.startswith("%"):          # %c / %p block -> not data
                in_data = False
                continue
            if not in_data:
                continue
            if s.startswith("Col "):
                parts = s.split()
                # "Col <index> <name> <type>"
                if len(parts) >= 3:
                    names.append(parts[2])
                continue
            if s.startswith("!"):          # in-data comment (e.g. acquisition end)
                continue
            parts = s.split()
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No FIO data block found in '{filename}'.")
    width = min(len(r) for r in rows)
    data = np.array([r[:width] for r in rows], dtype=float)
    return names, data


def _resolve_fio_column(spec, names, n_cols, default_idx):
    """Resolve a FIO column spec (name or 1-based index, '' = default) to a
    0-based column index."""
    if spec is None or str(spec).strip() == "":
        return default_idx
    spec = str(spec).strip()
    if names and spec in names:
        return names.index(spec)
    try:
        i = int(spec)                      # 1-based index in the file
        if 1 <= i <= n_cols:
            return i - 1
    except ValueError:
        pass
    raise ValueError(
        f"FIO column '{spec}' not found. Available columns: "
        f"{', '.join(names) if names else f'1..{n_cols}'}.")


def _load_fio(filename, cfg):
    """Load x/y from a FIO file using ``cfg.fio_x_column``/``cfg.fio_y_column``.

    Returns a plain ``(N, 2)`` array ``[x, y]``; the surrounding :func:`load_xy`
    then applies the usual transform, sorting, error handling and non-finite
    cleanup. FIO files carry no per-point error column, so the noise is always
    estimated downstream unless the user sets an explicit ``error_column`` that
    maps onto one of the FIO columns by index.
    """
    names, data = _fio_columns(filename)
    n_cols = data.shape[1]
    xi = _resolve_fio_column(getattr(cfg, "fio_x_column", ""), names, n_cols, 0)
    yi = _resolve_fio_column(getattr(cfg, "fio_y_column", "nisp"), names, n_cols, 1)
    return np.column_stack([data[:, xi], data[:, yi]])


def _split_numeric_row(line):
    """Split one text line into floats, tolerant of common delimiters.

    Handles whitespace-, tab-, comma- and semicolon-separated values, and
    distinguishes a *decimal comma* (German-style ``10,5``) from a comma used
    as a field separator (``10,100``). Returns ``[]`` for comment or
    non-numeric lines.

    Strategy: comment lines (starting with ``# ! %``) and blanks are dropped.
    Semicolons and tabs are treated as separators. If commas remain, the line
    is parsed twice -- once treating commas as decimal points, once as
    separators -- and the interpretation that yields more valid floats (ties
    going to the decimal reading) is kept. This keeps ``1,5 2,5`` as two values
    ``1.5, 2.5`` while still splitting ``1,2,3`` into three.
    """
    s = line.strip()
    if not s or s[0] in "#!%":
        return []
    # normalise tabs and semicolons to spaces (unambiguous separators)
    s = s.replace("\t", " ").replace(";", " ")

    def _floats(tokens):
        out = []
        for t in tokens:
            try:
                out.append(float(t))
            except ValueError:
                return None
        return out

    if "," not in s:
        vals = _floats(s.split())
        return vals if vals else []

    # Commas present. The number of whitespace-separated tokens tells us how
    # many columns there really are; if that already matches the data (each
    # such token being a number once its comma is read as a decimal point),
    # the commas are decimal points. Otherwise commas are field separators.
    ws_tokens = s.split()
    as_decimal = _floats([t.replace(",", ".") for t in ws_tokens])
    if as_decimal is not None and len(ws_tokens) >= 2:
        # every whitespace token is a number with a decimal comma -> use it
        return as_decimal
    # fall back to comma-as-separator (covers "10,100" and "1,2,3")
    as_sep = _floats(s.replace(",", " ").split())
    if as_sep is not None:
        return as_sep
    # last resort: single decimal-comma value split by whitespace
    return as_decimal if as_decimal is not None else []


def load_xy(filename, cfg):
    r"""Load a generic data file with 2 or 3 numeric columns: ``x, y [, y_err]``.

    A fast path (:func:`numpy.loadtxt`) is tried first; on failure a tolerant
    line-by-line parser takes over that copes with header lines, comment
    characters (``# ! %``), comma decimal separators, extra columns and ragged
    rows. Non-finite rows are dropped, the data are sorted by ascending ``x``
    and the affine transform :math:`x \mapsto x\cdot\texttt{x\_scale} +
    \texttt{x\_offset}` is applied (e.g. ``x_scale=8.06554`` converts meV to
    cm\ :sup:`-1`).

    The third column is optional and governed by ``cfg.error_column``:

    * ``"auto"`` -- use the third column if present, otherwise estimate a
      constant noise with :func:`estimate_noise`.
    * ``"none"`` -- ignore any third column and estimate the noise.
    * integer -- use that 0-based column as the error.

    Parameters
    ----------
    filename : str
        Path to the data file.
    cfg : Config
        Supplies ``error_column``, ``x_scale`` and ``x_offset``.

    Returns
    -------
    x, y, yerr : numpy.ndarray
        Abscissa, intensities and per-point 1-sigma uncertainties (always
        strictly positive: non-positive errors are replaced by the smallest
        positive one to keep the weighted fit finite).
    had_errors : bool
        ``True`` only if real per-point errors were read from the file (drives
        the error-reporting convention in :func:`fit_statistics`).

    Raises
    ------
    ValueError
        If no usable 2+ column numeric data are found.
    """
    # FIO files (DESY/PETRA III) have a structured header; parse them directly.
    if str(filename).lower().endswith(".fio"):
        data = _load_fio(filename, cfg)
    else:
        # Fast path, then a tolerant line-by-line fallback.
        try:
            data = np.loadtxt(filename, comments=("#", "!", "%"))
        except Exception:
            rows = []
            with open(filename, "r", errors="replace") as fh:
                for line in fh:
                    vals = _split_numeric_row(line)
                    if len(vals) >= 2:
                        rows.append(vals)
            if not rows:
                raise ValueError(
                    f"No numeric 2+ column data found in '{filename}'. "
                    "Expected: x  y  [y_error]."
                )
            width = min(len(r) for r in rows)
            data = np.array([r[:width] for r in rows])

    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(
            f"'{filename}' has only {data.shape[1]} column(s); need at least 2 (x, y)."
        )

    x = data[:, 0].astype(float)
    y = data[:, 1].astype(float)

    # decide error column
    ecol = cfg.error_column
    yerr = None
    had_errors = False
    if isinstance(ecol, str) and ecol.lower() == "none":
        pass
    elif isinstance(ecol, str) and ecol.lower() == "auto":
        if data.shape[1] >= 3:
            yerr = data[:, 2].astype(float).copy()
            had_errors = True
    else:
        try:
            ci = int(ecol)
            yerr = data[:, ci].astype(float).copy()
            had_errors = True
        except (ValueError, IndexError):
            print(f"  [warn] error_column={ecol!r} invalid; estimating noise instead.")

    # drop non-finite rows
    if yerr is not None:
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
    else:
        finite = np.isfinite(x) & np.isfinite(y)
    if not np.all(finite):
        n_bad = int(np.sum(~finite))
        print(f"  [warn] {os.path.basename(filename)}: dropped {n_bad} non-finite row(s).")
        x, y = x[finite], y[finite]
        if yerr is not None:
            yerr = yerr[finite]
    if len(x) < 2:
        raise ValueError(f"'{filename}' contains fewer than 2 usable data rows.")

    # ascending x
    if np.any(np.diff(x) < 0):
        order = np.argsort(x)
        x, y = x[order], y[order]
        if yerr is not None:
            yerr = yerr[order]

    # x transform (affine or reciprocal)
    if (str(cfg.x_transform).lower() == "reciprocal"
            or cfg.x_scale != 1.0 or cfg.x_offset != 0.0):
        x = apply_x_transform(x, cfg)
        # a reciprocal map reverses the order, so re-sort if needed
        if np.any(np.diff(x) < 0):
            order = np.argsort(x)
            x, y, yerr = x[order], y[order], (yerr[order] if yerr is not None else None)

    # finalise errors
    if yerr is None:
        sigma = estimate_noise(y)
        yerr = np.full_like(y, sigma)
    else:
        pos = yerr > 0
        min_err = yerr[pos].min() if np.any(pos) else 1.0
        yerr[yerr <= 0] = min_err

    return x, y, yerr, had_errors
