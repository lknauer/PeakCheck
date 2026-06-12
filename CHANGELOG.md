# Changelog

All notable changes to PeakCheck are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-06-01
First public release.

### Profiles & fitting
- True Voigt profile via the Faddeeva function (`scipy.special.wofz`), plus
  Gaussian and Lorentzian — all area-normalised, so a fitted amplitude is the
  integrated intensity and is comparable across profiles.
- Background subtraction (clipped at zero) via `baseline_method`: rolling
  minimum (default), plus a polynomial (ModPoly) and an asymmetric-least-
  squares (ALS) baseline for smooth, curved backgrounds.
- Weighted non-negative least-squares (NNLS) amplitude fit at fixed
  positions/widths (active-set algorithm, `scipy.optimize.nnls`).
- Command line: `--validate` (check a config and exit non-zero on error),
  `--list-conversions` (print the x-axis presets) and `--output-dir DIR`
  (send all outputs to `DIR` instead of next to the reference file), in
  addition to `--config`, `--no-gui`, `--write-template`, `--reference`
  and `--version`. A missing config or data file, a malformed TOML or an
  unreadable data file is reported as a short `PeakCheck: …` message with a
  non-zero exit code; `--debug` shows the full Python traceback instead.
- Portable configs: a *relative* `reference_file` (and a relative `output_dir`)
  in a config is resolved against that config file's directory (not the working
  directory), so a config and its data can be run from anywhere. Absolute paths
  are used as given; a command-line `--reference`/`--output-dir` keeps the usual
  working-directory semantics.
- Reads plain-text column data regardless of extension (`.dat`, `.txt`, `.csv`,
  `.xy`, `.asc`, `.tsv`, `.prn`, `.nis`, ...): two or three numeric columns
  separated by whitespace, tabs, commas or semicolons, with header lines,
  `# ! %` comments and a German-style decimal comma detected automatically.
- Reads DESY/PETRA-III `.fio` files directly: the `%d` data block with its
  `Col` declarations is parsed, and `fio_x_column`/`fio_y_column` choose the
  x/y columns by name or 1-based index (default y column `nisp`). FIO files
  have no error column, so the noise is estimated unless `error_column` points
  at one.
- GUI: the controls are grouped into labelled frames (Files & configuration,
  Profile & widths, x-axis, Background & peak presence, Refinement) with a
  subtle colour accent; parameters that do not apply to the current selection
  (e.g. the ALS fields when the baseline is not ALS, the width bounds when
  width refinement is off) are greyed out. A read-only effective-FWHM value is
  shown next to WG/WL and updates live with the widths and profile. Peak markers
  are labelled with one decimal, matching the fit and component plots (they were
  rounded to whole numbers before, which made placed peaks look as if they had
  moved after a fit).
- If a fit looks poor — a peak the user asked for is dropped to zero amplitude,
  or R^2 is below zero — PeakCheck now prints a short hint (and shows a warning
  dialog in the GUI) that the profile width `WG`/`WL` or the x-axis conversion
  may not match the data's peak spacing. A merely large reduced chi-square is
  not flagged, so the hint does not cry wolf on good fits with estimated errors.
- Goodness of fit (reduced chi^2, R^2) and per-amplitude 1-sigma standard errors
  from the covariance of the active set, with `statistical` (unscaled) and
  `scaled` (× sqrt(reduced chi^2)) error conventions.
- Optional bounded position refinement (`refine`): with the widths held fixed,
  the peak positions are refined by a variable-projection fit (amplitudes stay
  non-negative, each centre constrained by `refine_window`); off by default.
- Optional bounded per-peak width refinement (`refine_widths`): one width per
  peak is refined by the same variable-projection scheme, bounded between
  `width_min_factor` and `width_max_factor` times the instrumental width
  (default 1.0..2.0). `width_mode` chooses what broadens (`"fwhm"` scales the
  whole Voigt, `"sigma"` only the Gaussian part). Off by default. Refined
  positions and widths reduce the degrees of freedom, so the reduced chi-square
  is reported honestly.

### Presence check
- Multi-stage peak-presence search across component data sets (strict maximum,
  shoulder fallback via the derivative, duplicate resolution by proximity).
- Presence SNR evaluated on the background-corrected signal by default
  (peak height above background; physically the right measure), switchable to
  the raw signal via `presence_baseline_corrected = false`.

### Interface & configuration
- Tkinter GUI: folder chooser (pick one reference, any number of components),
  mouse-driven peak add/remove, live plot, parameter sliders, and hover tooltips
  on every field, slider and button.
- TOML-driven headless mode sharing the same analysis core; an explicit
  component-file list can override the automatic glob search.
- x-axis unit conversion: affine presets (e.g. meV → cm^-1) in the GUI and TOML,
  plus reciprocal conversions (wavelength ↔ wavenumber/energy/frequency) via the
  TOML. Configurable `output_dir` for all written files.

### Outputs
- Fit plot (`*_fit.png`) and one marked plot per component (`*_peaks.png`).
- Excel workbook (`*_results.xlsx`) with six sheets: Intensities, Positions,
  Presence, Fit_Data, Components_Raw, Parameters.
- A single plain 7-bit-ASCII CSV (`*_results.csv`) holding the same data in five
  labelled sections, each preceded by the full parameter block as a commented
  header — fully documenting and reproducing every run.

### Packaging & docs
- Modular Python package (`config`, `profiles`, `io`, `background`, `fit`,
  `presence`, `plots`, `output`, `pipeline`, `gui`, `cli`) with a
  backwards-compatible `peakcheck.py` launcher; importable via
  `python -m peakcheck`. A lazy GUI import keeps headless installs working
  without Tk.
- Runnable examples in `example_data/`: a minimal one (`sample.nis` with two
  components, meV->cm^-1) and a larger showcase in `decomposition/` — a total
  spectrum of eight bands built as the exact sum of ten sub-spectra, where each
  sub-spectrum is missing several bands so the presence check has a clear
  present/absent pattern to recover.
- German and English user manuals and technical supplements (method derivation
  with annotated source; code-snippet headers name the exact submodule path).
- pytest test suite (`tests/test_peakcheck.py`) covering profiles, noise
  estimation, background subtraction (all three methods), the NNLS fit and
  statistics, position refinement, the presence logic in both SNR modes, the
  x-axis conversions, the config round-trip, and an end-to-end regression test
  that pins the worked-example amplitudes, chi^2, R^2 and presence matrix.
- GitHub Actions workflow `tests.yml` (pytest on Python 3.10–3.13 with
  coverage).
- Colourblind-safe figures (Okabe–Ito palette with redundant linestyles and
  markers) and a Monte-Carlo recovery/error-calibration figure and section in
  the supplement; added verified NRVS/NIS review references.
