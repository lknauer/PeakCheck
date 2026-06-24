# PeakCheck

[![tests](https://github.com/lknauer/PeakCheck/actions/workflows/tests.yml/badge.svg)](https://github.com/lknauer/PeakCheck/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20836679.svg)](https://doi.org/10.5281/zenodo.20836679)

Interactive multi-peak fitting and peak-presence checking for generic x/y data.

PeakCheck finds peaks in a **reference** data set (e.g. a summed spectrum) and
then checks whether those same peaks are present in one or more **component**
data sets (e.g. sub-spectra). It works with any 2- or 3-column numeric data.

**Author:** Lukas Knauer, RPTU Kaiserslautern-Landau, AG Schünemann
**License:** MIT (see `LICENSE`) · **Version:** 1.0.0

## Documentation

- `docs/Manual_PeakCheck_EN.pdf` / `docs/Anleitung_PeakCheck.pdf` — user manual (EN / DE):
  installation, GUI, x-axis unit conversion, TOML, outputs.
- `docs/Supplement_PeakCheck_EN.pdf` — technical supplement (English): method,
  formulae, algorithms and annotated code excerpts closely following the
  `peakcheck/` package source, with literature references.

The `peakcheck/` modules carry NumPy-style docstrings with references for every
scientific function; the package docstrings list the full bibliography.

## Quick start

```bash
pip install -r requirements.txt    # Tk ships with Python on Windows/macOS
pip install .                      # optional: installs the `peakcheck` console script

# Three ways to start it:
peakcheck                          # console script (after `pip install .`, from anywhere)
python -m peakcheck                # from the repo root (or after `pip install .`, from anywhere)
python peakcheck.py                # backwards-compatible launcher (from the repo root)

python -m peakcheck --version
python -m peakcheck --write-template job.toml    # write a config template
python -m peakcheck --config job.toml --no-gui   # headless / batch run
python -m peakcheck --config job.toml --validate # check a config and exit
python -m peakcheck --list-conversions           # list x-axis presets
python -m peakcheck --config job.toml --no-gui --output-dir out/   # outputs to out/
```

## Try the example

```bash
cd example_data
python ../peakcheck.py --config nis_example.toml --no-gui
```

`sample.nis` is a synthetic reference (5 peaks, energy stored in meV);
`sample_A.nis` and `sample_B.nis` are components with some peaks deliberately
absent. The run converts meV -> cm^-1, fits the five peaks and writes a presence
matrix in which the absent peaks (A: 128 & 168; B: 190) show up as 0.

Because a relative `reference_file` is resolved against the config file's own
directory, you can also run this from the repository root without `cd`, e.g.
`python -m peakcheck --config example_data/nis_example.toml --no-gui`.

### A worked example: one spectrum, ten sub-spectra

`example_data/decomposition/` shows the core idea on a larger, didactic data set.
`spectrum.dat` is a total spectrum with **eight** bands and is the literal **sum**
of ten sub-spectra (`spectrum_01.dat` … `spectrum_10.dat`). Each sub-spectrum
contains only *some* of the eight bands, so most are missing several — exactly
the question the presence check answers.

```bash
cd example_data/decomposition
python ../../peakcheck.py --config decomposition.toml --no-gui
```

The reference fit recovers all eight bands (R² ≈ 0.999, 8/8 active). For every
sub-spectrum PeakCheck then marks each band **present** (green) or **absent**
(red): e.g. `spectrum_04` carries only the three lowest bands and the other five
are flagged absent, while `spectrum_05` has only the upper four. The per-component
plots and the presence matrix in `spectrum_results.xlsx` make the pattern obvious
at a glance. (After `pip install .` you can equivalently use `python -m peakcheck`,
or run from the repository root with
`python peakcheck.py --config example_data/decomposition/decomposition.toml --no-gui`.)

## x-axis unit conversion

The third GUI control row (and the TOML) convert the x-axis on load:

- **affine** `x -> x*x_scale + x_offset` — linear unit changes
  (meV<->cm^-1, meV/cm^-1<->THz, eV->cm^-1, meV->K). Available in GUI and TOML.
- **reciprocal** `x -> x_scale/x + x_offset` — wavelength <-> wavenumber/energy/
  frequency (nm<->cm^-1, nm<->eV, nm->THz). Available via the TOML `x_conversion`
  key, e.g. `x_conversion = "nm -> eV (reciprocal)"`.

## Peak-presence criterion

A reference peak counts as present in a component when its signal-to-noise ratio
clears `snr_thresh`. By default the SNR is measured on the **background-corrected**
signal (peak height above the background — the physically meaningful measure);
set `presence_baseline_corrected = false` for the legacy raw-signal behaviour.

## Baselines and optional refinement

The baseline defaults to a rolling minimum. For smooth, curved backgrounds set
`baseline_method = "polynomial"` or `"als"` (asymmetric least squares); the
same baseline is then used by the fit, the statistics and the presence check.
Set `refine = true` to refine the peak positions after the fit by a small
bounded step (`refine_window` caps the shift, in x-units); the amplitudes stay
non-negative. Set `refine_widths = true` to also refine one width per peak,
bounded between `width_min_factor` and `width_max_factor` times the instrumental
width (`width_mode = "fwhm"` scales the whole Voigt, `"sigma"` only the Gaussian
part). Both options default to the original behaviour.

## Input

Plain-text column data in any extension (`.dat`, `.txt`, `.csv`, `.xy`, `.asc`,
`.tsv`, `.prn`, `.nis`, ...): two or three numeric columns separated by
whitespace, tabs, commas or semicolons (headers, `# ! %` comments and a
German-style decimal comma are handled automatically):

```
x   y                # errors estimated from the data (unweighted fit)
x   y   y_error      # errors used as weights
```

DESY/PETRA-III `.fio` files are read directly; `fio_x_column` / `fio_y_column`
pick the columns by name or 1-based index (default y column `nisp`).

## Output

By default, results land **next to the reference file**. In the GUI you can
pick a different folder with **Output folder…**; from the TOML, set
`output_dir = "..."` under `[output]`.

For each run PeakCheck writes:

- `*_fit.png`, `*_peaks.png` — fit and per-component plots
- `*_results.xlsx` — Excel workbook with six sheets: Intensities, Positions,
  Presence, Fit_Data, Components_Raw, Parameters (the Parameters sheet lists
  the program version, the reference and every component path)
- `*_results.csv` — single, plain 7-bit-ASCII file with the full parameter
  block as a commented header followed by five labelled sections
  (`# === SECTION: Intensities ===` etc.). All results in one file, easy to
  import in Origin or any other tool.

## Requirements

Python 3.10+ (3.11+ ships `tomllib`; on 3.10 also `pip install tomli`), numpy, scipy, matplotlib, openpyxl, Tk.

## Tests

```bash
pip install pytest
pytest                 # runs tests/test_peakcheck.py
```

The suite checks the profiles (incl. exact match to `scipy.voigt_profile` and the
pure-profile limits), noise estimation, background subtraction, the NNLS fit and
statistics, the presence logic in both SNR modes, the x-axis conversions and the
configuration round-trip.

## Citing

See `CITATION.cff`. Please cite the software if you use it in published work.

## Package contents

```
peakcheck/                 the Python package
  __init__.py              public API (Config, voigt, run_analysis, ...)
  __main__.py              `python -m peakcheck` entry
  config.py                Config dataclass, TOML I/O, axis conversions
  profiles.py              true Voigt, Gaussian, Lorentzian
  io.py                    file reading + noise estimation
  background.py            baseline subtraction (rolling-min / polynomial /
                           ALS) + search masks
  fit.py                   peak search and NNLS amplitude fit + statistics
  presence.py              reference-vs-component presence test
  plots.py                 fit and per-component plotters
  output.py                Excel and single-CSV writers
  pipeline.py              shared analysis core (GUI + headless)
  gui.py                   Tkinter graphical interface
  cli.py                   argparse-based command-line interface
peakcheck.py               backwards-compatible launcher (forwards to package)
peakcheck_template.toml     commented configuration template
LICENSE                    MIT license
README.md                  this file
CITATION.cff               citation metadata
pyproject.toml             packaging / install metadata
requirements.txt           runtime dependencies
CHANGELOG.md               version history
docs/                      manuals + supplement (PDF) and their LaTeX sources
example_data/              runnable examples:
  sample.nis (+_A,_B)      minimal example (5 peaks, meV->cm^-1) + nis_example.toml
  decomposition/           total spectrum = sum of 10 sub-spectra (8 bands) + config
tests/                     pytest test suite
```
