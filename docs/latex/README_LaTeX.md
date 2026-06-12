# PeakCheck — LaTeX sources (Version 1.0.0)

This bundle contains the editable LaTeX sources for the three PeakCheck documents,
all set to **version 1.0.0**:

| File                          | Language | Description                                   |
|-------------------------------|----------|-----------------------------------------------|
| `Anleitung_PeakCheck.tex`     | German   | User manual (Anleitung)                       |
| `Manual_PeakCheck_EN.tex`     | English  | User manual                                   |
| `Supplement_PeakCheck_EN.tex` | English  | Technical supplement (method + annotated code)|

Shared infrastructure:

- `preamble.tex` — shared preamble (packages, listings styles, colours, macros).
  Each document sets its language via `\def\peakchecklang{...}` **before**
  `\input{preamble.tex}`, then the preamble loads `babel` accordingly.
- `references.bib` — bibliography database (used by the supplement via `natbib`).
- `figures/` — the figures (PNG), referenced with `\includegraphics`.
- `pdf_preview/` — reference PDFs already compiled from these sources (see note
  below).

## Compiling

**Engine:** `pdflatex`. **Bibliography:** `bibtex` + `natbib` (`plainnat`).

### On Overleaf (recommended)
Upload the whole folder, set the main document, and compile. Overleaf has the full
TeX Live, so German hyphenation (`texlive-lang-german`) and Latin Modern fonts are
present automatically. For the supplement, Overleaf runs `bibtex` on its own.

### Locally
Manuals (no bibliography):
```
pdflatex Anleitung_PeakCheck.tex
pdflatex Anleitung_PeakCheck.tex
pdflatex Manual_PeakCheck_EN.tex
pdflatex Manual_PeakCheck_EN.tex
```
Supplement (with bibliography):
```
pdflatex Supplement_PeakCheck_EN.tex
bibtex   Supplement_PeakCheck_EN
pdflatex Supplement_PeakCheck_EN.tex
pdflatex Supplement_PeakCheck_EN.tex
```
Or simply `latexmk -pdf <file>.tex` for any of them.

## Notes

- **German manual.** `preamble.tex` selects `ngerman` for `Anleitung_PeakCheck.tex`
  and falls back to English hyphenation only if the German babel files are absent,
  so the file always compiles; on a full TeX Live / Overleaf it is typeset in
  German. The `pdf_preview/Anleitung_PeakCheck.pdf` here is the proper German
  build. (Umlauts are written with `\"`-commands and render correctly regardless.)
- **Figures.** The figures carry English labels and match the v1.0.0 behaviour
  (e.g. the pipeline shows "Excel (6 sheets)" and the GUI mock-up the "Open
  folder…" button). The German manual uses a German-labelled pipeline figure,
  `figures/fig_pipeline_de.png`; the English manual and the supplement use
  `figures/fig_pipeline.png` and the remaining English figures.
- **Content.** The prose follows the original PDFs; every formula is taken from the
  NumPy-style docstrings in the source; the code excerpts closely follow the
  `peakcheck` modules (with brief inline comments for orientation and, where marked
  by `...`, abbreviations). The supplement additionally documents numerical
  validation (true Voigt vs `scipy` to machine precision, NNLS uniqueness,
  reproducibility), assumptions/limitations with parameter-selection guidance, and
  a notation glossary.

© 2026 Lukas Knauer, RPTU Kaiserslautern-Landau, AG Schünemann · MIT license.
