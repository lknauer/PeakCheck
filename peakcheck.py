#!/usr/bin/env python3
"""
Backwards-compatible launcher.

This file lets you keep running PeakCheck via `python peakcheck.py` even after
the code was split into the `peakcheck/` package. It locates the package and
dispatches to its entry point, so old shortcuts, scripts and documentation
continue to work unchanged.

For new uses prefer one of:
    peakcheck                       # console script (installed via pip)
    python -m peakcheck             # direct module call
    import peakcheck                # programmatic use
"""
import os
import sys

# Make sure the directory holding this file is on sys.path so that
# `import peakcheck` finds the sibling `peakcheck/` package even when the
# script is invoked from somewhere else (e.g. inside example_data/).
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from peakcheck.cli import main

if __name__ == "__main__":
    main()
