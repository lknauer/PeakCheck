"""
Command-line interface: argument parsing and `main()` entry point.

`peakcheck` (the console script) and `python -m peakcheck` both end up here.
Dispatches to (a) template writing, (b) the headless pipeline, or (c) the GUI.
"""
from __future__ import annotations

import argparse
import os

from . import __version__
from .config import Config, load_config, write_template, _coerce_nullable, X_CONVERSIONS
from .pipeline import run_headless


def build_arg_parser():
    """Build and return the command-line argument parser for PeakCheck."""
    p = argparse.ArgumentParser(
        prog="peakcheck",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="PeakCheck — interactive multi-peak fitting and peak-presence "
                    "checking for generic x/y data.",
        epilog="Examples:\n"
               "  peakcheck                                 # launch the GUI\n"
               "  peakcheck --write-template job.toml       # write a config template\n"
               "  peakcheck --config job.toml --no-gui      # headless / batch run\n"
               "  peakcheck --config job.toml --no-gui --output-dir out/  # outputs to out/\n"
               "  peakcheck --config job.toml --validate    # check a config and exit\n"
               "  peakcheck --list-conversions              # show x-axis presets\n")
    p.add_argument("config", nargs="?", default=None,
                   help="TOML config file (optional).")
    p.add_argument("--config", dest="config_opt", default=None,
                   help="TOML config file (alternative to the positional argument).")
    p.add_argument("--version", action="version",
                   version=f"PeakCheck {__version__}")
    p.add_argument("--write-template", metavar="PATH", default=None,
                   help="Write a commented TOML template to PATH and exit.")
    p.add_argument("--validate", action="store_true",
                   help="Load the config, check it for problems, report and exit.")
    p.add_argument("--list-conversions", action="store_true",
                   help="List the available x_conversion presets and exit.")
    p.add_argument("--no-gui", action="store_true",
                   help="Run headless (no window); needs a config file.")
    p.add_argument("--reference", default=None,
                   help="Override the reference data file from the config.")
    p.add_argument("--output-dir", metavar="DIR", default=None,
                   help="Write all outputs to DIR instead of next to the reference file.")
    p.add_argument("--debug", action="store_true",
                   help="On error, show the full Python traceback instead of a short message.")
    return p


def _fail(exc, debug, context=""):
    """Report a user-facing error and exit non-zero.

    With ``debug`` true the original exception is re-raised so the full Python
    traceback is shown; otherwise only a short, clean message is printed to
    stderr. The exit status is 1 either way (via the re-raise or SystemExit),
    so batch scripts can detect the failure.
    """
    if debug:
        raise exc
    msg = str(exc).rstrip(". ")
    where = f" (while {context})" if context else ""
    raise SystemExit(f"PeakCheck: {msg}{where}.")


def main(argv=None):
    """Entry point: parse arguments and dispatch to template writing, the
    headless run or the GUI."""
    args = build_arg_parser().parse_args(argv)

    if args.write_template:
        write_template(args.write_template)
        return

    if args.list_conversions:
        print("Available x_conversion presets (use under [xaxis] as x_conversion):")
        for name, (sc, off, tr, lab, unit) in X_CONVERSIONS.items():
            unit_s = f" ({unit})" if unit else ""
            print(f"  {name:30s} [{tr}]  scale={sc:g}, offset={off:g}  -> {lab}{unit_s}")
        return

    cfg_path = args.config_opt or args.config
    cfg = Config()
    if cfg_path:
        if not os.path.isfile(cfg_path):
            raise SystemExit(f"PeakCheck: config file not found: '{cfg_path}'.")
        try:
            cfg = load_config(cfg_path)
        except (ValueError, OSError) as exc:
            _fail(exc, args.debug, context=f"reading config '{cfg_path}'")
        _coerce_nullable(cfg)
    if args.reference:
        cfg.reference_file = args.reference
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    if args.validate:
        if not cfg_path:
            raise SystemExit("--validate needs a config file.")
        problems = cfg.validate()
        if problems:
            print("Configuration problems:")
            for pr in problems:
                print(f"  - {pr}")
            raise SystemExit(1)
        print("Configuration OK.")
        if not os.path.isfile(cfg.reference_file):
            print(f"  [note] reference file '{cfg.reference_file}' not found "
                  "(set it before a headless run).")
        return

    if args.no_gui:
        if not cfg_path and not args.reference:
            raise SystemExit("--no-gui needs a config file (or --reference).")
        print("=" * 60)
        print("  PeakCheck — headless run")
        print("=" * 60)
        try:
            run_headless(cfg, verbose=True)
        except (ValueError, OSError) as exc:
            _fail(exc, args.debug, context="the headless run")
        print("\n  Done.")
        return

    # Default: launch the GUI. Import here so a headless machine without Tk
    # can still use --no-gui without pulling in tkinter.
    from .gui import PeakCheckGUI
    try:
        gui = PeakCheckGUI(cfg)
    except Exception as exc:
        raise SystemExit(
            f"Could not start the GUI ({type(exc).__name__}: {exc}).\n"
            "If you are on a headless machine, use:  peakcheck --config job.toml --no-gui")
    gui.run()


if __name__ == "__main__":
    main()
