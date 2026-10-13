#!/usr/bin/env python3
"""Compatibility entrypoint; canonical recipe: alaro_analysis.recipes.iop.make_iop_rain_error_metrics."""

from importlib import import_module


def main(*args, **kwargs):
    return import_module("alaro_analysis.recipes.iop.make_iop_rain_error_metrics").main(*args, **kwargs)


def __getattr__(name):
    return getattr(import_module("alaro_analysis.recipes.iop.make_iop_rain_error_metrics"), name)


if __name__ == "__main__":
    raise SystemExit(main())
