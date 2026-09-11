#!/usr/bin/env python3
"""Compatibility entrypoint; canonical recipe: alaro_analysis.recipes.iop.run_iop_rainfall_updraft."""

from importlib import import_module


def main(*args, **kwargs):
    return import_module("alaro_analysis.recipes.iop.run_iop_rainfall_updraft").main(
        *args, **kwargs
    )


def __getattr__(name):
    return getattr(import_module("alaro_analysis.recipes.iop.run_iop_rainfall_updraft"), name)


if __name__ == "__main__":
    main()
