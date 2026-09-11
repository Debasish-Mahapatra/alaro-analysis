#!/usr/bin/env python3
"""Compatibility entrypoint; canonical recipe: alaro_analysis.recipes.maintenance.fix_paper5_flux_dilution."""

from importlib import import_module


def main(*args, **kwargs):
    return import_module("alaro_analysis.recipes.maintenance.fix_paper5_flux_dilution").main(
        *args, **kwargs
    )


def __getattr__(name):
    return getattr(
        import_module("alaro_analysis.recipes.maintenance.fix_paper5_flux_dilution"), name
    )


if __name__ == "__main__":
    raise SystemExit(main())
