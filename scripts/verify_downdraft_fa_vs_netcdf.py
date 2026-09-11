#!/usr/bin/env python3
"""Compatibility entrypoint; canonical recipe: alaro_analysis.recipes.diagnostics.verify_downdraft_fa_vs_netcdf."""

from importlib import import_module


def main(*args, **kwargs):
    return import_module("alaro_analysis.recipes.diagnostics.verify_downdraft_fa_vs_netcdf").main(
        *args, **kwargs
    )


def __getattr__(name):
    return getattr(
        import_module("alaro_analysis.recipes.diagnostics.verify_downdraft_fa_vs_netcdf"), name
    )


if __name__ == "__main__":
    raise SystemExit(main())
