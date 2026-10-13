#!/usr/bin/env python3
"""Compatibility entrypoint; canonical recipe: alaro_analysis.recipes.iop.build_iop_rwp_mass_flux."""

from importlib import import_module


def main(*args, **kwargs):
    return import_module("alaro_analysis.recipes.iop.build_iop_rwp_mass_flux").main(*args, **kwargs)


def __getattr__(name):
    return getattr(import_module("alaro_analysis.recipes.iop.build_iop_rwp_mass_flux"), name)


if __name__ == "__main__":
    raise SystemExit(main())
