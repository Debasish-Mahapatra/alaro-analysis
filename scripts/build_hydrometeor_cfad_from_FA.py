#!/usr/bin/env python3
"""Compatibility entrypoint; canonical recipe: alaro_analysis.recipes.diagnostics.build_hydrometeor_cfad_from_FA."""

import sys
from importlib import import_module


def main(*args, **kwargs):
    return import_module("alaro_analysis.recipes.diagnostics.build_hydrometeor_cfad_from_FA").main(
        *args, **kwargs
    )


def __getattr__(name):
    return getattr(
        import_module("alaro_analysis.recipes.diagnostics.build_hydrometeor_cfad_from_FA"), name
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
