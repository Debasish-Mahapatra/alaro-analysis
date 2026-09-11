#!/usr/bin/env python3
"""Compatibility entrypoint; canonical recipe: alaro_analysis.recipes.dsd.run_dsd_per_cell_normalized_gamma_raw_sets."""

from importlib import import_module


def main(*args, **kwargs):
    return import_module(
        "alaro_analysis.recipes.dsd.run_dsd_per_cell_normalized_gamma_raw_sets"
    ).main(*args, **kwargs)


def __getattr__(name):
    return getattr(
        import_module("alaro_analysis.recipes.dsd.run_dsd_per_cell_normalized_gamma_raw_sets"), name
    )


if __name__ == "__main__":
    main()
