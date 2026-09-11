#!/usr/bin/env python3
"""Compatibility entrypoint; canonical recipe: alaro_analysis.recipes.diagnostics.make_dm_updraft_profile."""

from importlib import import_module


def main(*args, **kwargs):
    return import_module("alaro_analysis.recipes.diagnostics.make_dm_updraft_profile").main(
        *args, **kwargs
    )


def __getattr__(name):
    return getattr(
        import_module("alaro_analysis.recipes.diagnostics.make_dm_updraft_profile"), name
    )


if __name__ == "__main__":
    main()
