#!/usr/bin/env python3
"""Compatibility entrypoint; canonical recipe: alaro_analysis.recipes.diagnostics.make_updraft_freezing_chain."""

from importlib import import_module


def main(*args, **kwargs):
    return import_module("alaro_analysis.recipes.diagnostics.make_updraft_freezing_chain").main(
        *args, **kwargs
    )


def __getattr__(name):
    return getattr(
        import_module("alaro_analysis.recipes.diagnostics.make_updraft_freezing_chain"), name
    )


if __name__ == "__main__":
    main()
