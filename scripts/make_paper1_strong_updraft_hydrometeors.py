#!/usr/bin/env python3
"""Compatibility entrypoint; canonical recipe: alaro_analysis.recipes.paper1.make_paper1_strong_updraft_hydrometeors."""

from importlib import import_module


def main(*args, **kwargs):
    return import_module(
        "alaro_analysis.recipes.paper1.make_paper1_strong_updraft_hydrometeors"
    ).main(*args, **kwargs)


def __getattr__(name):
    return getattr(
        import_module("alaro_analysis.recipes.paper1.make_paper1_strong_updraft_hydrometeors"), name
    )


if __name__ == "__main__":
    main()
