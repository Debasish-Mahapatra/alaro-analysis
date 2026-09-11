#!/usr/bin/env python3
"""Compatibility entrypoint; canonical recipe: alaro_analysis.recipes.paper1.make_paper_qg_qs_budget."""

from importlib import import_module


def main(*args, **kwargs):
    return import_module("alaro_analysis.recipes.paper1.make_paper_qg_qs_budget").main(
        *args, **kwargs
    )


def __getattr__(name):
    return getattr(import_module("alaro_analysis.recipes.paper1.make_paper_qg_qs_budget"), name)


if __name__ == "__main__":
    main()
