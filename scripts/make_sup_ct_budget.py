#!/usr/bin/env python3
"""Compatibility entrypoint; canonical recipe: alaro_analysis.recipes.supplementary.make_sup_ct_budget."""

from importlib import import_module


def main(*args, **kwargs):
    return import_module("alaro_analysis.recipes.supplementary.make_sup_ct_budget").main(
        *args, **kwargs
    )


def __getattr__(name):
    return getattr(import_module("alaro_analysis.recipes.supplementary.make_sup_ct_budget"), name)


if __name__ == "__main__":
    main()
