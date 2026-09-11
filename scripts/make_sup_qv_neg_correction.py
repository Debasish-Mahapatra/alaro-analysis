#!/usr/bin/env python3
"""Compatibility entrypoint; canonical recipe: alaro_analysis.recipes.supplementary.make_sup_qv_neg_correction."""

from importlib import import_module


def main(*args, **kwargs):
    return import_module("alaro_analysis.recipes.supplementary.make_sup_qv_neg_correction").main(
        *args, **kwargs
    )


def __getattr__(name):
    return getattr(
        import_module("alaro_analysis.recipes.supplementary.make_sup_qv_neg_correction"), name
    )


if __name__ == "__main__":
    main()
