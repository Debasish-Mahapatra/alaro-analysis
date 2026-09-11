#!/usr/bin/env python3
"""Compatibility entrypoint; canonical recipe: alaro_analysis.recipes.supplementary.make_sup_fusion_heat_feedback_boxmodel."""

from importlib import import_module


def main(*args, **kwargs):
    return import_module(
        "alaro_analysis.recipes.supplementary.make_sup_fusion_heat_feedback_boxmodel"
    ).main(*args, **kwargs)


def __getattr__(name):
    return getattr(
        import_module(
            "alaro_analysis.recipes.supplementary.make_sup_fusion_heat_feedback_boxmodel"
        ),
        name,
    )


if __name__ == "__main__":
    main()
