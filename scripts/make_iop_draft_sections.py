#!/usr/bin/env python3
"""Compatibility entrypoint; canonical recipe: alaro_analysis.recipes.iop.make_iop_draft_sections."""

from importlib import import_module


def main(*args, **kwargs):
    return import_module("alaro_analysis.recipes.iop.make_iop_draft_sections").main(
        *args, **kwargs
    )


def __getattr__(name):
    return getattr(import_module("alaro_analysis.recipes.iop.make_iop_draft_sections"), name)


if __name__ == "__main__":
    raise SystemExit(main())
