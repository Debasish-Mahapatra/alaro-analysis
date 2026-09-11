#!/usr/bin/env python3
"""Compatibility entrypoint; canonical recipe: alaro_analysis.recipes.diagnostics.make_moisture_flux_convergence_g1m_c1m."""

from importlib import import_module


def main(*args, **kwargs):
    return import_module(
        "alaro_analysis.recipes.diagnostics.make_moisture_flux_convergence_g1m_c1m"
    ).main(*args, **kwargs)


def __getattr__(name):
    return getattr(
        import_module("alaro_analysis.recipes.diagnostics.make_moisture_flux_convergence_g1m_c1m"),
        name,
    )


if __name__ == "__main__":
    raise SystemExit(main())
