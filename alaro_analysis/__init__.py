"""
alaro_analysis - Reusable analysis toolkit for ALARO NWP model outputs.

Quick start::

    from alaro_analysis.analysis import (
        ExperimentSet,
        compute_diurnal_profile,
        compute_surface_diurnal_cycle,
        load_or_compute_diurnal,
    )
    from alaro_analysis.converter import run_conversion
    from alaro_analysis.plotting.panels import (
        plot_surface_diurnal_cycle,
        plot_three_panel_diurnal,
    )
"""

__version__ = "0.1.0"

from .common.models import AxisSpec, PeriodSpec, SpatialWindow, VerticalAxis

__all__ = [
    "AxisSpec",
    "PeriodSpec",
    "SpatialWindow",
    "VerticalAxis",
]
