"""
alaro_analysis - Reusable analysis toolkit for ALARO NWP model outputs.

Quick start::

    from alaro_analysis import ExperimentSet

    exps = ExperimentSet.from_three_dirs(control, graupel, twomom)
    exps.plot_surface_diurnal("CLPMHAUT.MOD.XFU", "output.png",
                              label="BL height", unit="m")
"""

__version__ = "0.2.0"

from .analysis.experiment import ExperimentSet
from .common.models import AxisSpec, PeriodSpec, SpatialWindow, VerticalAxis

__all__ = [
    "ExperimentSet",
    "AxisSpec",
    "PeriodSpec",
    "SpatialWindow",
    "VerticalAxis",
]
