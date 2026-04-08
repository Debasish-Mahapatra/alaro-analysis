"""Reusable analysis building blocks for ALARO model outputs."""

from .caching import load_or_compute_diurnal, load_or_compute_height
from .derived import (
    compute_dp_pa,
    compute_theta_e_field,
    maybe_convert_pressure_to_pa,
)
from .experiment import ExperimentSet
from .profiles import (
    align_vertical_shapes,
    compute_diurnal_profile,
    compute_geopotential_height_profile,
    compute_surface_diurnal_cycle,
    finalize_line_means,
    finalize_profile_means,
    line_hour_accumulate,
    profile_hour_accumulate,
)

__all__ = [
    "ExperimentSet",
    "align_vertical_shapes",
    "compute_diurnal_profile",
    "compute_dp_pa",
    "compute_geopotential_height_profile",
    "compute_surface_diurnal_cycle",
    "compute_theta_e_field",
    "finalize_line_means",
    "finalize_profile_means",
    "line_hour_accumulate",
    "load_or_compute_diurnal",
    "load_or_compute_height",
    "maybe_convert_pressure_to_pa",
    "profile_hour_accumulate",
]
