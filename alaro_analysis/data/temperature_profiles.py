"""Temperature profiles. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alaro_analysis.analysis import profiles as profile_analysis
from alaro_analysis.common.models import SpatialWindow, VerticalAxis
from alaro_analysis.data.profile_products import (
    load_height,
    load_profile,
)


def compute_diurnal_profile(
    experiment_dir: Path,
    variable: str,
    max_days: int | None,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
) -> tuple[np.ndarray, np.ndarray, int, Path]:
    """Equal-weight mean of each file's finite spatial-mean profile."""
    return profile_analysis.compute_diurnal_profile(
        experiment_dir / variable,
        variable,
        max_days=max_days,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
    )


def compute_geopotential_height_profile(
    geopotential_dir: Path,
    height_variable: str,
    max_days: int | None,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
    aggregate: str,
    spatial_window: SpatialWindow | None = None,
) -> tuple[np.ndarray, int]:
    """Read the shared height profile; converted GEOPOTENTIEL is in metres."""
    return profile_analysis.compute_geopotential_height_profile(
        geopotential_dir,
        height_variable,
        max_days=max_days,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
        aggregate=aggregate,
        spatial_window=spatial_window,
    )


def load_or_compute_diurnal(
    cache_file: Path,
    experiment_dir: Path,
    variable: str,
    max_days: int | None,
    overwrite_intermediate: bool,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
):
    """Compatibility adapter for the shared, versioned profile product."""
    product = load_profile(
        cache_file,
        experiment_dir,
        variable,
        max_days=max_days,
        recompute=overwrite_intermediate,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
        spatial_window=None,
    )
    return product.mean, product.sample_file


def load_or_compute_height(
    cache_file: Path,
    geopotential_dir: Path,
    height_variable: str,
    max_days: int | None,
    overwrite_intermediate: bool,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
    aggregate: str,
) -> np.ndarray:
    """Compatibility adapter for the shared height product."""
    return load_height(
        cache_file,
        geopotential_dir,
        height_variable,
        max_days=max_days,
        recompute=overwrite_intermediate,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
        aggregate=aggregate,
        spatial_window=None,
    )


def align_vertical_shapes(
    axis: VerticalAxis, profiles: dict[str, np.ndarray]
) -> tuple[VerticalAxis, dict[str, np.ndarray]]:
    n_levels = min(axis.values.size, *(arr.shape[0] for arr in profiles.values()))
    axis_new = VerticalAxis(
        values=axis.values[:n_levels], label=axis.label, is_height_km=axis.is_height_km
    )
    prof_new = {exp: arr[:n_levels, :] for exp, arr in profiles.items()}
    return axis_new, prof_new


def run_profile_job(
    cache_file: str,
    experiment_dir: str,
    variable: str,
    max_days: int | None,
    overwrite_intermediate: bool,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
) -> tuple[np.ndarray, str]:
    mean, sample_file = load_or_compute_diurnal(
        cache_file=Path(cache_file),
        experiment_dir=Path(experiment_dir),
        variable=variable,
        max_days=max_days,
        overwrite_intermediate=overwrite_intermediate,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
    )
    return mean, str(sample_file)
