"""Cache-aware wrappers for expensive profile computations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alaro_analysis.common.models import SpatialWindow
from alaro_analysis.data.cache import (
    load_diurnal_profile_cache,
    load_height_profile_cache,
    save_diurnal_profile_cache,
    save_height_profile_cache,
)

from .profiles import compute_diurnal_profile, compute_geopotential_height_profile


def load_or_compute_diurnal(
    cache_file: Path,
    variable_dir: Path,
    variable: str,
    *,
    max_days: int | None = None,
    overwrite: bool = False,
    allowed_months: tuple[int, ...] | None = None,
    utc_offset_hours: int = -4,
    spatial_window: SpatialWindow | None = None,
) -> tuple[np.ndarray, Path, int, int]:
    """Load a cached diurnal profile or compute from scratch.

    Returns ``(mean, sample_file, count_min, count_max)``.
    """
    use_cache = cache_file.exists() and (not overwrite)
    if use_cache:
        mean, counts, _, sample_file = load_diurnal_profile_cache(cache_file)
        if sample_file is None:
            raise ValueError(f"Missing sample_file in cache: {cache_file}")
        if counts is not None:
            positive = counts[counts > 0]
            if positive.size > 0:
                return mean, sample_file, int(np.min(positive)), int(np.max(positive))
        return mean, sample_file, 0, 0

    mean, counts, n_files, sample_file = compute_diurnal_profile(
        variable_dir=variable_dir,
        variable=variable,
        max_days=max_days,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
        spatial_window=spatial_window,
    )

    if max_days is None:
        save_diurnal_profile_cache(
            cache_file,
            mean=mean,
            counts=counts,
            n_files=n_files,
            sample_file=sample_file,
        )
    positive = counts[counts > 0]
    if positive.size == 0:
        return mean, sample_file, 0, 0
    return mean, sample_file, int(np.min(positive)), int(np.max(positive))


def load_or_compute_height(
    cache_file: Path,
    geopotential_dir: Path,
    height_variable: str,
    *,
    max_days: int | None = None,
    overwrite: bool = False,
    allowed_months: tuple[int, ...] | None = None,
    utc_offset_hours: int = -4,
    aggregate: str = "first",
    spatial_window: SpatialWindow | None = None,
) -> np.ndarray:
    """Load a cached height profile or compute from geopotential files.

    Returns ``height_m`` as a 1-D array in metres.
    """
    use_cache = cache_file.exists() and (not overwrite)
    if use_cache:
        return load_height_profile_cache(cache_file)

    height_m, n_files = compute_geopotential_height_profile(
        geopotential_dir=geopotential_dir,
        height_variable=height_variable,
        max_days=max_days,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
        aggregate=aggregate,
        spatial_window=spatial_window,
    )
    if max_days is None:
        save_height_profile_cache(cache_file, height_m=height_m, n_files=n_files)
    return height_m
