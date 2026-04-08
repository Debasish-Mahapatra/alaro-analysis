"""
Reusable diurnal profile computation building blocks.

These functions handle the core patterns of:
- Accumulating vertical profiles by hour-of-day (profile_hour_accumulate)
- Accumulating scalar lines by hour-of-day (line_hour_accumulate)
- Computing mean diurnal profiles from masked-NetCDF file collections
- Computing geopotential height profiles
- Computing surface (2D) diurnal cycles
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from alaro_analysis.common.models import SpatialWindow, VerticalAxis
from alaro_analysis.data.dataset_io import (
    nanmean_with_count,
    read_time_level_yx,
    read_vertical_profile,
)
from alaro_analysis.data.discovery import collect_file_records


# ---------------------------------------------------------------------------
# Generic accumulation helpers
# ---------------------------------------------------------------------------


def profile_hour_accumulate(
    records: list[tuple[int, Path]],
    profile_reader: Callable[[Path], dict[str, np.ndarray] | None],
    progress_tag: str = "",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Accumulate vertical profiles by hour-of-day.

    *profile_reader(file_path)* must return ``dict[str, ndarray(level,)]``
    or ``None`` to skip a file.

    Returns ``(sums, counts, n_used)`` where each dict is keyed by
    diagnostic name with arrays of shape ``(n_levels, 24)``.
    """
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    used = 0
    for idx, (hour, file_path) in enumerate(records, start=1):
        result = profile_reader(file_path)
        if result is None:
            continue

        used += 1
        for diag, profile in result.items():
            if diag not in sums:
                sums[diag] = np.zeros((profile.size, 24), dtype=np.float64)
                counts[diag] = np.zeros((profile.size, 24), dtype=np.int64)
            valid = np.isfinite(profile)
            sums[diag][valid, hour] += profile[valid]
            counts[diag][valid, hour] += 1

        if idx % 2000 == 0 or idx == len(records):
            print(f"[{progress_tag}] {idx}/{len(records)} files", flush=True)
    return sums, counts, used


def finalize_profile_means(
    sums: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Convert accumulated sums/counts into NaN-aware means."""
    out: dict[str, np.ndarray] = {}
    for diag, sum_arr in sums.items():
        cnt_arr = counts[diag]
        mean = np.full(sum_arr.shape, np.nan, dtype=np.float64)
        valid = cnt_arr > 0
        mean[valid] = sum_arr[valid] / cnt_arr[valid]
        out[diag] = mean
    return out


def line_hour_accumulate(
    records: list[tuple[int, Path]],
    line_reader: Callable[[Path], dict[str, float] | None],
    progress_tag: str = "",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Accumulate scalar diagnostics by hour-of-day.

    *line_reader(file_path)* must return ``dict[str, float]``
    or ``None`` to skip a file.

    Returns ``(sums, counts, n_used)`` where each dict is keyed by
    diagnostic name with arrays of shape ``(24,)``.
    """
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    used = 0
    for idx, (hour, file_path) in enumerate(records, start=1):
        result = line_reader(file_path)
        if result is None:
            continue
        used += 1

        for diag, value in result.items():
            if diag not in sums:
                sums[diag] = np.zeros((24,), dtype=np.float64)
                counts[diag] = np.zeros((24,), dtype=np.int64)
            if np.isfinite(value):
                sums[diag][hour] += float(value)
                counts[diag][hour] += 1

        if idx % 2000 == 0 or idx == len(records):
            print(f"[{progress_tag}] {idx}/{len(records)} files", flush=True)
    return sums, counts, used


def finalize_line_means(
    sums: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Convert accumulated sums/counts into NaN-aware means."""
    out: dict[str, np.ndarray] = {}
    for diag, sum_arr in sums.items():
        cnt_arr = counts[diag]
        mean = np.full(sum_arr.shape, np.nan, dtype=np.float64)
        valid = cnt_arr > 0
        mean[valid] = sum_arr[valid] / cnt_arr[valid]
        out[diag] = mean
    return out


# ---------------------------------------------------------------------------
# High-level profile builders
# ---------------------------------------------------------------------------


def compute_diurnal_profile(
    variable_dir: Path,
    variable: str,
    *,
    max_days: int | None = None,
    allowed_months: tuple[int, ...] | None = None,
    utc_offset_hours: int = -4,
    spatial_window: SpatialWindow | None = None,
    compact_match: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, Path]:
    """Compute the mean diurnal vertical profile for *variable*.

    Returns ``(mean, counts, n_files, sample_file)`` where *mean* and
    *counts* have shape ``(n_levels, 24)``.
    """
    if spatial_window is None:
        spatial_window = SpatialWindow(y_start=None, y_end=None, x_start=None, x_end=None)

    records = collect_file_records(
        variable_dir=variable_dir,
        max_days=max_days,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
    )
    if not records:
        raise RuntimeError(f"No valid +0000..+0023 files found in {variable_dir}")

    first_profile, _ = read_vertical_profile(
        records[0][1],
        variable,
        spatial_window=spatial_window,
        compact_match=compact_match,
    )

    n_levels = first_profile.size
    sums = np.zeros((n_levels, 24), dtype=np.float64)
    counts = np.zeros((n_levels, 24), dtype=np.int64)

    for idx, (local_hour, file_path) in enumerate(records, start=1):
        profile, _ = read_vertical_profile(
            file_path,
            variable,
            spatial_window=spatial_window,
            compact_match=compact_match,
        )

        if profile.size != n_levels:
            raise ValueError(
                f"Inconsistent vertical levels in {file_path}: "
                f"{profile.size} vs expected {n_levels}"
            )
        valid = np.isfinite(profile)
        sums[valid, local_hour] += profile[valid]
        counts[valid, local_hour] += 1

        if idx % 2000 == 0 or idx == len(records):
            print(
                f"[{variable_dir.parent.name}/{variable}] {idx}/{len(records)} files",
                flush=True,
            )

    mean = np.full_like(sums, np.nan)
    nonzero = counts > 0
    mean[nonzero] = sums[nonzero] / counts[nonzero]
    return mean, counts, len(records), records[0][1]


def compute_geopotential_height_profile(
    geopotential_dir: Path,
    height_variable: str,
    *,
    max_days: int | None = None,
    allowed_months: tuple[int, ...] | None = None,
    utc_offset_hours: int = -4,
    aggregate: str = "first",
    spatial_window: SpatialWindow | None = None,
    compact_match: bool = True,
) -> tuple[np.ndarray, int]:
    """Compute height axis from geopotential files.

    *aggregate* can be ``"first"`` (fast, single file) or ``"mean-all"``
    (averaged over all files in the period).

    Returns ``(height_m, n_files_used)``.
    """
    if spatial_window is None:
        spatial_window = SpatialWindow(y_start=None, y_end=None, x_start=None, x_end=None)

    records = collect_file_records(
        variable_dir=geopotential_dir,
        max_days=max_days,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
    )
    if not records:
        raise RuntimeError(f"No valid geopotential files found in {geopotential_dir}")

    if aggregate == "first":
        profile, _ = read_vertical_profile(
            records[0][1],
            height_variable,
            spatial_window=spatial_window,
            compact_match=compact_match,
        )
        return profile, 1

    first, _ = read_vertical_profile(
        records[0][1],
        height_variable,
        spatial_window=spatial_window,
        compact_match=compact_match,
    )
    sums = np.zeros_like(first, dtype=np.float64)
    counts = np.zeros_like(first, dtype=np.int64)

    for idx, (_, file_path) in enumerate(records, start=1):
        profile, _ = read_vertical_profile(
            file_path,
            height_variable,
            spatial_window=spatial_window,
            compact_match=compact_match,
        )
        valid = np.isfinite(profile)
        sums[valid] += profile[valid]
        counts[valid] += 1

        if idx % 4000 == 0 or idx == len(records):
            print(
                f"[{geopotential_dir.parent.name}/GEOPOTENTIEL] "
                f"{idx}/{len(records)} files",
                flush=True,
            )

    mean = np.full_like(sums, np.nan)
    nonzero = counts > 0
    mean[nonzero] = sums[nonzero] / counts[nonzero]
    return mean, len(records)


def compute_surface_diurnal_cycle(
    records: list[tuple[int, Path]],
    variable_name: str,
    *,
    spatial_window: SpatialWindow | None = None,
    token_normalizer: Callable[[str], str] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute 24-hour mean surface diurnal cycle.

    Returns ``(mean, counts, n_used)`` where each has shape ``(24,)``.
    """
    if spatial_window is None:
        spatial_window = SpatialWindow(y_start=None, y_end=None, x_start=None, x_end=None)

    sums = np.zeros((24,), dtype=np.float64)
    counts = np.zeros((24,), dtype=np.int64)
    used = 0

    for idx, (hour, file_path) in enumerate(records, start=1):
        arr = read_time_level_yx(
            file_path,
            variable_name,
            spatial_window=spatial_window,
            token_normalizer=token_normalizer,
        )
        finite = arr[np.isfinite(arr)]
        value = float(np.mean(finite)) if finite.size > 0 else float("nan")
        used += 1
        if np.isfinite(value):
            sums[hour] += value
            counts[hour] += 1

        if idx % 2000 == 0 or idx == len(records):
            print(f"[{variable_name}] {idx}/{len(records)} files", flush=True)

    mean = np.full((24,), np.nan, dtype=np.float64)
    valid = counts > 0
    mean[valid] = sums[valid] / counts[valid]
    return mean, counts, used


# ---------------------------------------------------------------------------
# Vertical axis alignment utilities
# ---------------------------------------------------------------------------


def align_vertical_shapes(
    axis: VerticalAxis,
    profiles: dict[str, np.ndarray],
    *,
    variable: str = "",
    period_key: str = "",
) -> tuple[VerticalAxis, dict[str, np.ndarray]]:
    """Truncate axis and profiles to the minimum common level count."""
    n_levels = min(
        axis.values.size,
        *(profile.shape[0] for profile in profiles.values()),
    )
    mismatch = axis.values.size != n_levels or any(
        profile.shape[0] != n_levels for profile in profiles.values()
    )
    if mismatch:
        print(
            f"[warn] Vertical mismatch for {variable} ({period_key}); "
            f"truncating to {n_levels} levels.",
            flush=True,
        )
    axis_new = VerticalAxis(
        values=axis.values[:n_levels],
        label=axis.label,
        is_height_km=axis.is_height_km,
    )
    profiles_new = {exp: arr[:n_levels, :] for exp, arr in profiles.items()}
    return axis_new, profiles_new


def align_axis_and_profile(
    axis: VerticalAxis,
    profile: np.ndarray,
    *,
    variable: str = "",
    period_key: str = "",
    experiment: str = "",
) -> tuple[VerticalAxis, np.ndarray]:
    """Truncate a single axis+profile pair to the minimum common size."""
    n_levels = min(axis.values.size, profile.shape[0])
    if axis.values.size != profile.shape[0]:
        print(
            f"[warn] Vertical mismatch for {variable} ({period_key}, {experiment}); "
            f"truncating to {n_levels} levels.",
            flush=True,
        )
    axis_new = VerticalAxis(
        values=axis.values[:n_levels],
        label=axis.label,
        is_height_km=axis.is_height_km,
    )
    return axis_new, profile[:n_levels, :]
