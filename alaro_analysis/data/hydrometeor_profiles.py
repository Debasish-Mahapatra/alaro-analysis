"""Hydrometeor profiles. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from alaro_analysis.analysis import profiles as profile_analysis
from alaro_analysis.common.models import PeriodSpec, SpatialWindow, VerticalAxis
from alaro_analysis.common.timeparse import has_pf_subdirs
from alaro_analysis.data.cache import (
    build_height_cache_file,
    load_diurnal_profile_cache,
)
from alaro_analysis.data.dataset_io import read_vertical_profile
from alaro_analysis.data.profile_products import (
    compute_profile,
    load_height,
    load_profile,
)
from alaro_analysis.data.profile_sources import (
    UPDRAFT_OMEGA_VAR,
    is_updraft_derived_variable,
)


def compute_diurnal_profile(
    experiment_dir: Path,
    variable: str,
    max_days: int | None,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
    spatial_window: SpatialWindow,
) -> tuple[np.ndarray, np.ndarray, int, Path]:
    """Read raw or derived fields using the common diurnal reducer."""
    return compute_profile(
        experiment_dir,
        variable,
        max_days=max_days,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
        spatial_window=spatial_window,
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
    spatial_window: SpatialWindow | None = None,
):
    """Compatibility adapter for the shared, versioned profile product."""
    if (
        variable.upper() in {"TEMPERATURE", "TEMPERATUR"}
        and not has_pf_subdirs(experiment_dir / variable)
        and cache_file.exists()
        and not overwrite_intermediate
        and max_days is None
    ):
        mean, counts, _, sample = load_diurnal_profile_cache(cache_file)
        if mean.ndim != 2 or mean.shape[1] != 24:
            raise ValueError(f"Invalid cached temperature shape in {cache_file}: {mean.shape}")
        positive = counts[counts > 0] if counts is not None else np.array([], dtype=int)
        low, high = (int(positive.min()), int(positive.max())) if positive.size else (0, 0)
        print(
            f"[temperature overlay] Sources offline; using stored profile {cache_file}", flush=True
        )
        return mean, sample or experiment_dir / variable, low, high
    product = load_profile(
        cache_file,
        experiment_dir,
        variable,
        max_days=max_days,
        recompute=overwrite_intermediate,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
        spatial_window=spatial_window,
    )
    positive = product.counts[product.counts > 0]
    low, high = (int(positive.min()), int(positive.max())) if positive.size else (0, 0)
    return product.mean, product.sample_file, low, high


def load_or_compute_height(
    cache_file: Path,
    geopotential_dir: Path,
    height_variable: str,
    max_days: int | None,
    overwrite_intermediate: bool,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
    aggregate: str,
    spatial_window: SpatialWindow | None = None,
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
        spatial_window=spatial_window,
    )


def infer_axis_from_coord(
    sample_file: Path, variable: str, spatial_window: SpatialWindow
) -> VerticalAxis:
    profile, coord = read_vertical_profile(
        sample_file,
        variable,
        spatial_window=spatial_window,
        compact_match=True,
    )
    n_levels = profile.size
    if coord is None or coord.size != n_levels:
        return VerticalAxis(
            values=np.arange(n_levels, dtype=np.float64),
            label="Model level",
            is_height_km=False,
        )

    finite = np.isfinite(coord)
    if np.sum(finite) < max(2, n_levels // 2):
        return VerticalAxis(
            values=np.arange(n_levels, dtype=np.float64),
            label="Model level",
            is_height_km=False,
        )

    cmax = float(np.nanmax(np.abs(coord[finite])))
    cmin = float(np.nanmin(np.abs(coord[finite])))
    is_integerish = np.allclose(coord[finite], np.round(coord[finite]), atol=1e-6)

    if cmax <= 60.0 and not is_integerish:
        return VerticalAxis(values=coord, label="Height (km)", is_height_km=True)
    if 10.0 <= cmin and cmax <= 1200.0:
        return VerticalAxis(values=coord, label="Pressure (hPa)", is_height_km=False)
    if 1200.0 < cmax <= 20000.0:
        return VerticalAxis(values=coord / 1000.0, label="Height (km)", is_height_km=True)
    return VerticalAxis(values=coord, label="Model level", is_height_km=False)


def align_vertical_shapes(
    axis: VerticalAxis, profiles: dict[str, np.ndarray], variable: str, period_key: str
) -> tuple[VerticalAxis, dict[str, np.ndarray]]:
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
    variable: str,
    period_key: str,
    experiment: str,
) -> tuple[VerticalAxis, np.ndarray]:
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


def build_vertical_axis(
    args: argparse.Namespace,
    intermediate_dir: Path,
    period: PeriodSpec,
    experiment: str,
    geopotential_dir: Path,
    sample_file: Path,
    variable: str,
    spatial_window: SpatialWindow,
    spatial_tag: str,
) -> VerticalAxis:
    axis_source_variable = UPDRAFT_OMEGA_VAR if is_updraft_derived_variable(variable) else variable

    if args.height_mode in ("geopotential", "auto"):
        try:
            height_cache = build_height_cache_file(
                intermediate_dir=intermediate_dir,
                period_subdir=period.output_subdir,
                experiment=experiment,
                aggregate=args.height_aggregate,
                spatial_tag=spatial_tag,
            )
            height_m = load_or_compute_height(
                cache_file=height_cache,
                geopotential_dir=geopotential_dir,
                height_variable=args.height_variable,
                max_days=args.max_days,
                overwrite_intermediate=args.overwrite_intermediate,
                allowed_months=period.allowed_months,
                utc_offset_hours=args.utc_offset_hours,
                aggregate=args.height_aggregate,
                spatial_window=spatial_window,
            )
            return VerticalAxis(
                values=np.asarray(height_m, dtype=np.float64) / 1000.0,
                label="Height (km)",
                is_height_km=True,
            )
        except Exception as exc:
            if args.height_mode == "geopotential":
                raise
            print(
                f"[warn] Geopotential axis failed for {experiment} ({exc}); falling back.",
                flush=True,
            )

    if args.height_mode in ("coord", "auto"):
        try:
            return infer_axis_from_coord(
                sample_file, axis_source_variable, spatial_window=spatial_window
            )
        except Exception as exc:
            if args.height_mode == "coord":
                raise
            print(
                f"[warn] Coordinate axis failed for {experiment} ({exc}); falling back.",
                flush=True,
            )

    profile, _ = read_vertical_profile(
        sample_file, axis_source_variable, spatial_window=spatial_window
    )
    return VerticalAxis(
        values=np.arange(profile.size, dtype=np.float64),
        label="Model level",
        is_height_km=False,
    )


def run_profile_job(
    cache_file: str,
    experiment_dir: str,
    variable: str,
    max_days: int | None,
    overwrite_intermediate: bool,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
    y_start: int | None,
    y_end: int | None,
    x_start: int | None,
    x_end: int | None,
) -> tuple[np.ndarray, str, int, int]:
    spatial_window = SpatialWindow(
        y_start=y_start,
        y_end=y_end,
        x_start=x_start,
        x_end=x_end,
    )
    mean, sample_file, count_min, count_max = load_or_compute_diurnal(
        cache_file=Path(cache_file),
        experiment_dir=Path(experiment_dir),
        variable=variable,
        max_days=max_days,
        overwrite_intermediate=overwrite_intermediate,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
        spatial_window=spatial_window,
    )
    return mean, str(sample_file), count_min, count_max
