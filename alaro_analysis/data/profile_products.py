"""Versioned diurnal and height products shared by the profile commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alaro_analysis.analysis.profiles import (
    compute_diurnal_profile,
    compute_geopotential_height_profile,
)
from alaro_analysis.analysis.updrafts import UPDRAFT_CALCULATION_VERSION
from alaro_analysis.common.models import SpatialWindow
from alaro_analysis.data.cache import (
    load_diurnal_profile_cache,
    load_height_profile_cache,
    read_cache_signature,
    save_diurnal_profile_cache,
    save_height_profile_cache,
    signature,
    source_fingerprint,
)
from alaro_analysis.data.discovery import collect_file_records
from alaro_analysis.data.profile_fields import (
    compute_updraft_derived_profile_from_files,
)
from alaro_analysis.data.profile_sources import (
    UPDRAFT_MESH_VAR,
    UPDRAFT_OMEGA_VAR,
    is_updraft_derived_variable,
)


@dataclass(frozen=True)
class DiurnalProfile:
    """Mean and contributing-profile counts, both shaped (level, local_hour)."""

    mean: np.ndarray
    counts: np.ndarray
    n_files: int
    sample_file: Path


def profile_records(root, variable, max_days, allowed_months, utc_offset_hours):
    driver = UPDRAFT_OMEGA_VAR if is_updraft_derived_variable(variable) else variable
    return collect_file_records(root / driver, max_days, allowed_months, utc_offset_hours)


def compute_profile(
    root: Path,
    variable: str,
    *,
    max_days=None,
    allowed_months=None,
    utc_offset_hours=-4,
    spatial_window=None,
    records=None,
):
    """Compute a raw or derived profile with the same sampling and reduction."""
    if records is None:
        records = profile_records(root, variable, max_days, allowed_months, utc_offset_hours)
    reader = None
    if is_updraft_derived_variable(variable):
        window = spatial_window or SpatialWindow(None, None, None, None)

        def reader(path):
            mesh = root / UPDRAFT_MESH_VAR / path.parent.name / path.name
            return compute_updraft_derived_profile_from_files(path, mesh, variable, window)

    return compute_diurnal_profile(
        root / variable,
        variable,
        records=records,
        spatial_window=spatial_window,
        profile_reader=reader,
    )


def load_profile(
    cache: Path,
    root: Path,
    variable: str,
    *,
    max_days=None,
    allowed_months=None,
    utc_offset_hours=-4,
    spatial_window=None,
    recompute=False,
) -> DiurnalProfile:
    records = profile_records(root, variable, max_days, allowed_months, utc_offset_hours)
    sources = [path for _, path in records]
    if is_updraft_derived_variable(variable):
        sources += [root / UPDRAFT_MESH_VAR / path.parent.name / path.name for _, path in records]
    settings = {
        "schema_version": 2,
        "calculation_version": UPDRAFT_CALCULATION_VERSION,
        "variable": variable,
        "source_inventory": source_fingerprint(sources),
        "max_days": max_days,
        "allowed_months": allowed_months,
        "utc_offset_hours": utc_offset_hours,
        "spatial_window": spatial_window or SpatialWindow(None, None, None, None),
        "units": {
            "UPDRAFT_FLUX": "kg m-2 s-1",
            "UPDRAFT_EXTENT": "1",
            "UPDRAFT_INTENSITY": "Pa s-1",
        }.get(variable, "source variable units"),
    }
    sig = signature(settings)
    if not recompute and cache.is_file() and read_cache_signature(cache) == sig:
        mean, counts, n_files, sample = load_diurnal_profile_cache(cache)
        if counts is not None and n_files is not None and sample is not None:
            if mean.shape == counts.shape and mean.ndim == 2 and mean.shape[1] == 24:
                return DiurnalProfile(mean, counts, n_files, sample)
    product = DiurnalProfile(
        *compute_profile(root, variable, records=records, spatial_window=spatial_window)
    )
    # A short exploratory run must not replace the full-period product.
    if max_days is None:
        save_diurnal_profile_cache(
            cache,
            mean=product.mean,
            counts=product.counts,
            n_files=product.n_files,
            sample_file=product.sample_file,
            sig=sig,
            metadata=settings,
        )
    return product


def load_height(
    cache: Path,
    root: Path,
    variable: str,
    *,
    max_days=None,
    allowed_months=None,
    utc_offset_hours=-4,
    spatial_window=None,
    aggregate="first",
    recompute=False,
) -> np.ndarray:
    if aggregate not in {"first", "mean-all"}:
        raise ValueError(f"Unknown height aggregation {aggregate!r}")
    records = collect_file_records(root, max_days, allowed_months, utc_offset_hours)
    selected = records[:1] if aggregate == "first" else records
    sig = signature(
        {
            "schema_version": 2,
            "variable": variable,
            "aggregate": aggregate,
            "sources": source_fingerprint(selected),
            "spatial_window": spatial_window,
        }
    )
    if not recompute and cache.is_file() and read_cache_signature(cache) == sig:
        return load_height_profile_cache(cache)
    height, n_files = compute_geopotential_height_profile(
        root,
        variable,
        max_days=max_days,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
        spatial_window=spatial_window,
        aggregate=aggregate,
    )
    if max_days is None:
        save_height_profile_cache(cache, height_m=height, n_files=n_files, sig=sig)
    return height
