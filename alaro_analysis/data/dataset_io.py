from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import xarray as xr

from alaro_analysis.common.models import SpatialWindow
from alaro_analysis.common.spatial import apply_spatial_window_to_array


def _compact_token(token: str) -> str:
    return "".join(ch for ch in token if ch.isalnum()).lower()


def resolve_data_var_name(
    ds: xr.Dataset,
    requested: str,
    *,
    token_normalizer: Callable[[str], str] | None = None,
    compact_match: bool = False,
) -> str:
    if requested in ds.data_vars:
        return requested

    req_lower = requested.lower()
    for name in ds.data_vars:
        if name.lower() == req_lower:
            return name

    if token_normalizer is not None:
        req_token = token_normalizer(requested)
        token_hits = [
            name for name in ds.data_vars if token_normalizer(name) == req_token
        ]
        if len(token_hits) == 1:
            return token_hits[0]

    if compact_match:
        req_compact = _compact_token(requested)
        compact_hits = [name for name in ds.data_vars if _compact_token(name) == req_compact]
        if len(compact_hits) == 1:
            return compact_hits[0]

    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars.keys()))

    raise KeyError(
        f"Variable '{requested}' not found. Available: {list(ds.data_vars.keys())}"
    )


def select_data_var_name(
    ds: xr.Dataset,
    requested: str | None = None,
    *,
    preferred: Sequence[str] = ("PRESSURE", "GEOPOTENTIEL", "TEMPERATURE"),
    token_normalizer: Callable[[str], str] | None = None,
    compact_match: bool = False,
) -> str:
    if requested is not None:
        return resolve_data_var_name(
            ds,
            requested,
            token_normalizer=token_normalizer,
            compact_match=compact_match,
        )

    for name in preferred:
        if name in ds.data_vars:
            return name

    if not ds.data_vars:
        raise ValueError("Dataset has no data variables.")

    return next(iter(ds.data_vars.keys()))


def nanmean_with_count(
    data: np.ndarray,
    axis: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(data)
    count = np.sum(valid, axis=axis)
    total = np.nansum(data, axis=axis)
    mean = np.full(total.shape, np.nan, dtype=np.float64)
    nonzero = count > 0
    mean[nonzero] = total[nonzero] / count[nonzero]
    return mean, count.astype(np.int64)


def to_time_level_yx(
    arr: np.ndarray,
    dims: Sequence[str],
    file_path: Path,
    var_name: str,
) -> np.ndarray:
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        first = dims[0].lower() if dims else ""
        if first == "time":
            return arr[:, np.newaxis, :, :]
        return arr[np.newaxis, :, :, :]
    if arr.ndim == 2:
        return arr[np.newaxis, np.newaxis, :, :]
    raise ValueError(
        f"Unsupported shape for {var_name} in {file_path}: {arr.shape} (dims={tuple(dims)})"
    )


def read_time_level_yx(
    file_path: Path,
    requested_variable: str,
    spatial_window: SpatialWindow,
    *,
    token_normalizer: Callable[[str], str] | None = None,
    compact_match: bool = False,
) -> np.ndarray:
    engine = "netcdf4" if "".join(file_path.suffixes).endswith(".nc") else None
    with xr.open_dataset(file_path, decode_times=False, engine=engine) as ds:
        var_name = resolve_data_var_name(
            ds,
            requested_variable,
            token_normalizer=token_normalizer,
            compact_match=compact_match,
        )
        da = ds[var_name]
        arr = np.asarray(da.values, dtype=np.float64)
        arr = apply_spatial_window_to_array(arr, spatial_window, file_path)
        arr = to_time_level_yx(arr, da.dims, file_path, requested_variable)
    return arr


def read_vertical_profile(
    file_path: Path,
    requested_variable: str,
    *,
    spatial_window: SpatialWindow | None = None,
    token_normalizer: Callable[[str], str] | None = None,
    compact_match: bool = False,
    force_time_level_yx: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    if force_time_level_yx:
        if spatial_window is None:
            raise ValueError("spatial_window is required when force_time_level_yx=True")
        arr = read_time_level_yx(
            file_path,
            requested_variable,
            spatial_window=spatial_window,
            token_normalizer=token_normalizer,
            compact_match=compact_match,
        )
        profile, _ = nanmean_with_count(arr, axis=(0, 2, 3))
        return np.asarray(profile, dtype=np.float64), None

    engine = "netcdf4" if "".join(file_path.suffixes).endswith(".nc") else None
    with xr.open_dataset(file_path, decode_times=False, engine=engine) as ds:
        var_name = resolve_data_var_name(
            ds,
            requested_variable,
            token_normalizer=token_normalizer,
            compact_match=compact_match,
        )
        arr = np.asarray(ds[var_name].values, dtype=np.float64)
        if spatial_window is not None:
            arr = apply_spatial_window_to_array(arr, spatial_window, file_path)
        dims = tuple(ds[var_name].dims)

        if arr.ndim == 4:
            profile, _ = nanmean_with_count(arr, axis=(0, 2, 3))
            vertical_dim = dims[1]
        elif arr.ndim == 3:
            profile, _ = nanmean_with_count(arr, axis=(1, 2))
            vertical_dim = dims[0]
        elif arr.ndim == 2:
            profile, _ = nanmean_with_count(arr, axis=(1,))
            vertical_dim = dims[0]
        elif arr.ndim == 1:
            profile = arr
            vertical_dim = dims[0]
        else:
            raise ValueError(f"Unsupported shape {arr.shape} in {file_path}")

        vertical_coord = None
        if vertical_dim in ds.coords:
            coord = np.asarray(ds[vertical_dim].values, dtype=np.float64)
            if coord.ndim == 1 and coord.size == profile.size:
                vertical_coord = coord

    return np.asarray(profile, dtype=np.float64), vertical_coord
