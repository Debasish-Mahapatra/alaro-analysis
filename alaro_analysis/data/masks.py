"""Masks. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr
from netCDF4 import Dataset


def _guess_coord_name(
    da: xr.DataArray,
    explicit_name: Optional[str],
    kind: str,
) -> str:
    if explicit_name is not None:
        if explicit_name in da.coords or explicit_name in da.dims:
            return explicit_name
        raise ValueError(f"{kind} coordinate '{explicit_name}' not found in mask variable")

    candidates = {
        "lat": ["lat", "latitude", "nav_lat", "y_lat"],
        "lon": ["lon", "longitude", "nav_lon", "x_lon"],
    }[kind]
    for name in candidates:
        if name in da.coords or name in da.dims:
            return name
    for name in list(da.coords) + list(da.dims):
        low = name.lower()
        if kind == "lat" and "lat" in low:
            return name
        if kind == "lon" and ("lon" in low or "long" in low):
            return name
    raise ValueError(f"Could not auto-detect {kind} coordinate in mask variable")


def _select_mask_variable(mask_ds: xr.Dataset, requested_var: Optional[str]) -> str:
    if requested_var:
        if requested_var not in mask_ds.data_vars:
            raise ValueError(
                f"Mask variable '{requested_var}' not found. Available: {list(mask_ds.data_vars)}"
            )
        return requested_var

    if not mask_ds.data_vars:
        raise ValueError("Mask NetCDF has no data variables")

    preferred = [
        "mask",
        "MASK",
        "spatial_mask",
        "radar_mask",
        "Radar_mask",
        "Radar_mask_latlon",
    ]
    for name in preferred:
        if name in mask_ds.data_vars:
            return name
    return list(mask_ds.data_vars)[0]


def _regrid_mask_to_model(
    mask_file: Path,
    target_lat: xr.DataArray,
    target_lon: xr.DataArray,
    mask_var: Optional[str],
    mask_lat_name: Optional[str],
    mask_lon_name: Optional[str],
    mask_threshold: float,
) -> Tuple[np.ndarray, str]:
    mask_ds = xr.open_dataset(mask_file)
    try:
        selected_var = _select_mask_variable(mask_ds, mask_var)
        mask_da = mask_ds[selected_var].squeeze(drop=True)
        # xarray.interp() requires numeric dtype; some masks are stored as bool.
        if mask_da.dtype == np.bool_:
            mask_da = mask_da.astype(np.float32)

        lat_name = _guess_coord_name(mask_da, mask_lat_name, "lat")
        lon_name = _guess_coord_name(mask_da, mask_lon_name, "lon")

        # Drop non-spatial dimensions by taking first index (common for time/band singleton dims).
        for dim in list(mask_da.dims):
            if dim not in {lat_name, lon_name}:
                if mask_da.sizes[dim] < 1:
                    raise ValueError(f"Mask dimension '{dim}' is empty")
                mask_da = mask_da.isel({dim: 0})

        if lat_name in mask_da.coords and lon_name in mask_da.coords:
            lat_coord = mask_da.coords[lat_name]
            lon_coord = mask_da.coords[lon_name]
        else:
            raise ValueError(
                f"Mask variable '{selected_var}' must expose '{lat_name}' and '{lon_name}' coordinates"
            )

        if lat_coord.ndim == 1 and lon_coord.ndim == 1:
            target_lat_da = xr.DataArray(target_lat.values, dims=("y", "x"))
            target_lon_da = xr.DataArray(target_lon.values, dims=("y", "x"))
            remapped = mask_da.interp(
                {lat_name: target_lat_da, lon_name: target_lon_da},
                method="nearest",
            )
        elif lat_coord.ndim == 2 and lon_coord.ndim == 2:
            try:
                from scipy.spatial import cKDTree
            except ImportError as exc:
                raise ImportError(
                    "Mask has 2D lat/lon coordinates and requires scipy for nearest-neighbor remap."
                ) from exc

            mask_values = np.asarray(mask_da.values)
            if mask_values.ndim != 2:
                raise ValueError(
                    "Mask with 2D coordinates must be a 2D variable after squeezing extra dims."
                )

            source_points = np.column_stack(
                [np.asarray(lon_coord.values).ravel(), np.asarray(lat_coord.values).ravel()]
            )
            target_points = np.column_stack(
                [np.asarray(target_lon.values).ravel(), np.asarray(target_lat.values).ravel()]
            )
            tree = cKDTree(source_points)
            _, nearest_idx = tree.query(target_points, k=1)
            remapped_values = mask_values.ravel()[nearest_idx].reshape(target_lon.shape)
            remapped = xr.DataArray(remapped_values, dims=("y", "x"))
        else:
            raise ValueError(
                "Unsupported mask coordinate layout. Expected lat/lon as 1D or 2D coordinates."
            )

        remapped_values = np.asarray(remapped.values, dtype=np.float64)
        mask_binary = remapped_values > float(mask_threshold)
        if not np.any(mask_binary):
            raise ValueError(f"Mask threshold ({mask_threshold}) removed all points on model grid.")
        return mask_binary, selected_var
    finally:
        mask_ds.close()


@dataclass(frozen=True)
class DomainMask:
    mask: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    selected_var: str

    @property
    def n_cells(self) -> int:
        return int(self.mask.sum())


def build_domain_mask_from_netcdf(
    sample_netcdf: Path,
    mask_file: Path,
    *,
    mask_var: str | None = None,
    mask_threshold: float = 0.5,
) -> DomainMask:
    with Dataset(sample_netcdf) as ds:
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    target_lat = xr.DataArray(lat, dims=("y", "x"))
    target_lon = xr.DataArray(lon, dims=("y", "x"))
    mask_array, selected = _regrid_mask_to_model(
        mask_file=mask_file,
        target_lat=target_lat,
        target_lon=target_lon,
        mask_var=mask_var,
        mask_lat_name=None,
        mask_lon_name=None,
        mask_threshold=mask_threshold,
    )
    return DomainMask(
        mask=np.asarray(mask_array, dtype=bool),
        lat=lat,
        lon=lon,
        selected_var=selected,
    )
