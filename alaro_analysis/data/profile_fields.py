"""Profile fields. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from alaro_analysis.analysis.updrafts import updraft_intensity, updraft_mass_flux
from alaro_analysis.common.models import SpatialWindow
from alaro_analysis.common.spatial import apply_spatial_window_to_array
from alaro_analysis.data.dataset_io import (
    nanmean_with_count,
    resolve_data_var_name,
    to_time_level_yx,
)
from alaro_analysis.data.profile_sources import UPDRAFT_MESH_VAR, UPDRAFT_OMEGA_VAR


def read_field_array(file_path: Path, requested_variable: str) -> np.ndarray:
    with xr.open_dataset(file_path, decode_times=False) as ds:
        var_name = resolve_data_var_name(ds, requested_variable, compact_match=True)
        return to_time_level_yx(
            np.asarray(ds[var_name].values, dtype=np.float64),
            ds[var_name].dims,
            file_path,
            var_name,
        )


def as_time_level_yx(arr: np.ndarray, file_path: Path) -> np.ndarray:
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        return arr[np.newaxis, :, :, :]
    raise ValueError(f"Expected 3D/4D array in {file_path}, got shape {arr.shape}")


def compute_updraft_derived_profile_from_files(
    omega_file: Path,
    mesh_file: Path,
    derived_variable: str,
    spatial_window: SpatialWindow,
) -> np.ndarray:
    name = derived_variable.strip().upper()

    omega = as_time_level_yx(read_field_array(omega_file, UPDRAFT_OMEGA_VAR), omega_file)
    mesh = as_time_level_yx(read_field_array(mesh_file, UPDRAFT_MESH_VAR), mesh_file)
    omega = apply_spatial_window_to_array(omega, spatial_window, omega_file)
    mesh = apply_spatial_window_to_array(mesh, spatial_window, mesh_file)

    if omega.shape != mesh.shape:
        raise ValueError(
            f"Shape mismatch for derived updraft profile: "
            f"{omega_file} {omega.shape} vs {mesh_file} {mesh.shape}"
        )

    if name == "UPDRAFT_FLUX":
        # M_u = - \sigma_u * \omega_u / g
        # Since omega is negative for updrafts, `-omega` is positive.
        # We also enforce that we only care where mesh > 0.
        flux = updraft_mass_flux(omega, mesh)
        profile, _ = nanmean_with_count(flux, axis=(0, 2, 3))
        return profile

    if name == "UPDRAFT_EXTENT":
        profile, _ = nanmean_with_count(mesh, axis=(0, 2, 3))
        return profile

    if name == "UPDRAFT_INTENSITY":
        # Intensity is |UD_OMEGA| where the updraft is active. Inactive
        # in-domain cells contribute zero so this remains a domain-mean
        # diagnostic, matching UPDRAFT_EXTENT and UPDRAFT_FLUX.
        omega_abs = updraft_intensity(omega, mesh)
        profile, _ = nanmean_with_count(omega_abs, axis=(0, 2, 3))
        return profile

    raise ValueError(f"Unknown derived updraft variable: {derived_variable}")
