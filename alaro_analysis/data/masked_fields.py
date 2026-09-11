"""Discovery and single-step reading of masked-NetCDF model fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from alaro_analysis.data.dataset_io import resolve_data_var_name, to_time_level_yx


def list_days(exp_dir: Path, variable: str) -> list[Path]:
    var_dir = exp_dir / "masked-netcdf" / variable
    if not var_dir.exists():
        return []
    return sorted(d for d in var_dir.iterdir() if d.is_dir() and d.name.startswith("pf"))


def list_steps(day_dir: Path, min_lead_hour: int = 3) -> list[Path]:
    steps = []
    for f in sorted(day_dir.iterdir()):
        if not f.name.endswith(".nc"):
            continue
        try:
            lead = int(f.stem.split("+")[1])
        except (IndexError, ValueError):
            continue
        if lead >= min_lead_hour:
            steps.append(f)
    return steps


def list_steps_in_range(
    day_dir: Path,
    lead_min: int,
    lead_max: int,
) -> list[Path]:
    """Return step files with lead_min <= lead <= lead_max."""
    steps = []
    for f in sorted(day_dir.iterdir()):
        if not f.name.endswith(".nc"):
            continue
        try:
            lead = int(f.stem.split("+")[1])
        except (IndexError, ValueError):
            continue
        if lead_min <= lead <= lead_max:
            steps.append(f)
    return steps


def read_field(filepath: Path) -> np.ndarray:
    """Read one model step as (level, y, x), resolving its variable directory."""
    with xr.open_dataset(filepath, decode_times=False) as ds:
        name = resolve_data_var_name(ds, filepath.parent.parent.name, allow_single_variable=True)
        values = to_time_level_yx(
            np.asarray(ds[name].values, dtype=np.float64), ds[name].dims, filepath, name
        )
        if values.shape[0] != 1:
            raise ValueError(f"Expected one time step in {filepath}, found {values.shape[0]}")
        return values[0]
