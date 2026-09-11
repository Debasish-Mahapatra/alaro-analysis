"""Dsd profiles. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from alaro_analysis.dsd.surface import PF_DAY_RE, PF_FILE_RE


def discover_vertical_records(
    experiment: str,
    leads: tuple[int, ...] | None,
    netcdf_root: Path,
    max_days: int | None,
) -> list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]]:
    base = netcdf_root / experiment / "masked-netcdf"
    ref_dir = base / "RAIN"
    if not ref_dir.exists():
        raise FileNotFoundError(f"Missing variable folder: {ref_dir}")

    needed = ["RAIN", "TEMPERATURE", "PRESSURE", "GEOPOTENTIEL"]
    if experiment == "2mom":
        needed.append("PNR")

    day_dirs = sorted(d for d in ref_dir.iterdir() if d.is_dir() and PF_DAY_RE.match(d.name))
    if max_days is not None:
        day_dirs = day_dirs[:max_days]
    lead_set = set(leads) if leads is not None else None

    records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]] = []
    for day_dir in day_dirs:
        day_match = PF_DAY_RE.match(day_dir.name)
        if not day_match:
            continue
        init_dt = datetime.strptime(day_match.group(1), "%Y%m%d")
        for path in sorted(day_dir.iterdir()):
            file_match = PF_FILE_RE.match(path.name)
            if not file_match:
                continue
            lead = int(file_match.group(1))
            if lead_set is not None and lead not in lead_set:
                continue

            paths: dict[str, Path] = {}
            for var in needed:
                candidate = base / var / day_dir.name / path.name
                if not candidate.exists():
                    break
                paths[var] = candidate
            else:
                valid_dt = init_dt + timedelta(hours=lead)
                records.append(
                    (np.datetime64(valid_dt, "s"), np.datetime64(init_dt, "s"), lead, paths)
                )
    records.sort(key=lambda rec: rec[0])
    return records


def _as_float_array(raw: Any) -> np.ndarray:
    if np.ma.isMaskedArray(raw):
        return np.asarray(raw.filled(np.nan), dtype=np.float64)
    return np.asarray(raw, dtype=np.float64)


def _to_level_yx(arr: np.ndarray, mask_shape: tuple[int, int], path: Path, var: str) -> np.ndarray:
    if arr.ndim == 4:
        if arr.shape[-2:] != mask_shape:
            raise ValueError(f"{var} in {path} has incompatible shape {arr.shape}")
        if arr.shape[0] == 1:
            return arr[0]
        valid = np.isfinite(arr)
        counts = np.sum(valid, axis=0)
        sums = np.sum(np.where(valid, arr, 0.0), axis=0)
        out = np.full(counts.shape, np.nan, dtype=np.float64)
        ok = counts > 0
        out[ok] = sums[ok] / counts[ok]
        return out
    if arr.ndim == 3:
        if arr.shape[-2:] != mask_shape:
            raise ValueError(f"{var} in {path} has incompatible shape {arr.shape}")
        return arr
    if arr.ndim == 2:
        if arr.shape != mask_shape:
            raise ValueError(f"{var} in {path} has incompatible shape {arr.shape}")
        return arr[np.newaxis, :, :]
    raise ValueError(f"Unexpected ndim for {var} in {path}: {arr.ndim}")


def read_masked_level_mean(path: Path, var: str, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(path) as ds:
        arr = _as_float_array(ds.variables[var][:])
    field = _to_level_yx(arr, mask.shape, path, var)
    valid = np.isfinite(field) & mask[np.newaxis, :, :]
    counts = np.sum(valid, axis=(1, 2)).astype(np.int64)
    sums = np.sum(np.where(valid, field, 0.0), axis=(1, 2))
    profile = np.full(counts.shape, np.nan, dtype=np.float64)
    ok = counts > 0
    profile[ok] = sums[ok] / counts[ok]
    return profile, counts


def read_level_yx(path: Path, variable: str, mask_shape: tuple[int, int]) -> np.ndarray:
    with Dataset(path) as ds:
        arr = _as_float_array(ds.variables[variable][:])
    return _to_level_yx(arr, mask_shape, path, variable)
