#!/usr/bin/env python3
"""
Extended diurnal diagnostics for ALARO masked NetCDF outputs.

Analyses implemented:
1) Downdraft/updraft diagnostics:
   - DD_EXTENT = mean(DD_MESH_FRAC)
   - DD_FLUX = mean(abs(DD_OMEGA * DD_MESH_FRAC))
   - DD_INTENSITY = mean(abs(DD_OMEGA))
   - DD_TO_UD_FLUX_RATIO = mean((DD_OMEGA * DD_MESH_FRAC) / (UD_OMEGA * UD_MESH_FRAC))
   - NET_CONVECTIVE_FLUX = mean(UD_OMEGA * UD_MESH_FRAC + DD_OMEGA * DD_MESH_FRAC)

2) Precipitation flux diagnostics:
   - CV_PREC_FLUX, ST_PREC_FLUX
   - TOTAL_PREC_FLUX = CV + ST
   - CONVECTIVE_PRECIP_FRACTION = CV / (CV + ST)
   - D(TOTAL_PREC_FLUX)/DZ (computed on height-aligned profiles)

3) Thermodynamic diagnostics:
   - THETA_E (Bolton-style approximation)
   - MSE = cp*T + Lv*q + g*z
   - BL theta-e gradient (0-2 km) as line plot

4) Freezing-level (KT273*) diagnostics:
   - KT273GRAUPEL
   - KT273DD_FLUX = KT273DD_OMEGA * KT273DD_MESH_FRA
   - KT273RAIN
   - KT273HUMI.SPECIF
   Stacked diurnal-cycle line figure with C1M/G1M/G2M overlays.

5) Column diagnostics:
   - COLUMN_CONDENSATE (kg m-2 proxy from q * dp/g)
   - SURFACE_PRECIP_FLUX (lowest model level of CV+ST)
   - RESIDENCE_TIME = COLUMN_CONDENSATE / SURFACE_PRECIP_FLUX
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import cmaps
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from alaro_analysis.common.constants import (
    CP_D,
    EPS,
    EXPERIMENTS,
    EXPERIMENT_LABELS,
    G,
    LV,
    P0,
    SEASONS,
)
from alaro_analysis.common.models import PeriodSpec, SpatialWindow, VerticalAxis
from alaro_analysis.common.naming import safe_name
from alaro_analysis.common.seasons import build_period_specs, resolve_seasons
from alaro_analysis.common.spatial import build_spatial_window, spatial_window_tag
from alaro_analysis.common.timeparse import (
    has_pf_subdirs,
    parse_month_from_day_name,
    parse_utc_hour_from_name,
)
from alaro_analysis.common.vertical import centers_to_edges

DEFAULT_CONTROL_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/masked-netcdf"
)
DEFAULT_GRAUPEL_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/graupel/masked-netcdf"
)
DEFAULT_2MOM_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/2mom/masked-netcdf"
)

DEFAULT_CONTROL_GEO_DIR = DEFAULT_CONTROL_DIR / "GEOPOTENTIEL"
DEFAULT_GRAUPEL_GEO_DIR = DEFAULT_GRAUPEL_DIR / "GEOPOTENTIEL"
DEFAULT_2MOM_GEO_DIR = DEFAULT_2MOM_DIR / "GEOPOTENTIEL"

DEFAULT_OUTPUT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/diagnostics")
DEFAULT_INTERMEDIATE_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/diagnostics"
)

ANALYSIS_CHOICES = ("downdraft", "precip", "thermo", "kt273", "column")
VAR_TOKEN_RE = re.compile(r"[^A-Za-z0-9]+")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extended diagnostics (downdraft/precip/thermo/KT273/column) from masked NetCDF."
    )
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL_DIR)
    parser.add_argument("--graupel-dir", type=Path, default=DEFAULT_GRAUPEL_DIR)
    parser.add_argument("--twomom-dir", type=Path, default=DEFAULT_2MOM_DIR)

    parser.add_argument("--control-geopotential-dir", type=Path, default=DEFAULT_CONTROL_GEO_DIR)
    parser.add_argument("--graupel-geopotential-dir", type=Path, default=DEFAULT_GRAUPEL_GEO_DIR)
    parser.add_argument("--twomom-geopotential-dir", type=Path, default=DEFAULT_2MOM_GEO_DIR)
    parser.add_argument("--height-variable", default="GEOPOTENTIEL")
    parser.add_argument(
        "--height-aggregate",
        choices=("first", "mean-all"),
        default="first",
        help="How to aggregate geopotential profiles by period.",
    )

    parser.add_argument(
        "--analyses",
        nargs="+",
        default=list(ANALYSIS_CHOICES),
        choices=ANALYSIS_CHOICES,
        help="Subset of analyses to run.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=list(SEASONS.keys()),
        help="Subset of seasons (wet dry ...). Use 'all' for all seasons.",
    )
    parser.add_argument(
        "--analysis-modes",
        nargs="+",
        default=("full", "seasonal"),
        choices=("full", "seasonal"),
        help="Run full 2-year analysis, seasonal analysis, or both.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--intermediate-dir", type=Path, default=DEFAULT_INTERMEDIATE_DIR)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--overwrite-intermediate",
        action="store_true",
        help="Overwrite existing intermediate cache files.",
    )
    parser.add_argument(
        "--recompute",
        dest="overwrite_intermediate",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--utc-offset-hours",
        type=int,
        default=-4,
        help="Local timezone offset from UTC (Amazon is -4).",
    )
    parser.add_argument(
        "--max-height-km",
        type=float,
        default=20.0,
        help="Upper y-limit for height-based plots.",
    )
    parser.add_argument(
        "--y-slice",
        default=None,
        help="Optional Python-style Y slice start:end for spatial averaging.",
    )
    parser.add_argument(
        "--x-slice",
        default=None,
        help="Optional Python-style X slice start:end for spatial averaging.",
    )
    parser.add_argument(
        "--list-variables",
        action="store_true",
        help="List discovered variables per experiment and exit.",
    )
    return parser.parse_args()


def normalize_var_token(name: str) -> str:
    return VAR_TOKEN_RE.sub("", name).upper()


def discover_variable_maps(experiment_dirs: dict[str, Path]) -> dict[str, dict[str, str]]:
    maps: dict[str, dict[str, str]] = {}
    for exp, exp_dir in experiment_dirs.items():
        token_map: dict[str, str] = {}
        for p in sorted(exp_dir.iterdir()):
            if not p.is_dir() or p.name.startswith(".") or not has_pf_subdirs(p):
                continue
            token = normalize_var_token(p.name)
            if token and token not in token_map:
                token_map[token] = p.name
        maps[exp] = token_map
    return maps


def resolve_var_name(
    variable_maps: dict[str, dict[str, str]],
    experiment: str,
    candidates: Sequence[str],
) -> str | None:
    token_map = variable_maps[experiment]
    for cand in candidates:
        token = normalize_var_token(cand)
        if token in token_map:
            return token_map[token]
    return None


def collect_file_records(
    variable_dir: Path,
    max_days: int | None,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
) -> list[tuple[int, Path]]:
    if not variable_dir.exists():
        raise FileNotFoundError(f"Missing directory: {variable_dir}")
    allowed_set = set(allowed_months) if allowed_months is not None else None
    day_dirs = sorted(
        p for p in variable_dir.iterdir() if p.is_dir() and p.name.startswith("pf")
    )
    if max_days is not None:
        day_dirs = day_dirs[:max_days]

    records: list[tuple[int, Path]] = []
    for day_dir in day_dirs:
        month = parse_month_from_day_name(day_dir.name)
        if allowed_set is not None:
            if month is None or month not in allowed_set:
                continue

        for file_path in sorted(day_dir.glob("*.nc")):
            utc_hour = parse_utc_hour_from_name(file_path.name)
            if utc_hour is None:
                continue
            local_hour = (utc_hour + utc_offset_hours) % 24
            records.append((local_hour, file_path))
    return records


def resolve_data_var_name(ds: xr.Dataset, requested: str) -> str:
    if requested in ds.data_vars:
        return requested

    req_lower = requested.lower()
    for name in ds.data_vars:
        if name.lower() == req_lower:
            return name

    req_token = normalize_var_token(requested)
    token_hits = [name for name in ds.data_vars if normalize_var_token(name) == req_token]
    if len(token_hits) == 1:
        return token_hits[0]

    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars.keys()))

    raise KeyError(f"Variable '{requested}' not found. Available: {list(ds.data_vars.keys())}")


def apply_spatial_window_to_array(
    arr: np.ndarray, spatial_window: SpatialWindow, file_path: Path
) -> np.ndarray:
    if arr.ndim < 2:
        return arr
    y_slice = slice(spatial_window.y_start, spatial_window.y_end)
    x_slice = slice(spatial_window.x_start, spatial_window.x_end)
    trimmed = arr[(slice(None),) * (arr.ndim - 2) + (y_slice, x_slice)]
    if trimmed.shape[-2] == 0 or trimmed.shape[-1] == 0:
        raise ValueError(
            f"Spatial slice produced empty Y/X domain for {file_path}: "
            f"shape {arr.shape} -> {trimmed.shape}"
        )
    return trimmed


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
) -> np.ndarray:
    with xr.open_dataset(file_path, decode_times=False) as ds:
        var_name = resolve_data_var_name(ds, requested_variable)
        da = ds[var_name]
        arr = np.asarray(da.values, dtype=np.float64)
        arr = apply_spatial_window_to_array(arr, spatial_window, file_path)
        arr = to_time_level_yx(arr, da.dims, file_path, requested_variable)
    return arr


def read_vertical_profile(
    file_path: Path,
    requested_variable: str,
    spatial_window: SpatialWindow,
) -> np.ndarray:
    arr = read_time_level_yx(file_path, requested_variable, spatial_window=spatial_window)
    profile, _ = nanmean_with_count(arr, axis=(0, 2, 3))
    return np.asarray(profile, dtype=np.float64)


def align_tlyx_shapes(arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    t_min = min(arr.shape[0] for arr in arrays)
    l_min = min(arr.shape[1] for arr in arrays)
    y_min = min(arr.shape[2] for arr in arrays)
    x_min = min(arr.shape[3] for arr in arrays)
    return [arr[:t_min, :l_min, :y_min, :x_min] for arr in arrays]


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


def mean_profile(field_tlyx: np.ndarray) -> np.ndarray:
    mean, _ = nanmean_with_count(field_tlyx, axis=(0, 2, 3))
    return np.asarray(mean, dtype=np.float64)


def profile_hour_accumulate(
    records: list[tuple[int, Path]],
    profile_reader,
    progress_tag: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """
    profile_reader(file_path) -> dict[str, np.ndarray(level,)]
    Returns sums/counts dicts keyed by diagnostic, and number of used files.
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
    sums: dict[str, np.ndarray], counts: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
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
    line_reader,
    progress_tag: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """
    line_reader(file_path) -> dict[str, float]
    Returns sums/counts arrays with shape (24,) keyed by diagnostic.
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
    sums: dict[str, np.ndarray], counts: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for diag, sum_arr in sums.items():
        cnt_arr = counts[diag]
        mean = np.full(sum_arr.shape, np.nan, dtype=np.float64)
        valid = cnt_arr > 0
        mean[valid] = sum_arr[valid] / cnt_arr[valid]
        out[diag] = mean
    return out


def build_cache_file(
    intermediate_dir: Path,
    analysis_name: str,
    period_subdir: Path,
    experiment: str,
    spatial_tag: str,
) -> Path:
    return (
        intermediate_dir
        / safe_name(analysis_name)
        / period_subdir
        / f"{experiment}_{spatial_tag}.npz"
    )


def save_cache(cache_file: Path, payload: dict[str, np.ndarray]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, **payload)


def load_cache(cache_file: Path) -> dict[str, np.ndarray]:
    with np.load(cache_file) as data:
        return {k: np.asarray(data[k], dtype=np.float64) for k in data.files}


def infer_freezing_threshold(temperature_profile: np.ndarray) -> float | None:
    valid = temperature_profile[np.isfinite(temperature_profile)]
    if valid.size == 0:
        return None
    return 273.15 if float(np.median(valid)) > 150.0 else 0.0


def compute_freezing_line_km(axis: VerticalAxis, temperature_profiles: list[np.ndarray]) -> np.ndarray | None:
    if not axis.is_height_km or not temperature_profiles:
        return None
    n_levels = min(axis.values.size, *(arr.shape[0] for arr in temperature_profiles))
    if n_levels < 2:
        return None

    stacked = np.stack([arr[:n_levels, :] for arr in temperature_profiles], axis=0)
    mean_temp = np.nanmean(stacked, axis=0)
    threshold = infer_freezing_threshold(mean_temp)
    if threshold is None:
        return None

    y_km = np.asarray(axis.values[:n_levels], dtype=np.float64)
    order = np.argsort(y_km)
    y_sorted = y_km[order]
    t_sorted = mean_temp[order, :]

    freeze_line = np.full((24,), np.nan, dtype=np.float64)
    for hour in range(24):
        column = t_sorted[:, hour]
        finite = np.isfinite(column) & np.isfinite(y_sorted)
        if np.sum(finite) < 2:
            continue
        yy = y_sorted[finite]
        tt = column[finite]
        for i in range(yy.size - 1):
            t1 = tt[i]
            t2 = tt[i + 1]
            y1 = yy[i]
            y2 = yy[i + 1]
            if np.isclose(t1, threshold):
                freeze_line[hour] = y1
                break
            if np.isclose(t2, threshold):
                freeze_line[hour] = y2
                break
            d1 = t1 - threshold
            d2 = t2 - threshold
            if d1 * d2 < 0 and not np.isclose(t1, t2):
                frac = (threshold - t1) / (t2 - t1)
                freeze_line[hour] = y1 + frac * (y2 - y1)
                break
    return freeze_line


def compute_geopotential_height_profile(
    geopotential_dir: Path,
    height_variable: str,
    max_days: int | None,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
    aggregate: str,
    spatial_window: SpatialWindow,
) -> tuple[np.ndarray, int]:
    records = collect_file_records(
        variable_dir=geopotential_dir,
        max_days=max_days,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
    )
    if not records:
        raise RuntimeError(f"No valid geopotential files found in {geopotential_dir}")

    if aggregate == "first":
        profile = read_vertical_profile(records[0][1], height_variable, spatial_window)
        return profile, 1

    first = read_vertical_profile(records[0][1], height_variable, spatial_window)
    sums = np.zeros_like(first, dtype=np.float64)
    counts = np.zeros_like(first, dtype=np.int64)
    for idx, (_, file_path) in enumerate(records, start=1):
        profile = read_vertical_profile(file_path, height_variable, spatial_window)
        n = min(sums.size, profile.size)
        valid = np.isfinite(profile[:n])
        sums[:n][valid] += profile[:n][valid]
        counts[:n][valid] += 1
        if idx % 4000 == 0 or idx == len(records):
            print(
                f"[{geopotential_dir.parent.name}/GEOPOTENTIEL] {idx}/{len(records)} files",
                flush=True,
            )

    mean = np.full_like(sums, np.nan)
    ok = counts > 0
    mean[ok] = sums[ok] / counts[ok]
    return mean, len(records)


def align_axis_and_profile(
    axis: VerticalAxis,
    profile: np.ndarray,
) -> tuple[VerticalAxis, np.ndarray]:
    n = min(axis.values.size, profile.shape[0])
    axis_new = VerticalAxis(values=axis.values[:n], label=axis.label, is_height_km=axis.is_height_km)
    return axis_new, profile[:n, :]


def interpolate_profile_to_target_height(
    source_height_km: np.ndarray,
    source_profile: np.ndarray,
    target_height_km: np.ndarray,
) -> np.ndarray:
    source_height_km = np.asarray(source_height_km, dtype=np.float64)
    target_height_km = np.asarray(target_height_km, dtype=np.float64)
    out = np.full((target_height_km.size, source_profile.shape[1]), np.nan, dtype=np.float64)
    for hour in range(source_profile.shape[1]):
        col = source_profile[:, hour]
        finite = np.isfinite(source_height_km) & np.isfinite(col)
        if np.sum(finite) < 2:
            continue
        z = source_height_km[finite]
        v = col[finite]
        order = np.argsort(z)
        z = z[order]
        v = v[order]
        unique_mask = np.concatenate(([True], np.diff(z) > 0.0))
        z = z[unique_mask]
        v = v[unique_mask]
        if z.size < 2:
            continue
        out[:, hour] = np.interp(target_height_km, z, v, left=np.nan, right=np.nan)
    return out


def maybe_convert_pressure_to_pa(p: np.ndarray) -> np.ndarray:
    finite = p[np.isfinite(p)]
    if finite.size == 0:
        return p
    p01 = float(np.nanpercentile(finite, 1))
    p99 = float(np.nanpercentile(finite, 99))
    if 100.0 <= p01 <= 1200.0 and 100.0 <= p99 <= 2000.0:
        return p * 100.0
    return p


def compute_theta_e_field(
    temperature_k: np.ndarray,
    specific_humidity: np.ndarray,
    pressure_pa: np.ndarray,
) -> np.ndarray:
    t = np.asarray(temperature_k, dtype=np.float64)
    q = np.asarray(specific_humidity, dtype=np.float64)
    p = np.asarray(pressure_pa, dtype=np.float64)

    q = np.where(q < 0.0, np.nan, q)
    p = maybe_convert_pressure_to_pa(p)
    p = np.where(p <= 0.0, np.nan, p)

    e = q * p / (EPS + (1.0 - EPS) * q)
    e = np.where(e > 1.0, e, np.nan)

    ln_e = np.log(e / 611.2)
    td_c = (243.5 * ln_e) / (17.67 - ln_e)
    td_k = td_c + 273.15

    td_k = np.where(np.isfinite(td_k), td_k, np.nan)
    td_k = np.clip(td_k, 180.0, 350.0)
    t = np.clip(t, 180.0, 350.0)

    tl = 1.0 / ((1.0 / (td_k - 56.0)) + (np.log(t / td_k) / 800.0)) + 56.0
    kappa = 0.2854 * (1.0 - 0.28 * q)
    expo = q * (1.0 + 0.81 * q) * ((3376.0 / tl) - 2.54)
    theta_e = t * np.power(P0 / p, kappa) * np.exp(expo)
    return theta_e


def compute_dp_pa(pressure_tlyx: np.ndarray) -> np.ndarray:
    p = np.asarray(pressure_tlyx, dtype=np.float64)
    if p.shape[1] == 1:
        return np.abs(p)
    p_half = np.empty((p.shape[0], p.shape[1] + 1, p.shape[2], p.shape[3]), dtype=np.float64)
    p_half[:, 1:-1, :, :] = 0.5 * (p[:, :-1, :, :] + p[:, 1:, :, :])
    p_half[:, 0, :, :] = p[:, 0, :, :] + (p[:, 0, :, :] - p_half[:, 1, :, :])
    p_half[:, -1, :, :] = p[:, -1, :, :] - (p_half[:, -2, :, :] - p[:, -1, :, :])
    dp = np.abs(p_half[:, :-1, :, :] - p_half[:, 1:, :, :])
    return dp


def choose_bottom_level_index(pressure_tlyx: np.ndarray) -> int:
    mean_p, _ = nanmean_with_count(pressure_tlyx, axis=(0, 2, 3))
    if mean_p.size == 0 or not np.isfinite(mean_p).any():
        return pressure_tlyx.shape[1] - 1
    return int(np.nanargmax(mean_p))


def get_peer_file(base_file: Path, experiment_dir: Path, var_name: str) -> Path:
    return experiment_dir / var_name / base_file.parent.name / base_file.name


def compute_downdraft_profiles(
    experiment: str,
    experiment_dir: Path,
    records: list[tuple[int, Path]],
    names: dict[str, str],
    spatial_window: SpatialWindow,
) -> dict[str, np.ndarray]:
    def reader(base_file: Path):
        req = (
            get_peer_file(base_file, experiment_dir, names["UD_OMEGA"]),
            get_peer_file(base_file, experiment_dir, names["UD_MESH_FRAC"]),
            get_peer_file(base_file, experiment_dir, names["DD_OMEGA"]),
            get_peer_file(base_file, experiment_dir, names["DD_MESH_FRAC"]),
        )
        if not all(p.exists() for p in req):
            return None
        ud_omega, ud_mesh, dd_omega, dd_mesh = align_tlyx_shapes(
            [
                read_time_level_yx(req[0], names["UD_OMEGA"], spatial_window),
                read_time_level_yx(req[1], names["UD_MESH_FRAC"], spatial_window),
                read_time_level_yx(req[2], names["DD_OMEGA"], spatial_window),
                read_time_level_yx(req[3], names["DD_MESH_FRAC"], spatial_window),
            ]
        )

        # Mass flux: M_d = sigma_d * omega_d / g
        # DD omega is positive (downward in pressure coords), so flux is positive.
        # We only care where mesh > 0 (active downdrafts).
        dd_flux = np.where(dd_mesh > 0, (dd_omega * dd_mesh) / 9.80665, 0.0)

        # UD mass flux: M_u = - sigma_u * omega_u / g
        # UD omega is negative (upward), so -omega gives positive flux.
        ud_flux = np.where(ud_mesh > 0, (-ud_omega * ud_mesh) / 9.80665, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = dd_flux / ud_flux
        ratio = np.where(np.abs(ud_flux) < 1e-10, np.nan, ratio)

        # DD intensity: |omega| only where downdraft is active
        dd_omega_abs = np.where(dd_mesh > 0, np.abs(dd_omega), np.nan)

        return {
            "DD_EXTENT": mean_profile(dd_mesh),
            "DD_FLUX": mean_profile(dd_flux),
            "DD_INTENSITY": mean_profile(dd_omega_abs),
            "DD_TO_UD_FLUX_RATIO": mean_profile(ratio),
            "NET_CONVECTIVE_FLUX": mean_profile(ud_flux - dd_flux),
        }

    sums, counts, used = profile_hour_accumulate(
        records=records,
        profile_reader=reader,
        progress_tag=f"{experiment}/downdraft",
    )
    print(f"[{experiment}/downdraft] used files: {used}/{len(records)}", flush=True)
    return finalize_profile_means(sums, counts)


def compute_precip_profiles(
    experiment: str,
    experiment_dir: Path,
    records: list[tuple[int, Path]],
    names: dict[str, str],
    spatial_window: SpatialWindow,
) -> dict[str, np.ndarray]:
    def reader(base_file: Path):
        cv_file = get_peer_file(base_file, experiment_dir, names["CV_PREC_FLUX"])
        st_file = get_peer_file(base_file, experiment_dir, names["ST_PREC_FLUX"])
        if not cv_file.exists() or not st_file.exists():
            return None
        cv, st = align_tlyx_shapes(
            [
                read_time_level_yx(cv_file, names["CV_PREC_FLUX"], spatial_window),
                read_time_level_yx(st_file, names["ST_PREC_FLUX"], spatial_window),
            ]
        )
        total = cv + st
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = cv / total
        frac = np.where(np.abs(total) < 1e-12, np.nan, frac)
        return {
            "CV_PREC_FLUX": mean_profile(cv),
            "ST_PREC_FLUX": mean_profile(st),
            "TOTAL_PREC_FLUX": mean_profile(total),
            "CONVECTIVE_PRECIP_FRACTION": mean_profile(frac),
        }

    sums, counts, used = profile_hour_accumulate(
        records=records,
        profile_reader=reader,
        progress_tag=f"{experiment}/precip",
    )
    print(f"[{experiment}/precip] used files: {used}/{len(records)}", flush=True)
    return finalize_profile_means(sums, counts)


def compute_thermo_profiles(
    experiment: str,
    experiment_dir: Path,
    records: list[tuple[int, Path]],
    names: dict[str, str],
    z_m: np.ndarray,
    spatial_window: SpatialWindow,
) -> dict[str, np.ndarray]:
    def reader(base_file: Path):
        t_file = get_peer_file(base_file, experiment_dir, names["TEMPERATURE"])
        q_file = get_peer_file(base_file, experiment_dir, names["HUMI.SPECIFI"])
        p_file = get_peer_file(base_file, experiment_dir, names["PRESSURE"])
        if not t_file.exists() or not q_file.exists() or not p_file.exists():
            return None
        t, q, p = align_tlyx_shapes(
            [
                read_time_level_yx(t_file, names["TEMPERATURE"], spatial_window),
                read_time_level_yx(q_file, names["HUMI.SPECIFI"], spatial_window),
                read_time_level_yx(p_file, names["PRESSURE"], spatial_window),
            ]
        )
        nlev = min(t.shape[1], z_m.size)
        t = t[:, :nlev, :, :]
        q = q[:, :nlev, :, :]
        p = p[:, :nlev, :, :]
        z = np.asarray(z_m[:nlev], dtype=np.float64)[None, :, None, None]

        theta_e = compute_theta_e_field(t, q, p)
        mse = CP_D * t + LV * q + G * z
        return {
            "THETA_E": mean_profile(theta_e),
            "MSE": mean_profile(mse),
        }

    sums, counts, used = profile_hour_accumulate(
        records=records,
        profile_reader=reader,
        progress_tag=f"{experiment}/thermo",
    )
    print(f"[{experiment}/thermo] used files: {used}/{len(records)}", flush=True)
    return finalize_profile_means(sums, counts)


def compute_kt273_lines(
    experiment: str,
    experiment_dir: Path,
    records: list[tuple[int, Path]],
    names: dict[str, str],
    spatial_window: SpatialWindow,
) -> dict[str, np.ndarray]:
    def safe_scalar_mean(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan")
        return float(np.mean(finite))

    def mean_scalar(arr: np.ndarray) -> float:
        if arr.ndim >= 2:
            arr = apply_spatial_window_to_array(arr, spatial_window, Path("<in-memory>"))
        return safe_scalar_mean(arr)

    def reader(base_file: Path):
        required = (
            "KT273GRAUPEL",
            "KT273DD_OMEGA",
            "KT273DD_MESH_FRA",
            "KT273RAIN",
            "KT273HUMI.SPECIF",
        )
        files = {k: get_peer_file(base_file, experiment_dir, names[k]) for k in required}
        if not all(fp.exists() for fp in files.values()):
            return None

        vals: dict[str, float] = {}
        for key in required:
            with xr.open_dataset(files[key], decode_times=False) as ds:
                vn = resolve_data_var_name(ds, names[key])
                vals[key] = mean_scalar(np.asarray(ds[vn].values, dtype=np.float64))

        return {
            "KT273GRAUPEL": vals["KT273GRAUPEL"],
            "KT273DD_FLUX": vals["KT273DD_OMEGA"] * vals["KT273DD_MESH_FRA"],
            "KT273RAIN": vals["KT273RAIN"],
            "KT273HUMI_SPECIF": vals["KT273HUMI.SPECIF"],
        }

    sums, counts, used = line_hour_accumulate(
        records=records,
        line_reader=reader,
        progress_tag=f"{experiment}/kt273",
    )
    print(f"[{experiment}/kt273] used files: {used}/{len(records)}", flush=True)
    return finalize_line_means(sums, counts)


def compute_column_lines(
    experiment: str,
    experiment_dir: Path,
    records: list[tuple[int, Path]],
    names: dict[str, str],
    available_condensate_names: list[str],
    spatial_window: SpatialWindow,
) -> dict[str, np.ndarray]:
    def safe_scalar_mean(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan")
        return float(np.mean(finite))

    def reader(base_file: Path):
        p_file = get_peer_file(base_file, experiment_dir, names["PRESSURE"])
        cv_file = get_peer_file(base_file, experiment_dir, names["CV_PREC_FLUX"])
        st_file = get_peer_file(base_file, experiment_dir, names["ST_PREC_FLUX"])
        if not p_file.exists() or not cv_file.exists() or not st_file.exists():
            return None

        p = read_time_level_yx(p_file, names["PRESSURE"], spatial_window)
        p = maybe_convert_pressure_to_pa(p)
        cv = read_time_level_yx(cv_file, names["CV_PREC_FLUX"], spatial_window)
        st = read_time_level_yx(st_file, names["ST_PREC_FLUX"], spatial_window)
        p, cv, st = align_tlyx_shapes([p, cv, st])
        total_prec = cv + st

        q_fields: list[np.ndarray] = []
        for q_name in available_condensate_names:
            q_file = get_peer_file(base_file, experiment_dir, q_name)
            if not q_file.exists():
                continue
            q_arr = read_time_level_yx(q_file, q_name, spatial_window)
            q_fields.append(q_arr)
        if not q_fields:
            return None

        aligned = align_tlyx_shapes([p, total_prec, *q_fields])
        p = aligned[0]
        total_prec = aligned[1]
        q_fields = aligned[2:]

        q_sum = np.zeros_like(p, dtype=np.float64)
        for q_arr in q_fields:
            q_sum += q_arr

        dp = compute_dp_pa(p)
        column_cond = np.nansum(q_sum * dp / G, axis=1)  # (time, y, x)
        col_mean = safe_scalar_mean(column_cond)

        bottom_idx = choose_bottom_level_index(p)
        surface_flux = total_prec[:, bottom_idx, :, :]
        surface_mean = safe_scalar_mean(surface_flux)

        with np.errstate(divide="ignore", invalid="ignore"):
            residence = col_mean / surface_mean
        if not np.isfinite(residence):
            residence = np.nan

        return {
            "COLUMN_CONDENSATE": col_mean,
            "SURFACE_PRECIP_FLUX": surface_mean,
            "RESIDENCE_TIME": float(residence),
        }

    sums, counts, used = line_hour_accumulate(
        records=records,
        line_reader=reader,
        progress_tag=f"{experiment}/column",
    )
    print(f"[{experiment}/column] used files: {used}/{len(records)}", flush=True)
    return finalize_line_means(sums, counts)


def infer_abs_limits(control: np.ndarray, linear: bool = True) -> tuple[float, float]:
    valid = control[np.isfinite(control)]
    if valid.size == 0:
        return (0.0, 1.0) if linear else (1e-12, 1.0)
    if linear:
        vmin = float(np.percentile(valid, 2))
        vmax = float(np.percentile(valid, 98))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        return vmin, vmax
    positive = valid[valid > 0]
    if positive.size == 0:
        return 1e-12, 1.0
    vmin = float(np.percentile(positive, 2))
    vmax = float(np.percentile(positive, 98))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return vmin, vmax


def infer_anom_scale(*anomalies: np.ndarray) -> float:
    vals = []
    for arr in anomalies:
        finite = np.abs(arr[np.isfinite(arr)])
        if finite.size > 0:
            vals.append(finite)
    if not vals:
        return 1.0
    merged = np.concatenate(vals)
    scale = float(np.percentile(merged, 98))
    if scale <= 0:
        scale = float(np.max(merged))
    if scale <= 0:
        scale = 1.0
    return scale


def plot_three_panel_anomaly(
    title_label: str,
    unit: str,
    control: np.ndarray,
    graupel: np.ndarray,
    twomom: np.ndarray,
    axis: VerticalAxis,
    output_file: Path,
    period_label: str,
    max_height_km: float,
    abs_linear: bool = True,
    use_abs_control: bool = True,
    freezing_lines: dict[str, np.ndarray] | None = None,
) -> None:
    y = np.asarray(axis.values, dtype=np.float64)
    order = np.argsort(y)
    y = y[order]
    c = control[order, :]
    g = graupel[order, :]
    t = twomom[order, :]

    d1 = g - c
    d2 = t - g

    if axis.is_height_km:
        keep = np.isfinite(y) & (y >= 0.0) & (y <= max_height_km)
        y = y[keep]
        c = c[keep, :]
        g = g[keep, :]
        t = t[keep, :]
        d1 = d1[keep, :]
        d2 = d2[keep, :]

    abs_field = c if use_abs_control else g
    vmin_abs, vmax_abs = infer_abs_limits(abs_field, linear=abs_linear)
    anom_scale = infer_anom_scale(d1, d2)

    if abs_linear:
        abs_norm = mcolors.Normalize(vmin=vmin_abs, vmax=vmax_abs)
    else:
        abs_norm = mcolors.LogNorm(vmin=max(vmin_abs, 1e-12), vmax=max(vmax_abs, vmin_abs * 10.0))
    diff_norm = mcolors.TwoSlopeNorm(vmin=-anom_scale, vcenter=0.0, vmax=anom_scale)

    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    y_edges = centers_to_edges(y)
    c_plot = np.ma.masked_invalid(abs_field)
    d1_plot = np.ma.masked_invalid(d1)
    d2_plot = np.ma.masked_invalid(d2)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    for idx, ax in enumerate(axes):
        ax.set_facecolor("#d3d3d3")
        ax.set_xlabel("Hour (Amazon UTC-4)", fontsize=14)
        ax.set_xticks(np.arange(0, 24, 6))
        ax.set_xlim(-0.5, 23.5)
        ax.set_ylabel(axis.label, fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        if axis.is_height_km:
            ax.set_ylim(0.0, max_height_km)
        ax.text(
            0.02,
            0.98,
            f"({chr(ord('a') + idx)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2.0},
        )
        if freezing_lines is not None:
            exp_key = EXPERIMENTS[idx] if idx < len(EXPERIMENTS) else None
            if exp_key and exp_key in freezing_lines:
                fl = freezing_lines[exp_key]
                if fl is not None and np.isfinite(fl).any():
                    hours = np.arange(24, dtype=np.float64)
                    axes[idx].plot(
                        hours,
                        fl,
                        color="black",
                        linewidth=1.8,
                        linestyle="--",
                        label=f"Freezing level ({EXPERIMENT_LABELS[exp_key]})",
                        zorder=10,
                    )
                    if idx == 0:
                        axes[idx].legend(loc="upper right", fontsize=12, framealpha=0.9)


    pcm_abs = axes[0].pcolormesh(
        hour_edges, y_edges, c_plot, cmap=cmaps.WhiteBlueGreenYellowRed, norm=abs_norm, shading="auto"
    )
    axes[0].set_title(f"{EXPERIMENT_LABELS['control']} ({title_label}, Absolute)", fontsize=12, fontweight="bold")

    axes[1].pcolormesh(hour_edges, y_edges, d1_plot, cmap="RdBu_r", norm=diff_norm, shading="auto")
    axes[1].set_title(f"{EXPERIMENT_LABELS['graupel']} - {EXPERIMENT_LABELS['control']}", fontsize=12, fontweight="bold")

    pcm_diff = axes[2].pcolormesh(hour_edges, y_edges, d2_plot, cmap="RdBu_r", norm=diff_norm, shading="auto")
    axes[2].set_title(f"{EXPERIMENT_LABELS['2mom']} - {EXPERIMENT_LABELS['graupel']}", fontsize=12, fontweight="bold")

    fig.suptitle(f"{period_label} - {title_label}", fontsize=15, fontweight="bold")
    unit_tag = f" [{unit}]" if unit else ""

    cbar_abs = fig.colorbar(pcm_abs, ax=axes[0], orientation="horizontal", fraction=0.08, pad=0.16)
    cbar_abs.set_label(f"Mean {title_label}{unit_tag}", fontsize=12)
    cbar_abs.ax.tick_params(labelsize=11)

    cbar_diff = fig.colorbar(pcm_diff, ax=axes[1:], orientation="horizontal", fraction=0.08, pad=0.16)
    cbar_diff.set_label(f"{title_label} anomaly{unit_tag}", fontsize=12)
    cbar_diff.ax.tick_params(labelsize=11)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def plot_three_experiment_absolute(
    title_label: str,
    unit: str,
    profiles: dict[str, np.ndarray],
    axis: VerticalAxis,
    output_file: Path,
    period_label: str,
    max_height_km: float,
    diverging: bool = False,
    freezing_lines: dict[str, np.ndarray] | None = None,
) -> None:
    y = np.asarray(axis.values, dtype=np.float64)
    order = np.argsort(y)
    y = y[order]
    p = {exp: profiles[exp][order, :] for exp in EXPERIMENTS}
    if axis.is_height_km:
        keep = np.isfinite(y) & (y >= 0.0) & (y <= max_height_km)
        y = y[keep]
        p = {exp: arr[keep, :] for exp, arr in p.items()}

    chunks = [arr[np.isfinite(arr)] for arr in p.values() if np.isfinite(arr).any()]
    if not chunks:
        vmin, vmax = -1.0, 1.0
    else:
        all_vals = np.concatenate(chunks)
        if diverging:
            scale = float(np.percentile(np.abs(all_vals), 98))
            if scale <= 0:
                scale = 1.0
            vmin, vmax = -scale, scale
        else:
            vmin = float(np.percentile(all_vals, 2))
            vmax = float(np.percentile(all_vals, 98))
            if vmax <= vmin:
                vmax = vmin + 1e-6

    if diverging:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = "RdBu_r"
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cmaps.WhiteBlueGreenYellowRed

    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    y_edges = centers_to_edges(y)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    pcm = None
    for idx, exp in enumerate(EXPERIMENTS):
        arr = np.ma.masked_invalid(p[exp])
        pcm = axes[idx].pcolormesh(hour_edges, y_edges, arr, cmap=cmap, norm=norm, shading="auto")
        axes[idx].set_title(EXPERIMENT_LABELS[exp], fontsize=12, fontweight="bold")
        axes[idx].set_facecolor("#d3d3d3")
        axes[idx].set_xlabel("Hour (Amazon UTC-4)", fontsize=14)
        axes[idx].set_xticks(np.arange(0, 24, 6))
        axes[idx].set_xlim(-0.5, 23.5)
        axes[idx].set_ylabel(axis.label, fontsize=14)
        axes[idx].tick_params(axis="both", labelsize=12)
        if axis.is_height_km:
            axes[idx].set_ylim(0.0, max_height_km)
            
        if freezing_lines is not None and exp in freezing_lines:
            fl = freezing_lines[exp]
            if fl is not None and np.isfinite(fl).any():
                hours = np.arange(24, dtype=np.float64)
                axes[idx].plot(
                    hours,
                    fl,
                    color="black",
                    linewidth=1.8,
                    linestyle="--",
                    label=f"Freezing level ({EXPERIMENT_LABELS[exp]})",
                    zorder=10,
                )
                if idx == 0:
                    axes[idx].legend(loc="upper right", fontsize=12, framealpha=0.9)
                    
        axes[idx].text(
            0.02,
            0.98,
            f"({chr(ord('a') + idx)})",
            transform=axes[idx].transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2.0},
        )

    fig.suptitle(f"{period_label} - {title_label}", fontsize=15, fontweight="bold")
    unit_tag = f" [{unit}]" if unit else ""
    cbar = fig.colorbar(pcm, ax=axes, orientation="horizontal", fraction=0.08, pad=0.16)
    cbar.set_label(f"{title_label}{unit_tag}", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def plot_lines_stacked(
    title: str,
    period_label: str,
    diag_order: Sequence[str],
    diag_labels: dict[str, str],
    unit_map: dict[str, str],
    line_data: dict[str, dict[str, np.ndarray]],
    output_file: Path,
) -> None:
    n = len(diag_order)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.8 * n), sharex=True, constrained_layout=True)
    if n == 1:
        axes = [axes]
    hours = np.arange(24, dtype=np.float64)

    for idx, diag in enumerate(diag_order):
        ax = axes[idx]
        for exp in EXPERIMENTS:
            arr = line_data.get(diag, {}).get(exp)
            if arr is None:
                continue
            ax.plot(hours, arr, linewidth=2.0, label=EXPERIMENT_LABELS[exp])
        ylabel = diag_labels.get(diag, diag)
        unit = unit_map.get(diag, "")
        if unit:
            ylabel = f"{ylabel} [{unit}]"
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(alpha=0.25, linestyle="--")
        ax.tick_params(axis="both", labelsize=10)
        ax.text(
            0.01,
            0.92,
            f"({chr(ord('a') + idx)})",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
        )

    axes[-1].set_xlabel("Hour (Amazon UTC-4)", fontsize=12)
    axes[-1].set_xticks(np.arange(0, 24, 3))
    axes[0].legend(loc="upper right", fontsize=10, framealpha=0.9, ncol=3)
    fig.suptitle(f"{period_label} - {title}", fontsize=14, fontweight="bold")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def compute_vertical_derivative(
    profile: np.ndarray, height_km: np.ndarray
) -> np.ndarray:
    z = np.asarray(height_km, dtype=np.float64) * 1000.0  # m
    out = np.full(profile.shape, np.nan, dtype=np.float64)
    for hour in range(profile.shape[1]):
        col = profile[:, hour]
        finite = np.isfinite(z) & np.isfinite(col)
        if np.sum(finite) < 3:
            continue
        idx = np.where(finite)[0]
        zf = z[idx]
        vf = col[idx]
        order = np.argsort(zf)
        idx = idx[order]
        zf = zf[order]
        vf = vf[order]
        if np.any(np.diff(zf) <= 0):
            unique = np.concatenate(([True], np.diff(zf) > 0.0))
            idx = idx[unique]
            zf = zf[unique]
            vf = vf[unique]
        if zf.size < 3:
            continue
        grad = np.gradient(vf, zf)
        full = np.full(col.shape, np.nan, dtype=np.float64)
        full[idx] = grad
        out[:, hour] = full
    return out


def compute_bl_gradient_line(
    theta_e_profile: np.ndarray,
    height_km: np.ndarray,
    zmax_km: float = 2.0,
) -> np.ndarray:
    z = np.asarray(height_km, dtype=np.float64)
    line = np.full((theta_e_profile.shape[1],), np.nan, dtype=np.float64)
    for hour in range(theta_e_profile.shape[1]):
        v = theta_e_profile[:, hour]
        finite = np.isfinite(z) & np.isfinite(v) & (z >= 0.0) & (z <= zmax_km)
        if np.sum(finite) < 2:
            continue
        zz = z[finite]
        vv = v[finite]
        order = np.argsort(zz)
        zz = zz[order]
        vv = vv[order]
        coeff = np.polyfit(zz, vv, deg=1)
        line[hour] = coeff[0]  # K/km
    return line


def main() -> None:
    args = parse_args()
    analyses = list(dict.fromkeys(args.analyses))
    spatial_window = build_spatial_window(args.y_slice, args.x_slice)
    spatial_tag = spatial_window_tag(spatial_window)

    experiment_dirs = {
        "control": args.control_dir.resolve(),
        "graupel": args.graupel_dir.resolve(),
        "2mom": args.twomom_dir.resolve(),
    }
    geopotential_dirs = {
        "control": args.control_geopotential_dir.resolve(),
        "graupel": args.graupel_geopotential_dir.resolve(),
        "2mom": args.twomom_geopotential_dir.resolve(),
    }
    for exp, d in experiment_dirs.items():
        if not d.exists():
            raise FileNotFoundError(f"{exp} data dir not found: {d}")
    for exp, d in geopotential_dirs.items():
        if not d.exists():
            print(f"[warn] {exp} geopotential dir not found: {d}", flush=True)

    variable_maps = discover_variable_maps(experiment_dirs)

    if args.list_variables:
        for exp in EXPERIMENTS:
            names = sorted(variable_maps[exp].values())
            print(f"\n{exp} ({len(names)} vars):")
            print(", ".join(names) if names else "(none)")
        return

    selected_seasons = resolve_seasons(args.seasons)
    periods = build_period_specs(set(args.analysis_modes), selected_seasons)
    if not periods:
        raise RuntimeError("No analysis periods selected.")

    output_dir = args.output_dir.resolve()
    intermediate_dir = args.intermediate_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    print("\nInput data directories:", flush=True)
    for exp in EXPERIMENTS:
        print(f"- {exp}: {experiment_dirs[exp]}", flush=True)
    print("\nGeopotential directories:", flush=True)
    for exp in EXPERIMENTS:
        print(f"- {exp}: {geopotential_dirs[exp]}", flush=True)
    print("\nOutput directory:", output_dir, flush=True)
    print("Intermediate directory:", intermediate_dir, flush=True)
    print("Periods:", [p.key for p in periods], flush=True)
    print("Analyses:", analyses, flush=True)
    print("Spatial averaging tag:", spatial_tag, flush=True)
    print(
        "Spatial Y slice:",
        f"{spatial_window.y_start}:{spatial_window.y_end}",
        "| X slice:",
        f"{spatial_window.x_start}:{spatial_window.x_end}",
        flush=True,
    )
    print("Ignoring +0024 files by design.", flush=True)

    # Resolve variable names per experiment.
    required_candidates: dict[str, Sequence[str]] = {
        "UD_OMEGA": ("UD_OMEGA", "UD.OMEGA", "UDOMEGA"),
        "UD_MESH_FRAC": ("UD_MESH_FRAC", "UD_MESH_FRA", "UD.MESH.FRAC", "UD.MESH.FRA"),
        "DD_OMEGA": ("DD_OMEGA", "DD.OMEGA", "DDOMEGA"),
        "DD_MESH_FRAC": ("DD_MESH_FRAC", "DD_MESH_FRA", "DD.MESH.FRAC", "DD.MESH.FRA"),
        "CV_PREC_FLUX": ("CV_PREC_FLUX", "CV.PREC.FLUX", "CV_PREC_FL"),
        "ST_PREC_FLUX": ("ST_PREC_FLUX", "ST.PREC.FLUX", "ST_PREC_FL"),
        "TEMPERATURE": ("TEMPERATURE", "TEMPERATUR"),
        "HUMI.SPECIFI": ("HUMI.SPECIFI", "HUMI_SPECIFI"),
        "PRESSURE": ("PRESSURE",),
        "LIQUID_WATER": ("LIQUID_WATER", "LIQUID_WAT"),
        "RAIN": ("RAIN",),
        "SNOW": ("SNOW",),
        "SOLID_WATER": ("SOLID_WATER", "SOLID_WATE"),
        "GRAUPEL": ("GRAUPEL",),
        "KT273GRAUPEL": ("KT273GRAUPEL",),
        "KT273DD_OMEGA": ("KT273DD_OMEGA",),
        "KT273DD_MESH_FRA": ("KT273DD_MESH_FRA", "KT273DD_MESH_FRAC"),
        "KT273RAIN": ("KT273RAIN",),
        "KT273HUMI.SPECIF": ("KT273HUMI.SPECIF", "KT273HUMI_SPECIF"),
    }
    resolved: dict[str, dict[str, str]] = {exp: {} for exp in EXPERIMENTS}
    for exp in EXPERIMENTS:
        for logical, cands in required_candidates.items():
            actual = resolve_var_name(variable_maps, exp, cands)
            if actual is not None:
                resolved[exp][logical] = actual

    print("\nResolved variable names:", flush=True)
    for exp in EXPERIMENTS:
        print(f"- {exp}:", flush=True)
        for k in sorted(resolved[exp]):
            print(f"    {k} -> {resolved[exp][k]}", flush=True)

    # Height axes per period/experiment.
    axis_by_period_exp: dict[tuple[str, str], VerticalAxis] = {}
    for period in periods:
        for exp in EXPERIMENTS:
            cache_file = build_cache_file(
                intermediate_dir=intermediate_dir,
                analysis_name="geopotential",
                period_subdir=period.output_subdir,
                experiment=exp,
                spatial_tag=spatial_tag,
            )
            if cache_file.exists() and not args.overwrite_intermediate:
                payload = load_cache(cache_file)
                height_m = payload.get("height_m")
                if height_m is None:
                    raise RuntimeError(f"Corrupt height cache: {cache_file}")
            else:
                height_m, _ = compute_geopotential_height_profile(
                    geopotential_dir=geopotential_dirs[exp],
                    height_variable=args.height_variable,
                    max_days=args.max_days,
                    allowed_months=period.allowed_months,
                    utc_offset_hours=args.utc_offset_hours,
                    aggregate=args.height_aggregate,
                    spatial_window=spatial_window,
                )
                if args.max_days is None:
                    save_cache(cache_file, {"height_m": np.asarray(height_m, dtype=np.float64)})

            axis_by_period_exp[(period.key, exp)] = VerticalAxis(
                values=np.asarray(height_m, dtype=np.float64) / 1000.0,
                label="Height (km)",
                is_height_km=True,
            )

    # Storage for diagnostics.
    vertical_results: dict[tuple[str, str, str], np.ndarray] = {}
    line_results: dict[tuple[str, str, str], np.ndarray] = {}

    for period in periods:
        print(f"\n===== Computing diagnostics for {period.label} ({period.key}) =====", flush=True)
        for exp in EXPERIMENTS:
            print(f"[{period.key}/{exp}] collecting records...", flush=True)
            # Pick a stable reference variable for records.
            ref_candidates = (
                "PRESSURE",
                "CV_PREC_FLUX",
                "DD_OMEGA",
                "KT273GRAUPEL",
            )
            ref_var = next((resolved[exp].get(v) for v in ref_candidates if resolved[exp].get(v)), None)
            if ref_var is None:
                print(f"[warn] {period.key}/{exp}: no reference variable found; skipping.", flush=True)
                continue
            ref_dir = experiment_dirs[exp] / ref_var
            records = collect_file_records(
                variable_dir=ref_dir,
                max_days=args.max_days,
                allowed_months=period.allowed_months,
                utc_offset_hours=args.utc_offset_hours,
            )
            if not records:
                print(f"[warn] {period.key}/{exp}: no records found in {ref_dir}", flush=True)
                continue

            # Analysis 1: downdraft/updraft diagnostics
            freezing_lines: dict[str, np.ndarray] = {}
            for fexp in EXPERIMENTS:
                axis_fexp = axis_by_period_exp[(period.key, fexp)]
                
                temp_cache_file = intermediate_dir / "temperature" / period.output_subdir / f"{fexp}_{spatial_tag}_diurnal_profile.npz"
                if temp_cache_file.exists():
                    try:
                        temp_profile = np.load(temp_cache_file, allow_pickle=True)["mean"]
                    except Exception:
                        temp_profile = None
                else:
                    temp_profile = vertical_results.get((period.key, "TEMPERATURE", fexp))
                    
                if temp_profile is not None:
                    fl = compute_freezing_line_km(axis_fexp, [temp_profile])
                    if fl is not None:
                        freezing_lines[fexp] = fl

            if "downdraft" in analyses:
                needed = ("UD_OMEGA", "UD_MESH_FRAC", "DD_OMEGA", "DD_MESH_FRAC")
                if all(k in resolved[exp] for k in needed):
                    cache_file = build_cache_file(
                        intermediate_dir=intermediate_dir,
                        analysis_name="downdraft",
                        period_subdir=period.output_subdir,
                        experiment=exp,
                        spatial_tag=spatial_tag,
                    )
                    if cache_file.exists() and not args.overwrite_intermediate:
                        payload = load_cache(cache_file)
                    else:
                        payload = compute_downdraft_profiles(
                            experiment=exp,
                            experiment_dir=experiment_dirs[exp],
                            records=records,
                            names={
                                k: resolved[exp][k]
                                for k in ("UD_OMEGA", "UD_MESH_FRAC", "DD_OMEGA", "DD_MESH_FRAC")
                            },
                            spatial_window=spatial_window,
                        )
                        if args.max_days is None:
                            save_cache(cache_file, payload)
                    for diag, arr in payload.items():
                        vertical_results[(period.key, diag, exp)] = arr
                else:
                    missing = [k for k in needed if k not in resolved[exp]]
                    print(
                        f"[warn] {period.key}/{exp}: skipping downdraft analysis; missing {missing}",
                        flush=True,
                    )

            # Analysis 2: precipitation diagnostics
            if "precip" in analyses:
                needed = ("CV_PREC_FLUX", "ST_PREC_FLUX")
                if all(k in resolved[exp] for k in needed):
                    cache_file = build_cache_file(
                        intermediate_dir=intermediate_dir,
                        analysis_name="precip",
                        period_subdir=period.output_subdir,
                        experiment=exp,
                        spatial_tag=spatial_tag,
                    )
                    if cache_file.exists() and not args.overwrite_intermediate:
                        payload = load_cache(cache_file)
                    else:
                        payload = compute_precip_profiles(
                            experiment=exp,
                            experiment_dir=experiment_dirs[exp],
                            records=records,
                            names={"CV_PREC_FLUX": resolved[exp]["CV_PREC_FLUX"], "ST_PREC_FLUX": resolved[exp]["ST_PREC_FLUX"]},
                            spatial_window=spatial_window,
                        )
                        if args.max_days is None:
                            save_cache(cache_file, payload)
                    for diag, arr in payload.items():
                        vertical_results[(period.key, diag, exp)] = arr
                else:
                    missing = [k for k in needed if k not in resolved[exp]]
                    print(
                        f"[warn] {period.key}/{exp}: skipping precip analysis; missing {missing}",
                        flush=True,
                    )

            # Analysis 3: thermo diagnostics
            if "thermo" in analyses:
                needed = ("TEMPERATURE", "HUMI.SPECIFI", "PRESSURE")
                if all(k in resolved[exp] for k in needed):
                    z_km = axis_by_period_exp[(period.key, exp)].values
                    cache_file = build_cache_file(
                        intermediate_dir=intermediate_dir,
                        analysis_name="thermo",
                        period_subdir=period.output_subdir,
                        experiment=exp,
                        spatial_tag=spatial_tag,
                    )
                    if cache_file.exists() and not args.overwrite_intermediate:
                        payload = load_cache(cache_file)
                    else:
                        payload = compute_thermo_profiles(
                            experiment=exp,
                            experiment_dir=experiment_dirs[exp],
                            records=records,
                            names={
                                "TEMPERATURE": resolved[exp]["TEMPERATURE"],
                                "HUMI.SPECIFI": resolved[exp]["HUMI.SPECIFI"],
                                "PRESSURE": resolved[exp]["PRESSURE"],
                            },
                            z_m=np.asarray(z_km, dtype=np.float64) * 1000.0,
                            spatial_window=spatial_window,
                        )
                        if args.max_days is None:
                            save_cache(cache_file, payload)
                    for diag, arr in payload.items():
                        vertical_results[(period.key, diag, exp)] = arr
                else:
                    missing = [k for k in needed if k not in resolved[exp]]
                    print(
                        f"[warn] {period.key}/{exp}: skipping thermo analysis; missing {missing}",
                        flush=True,
                    )

            # Analysis 4: KT273 line diagnostics
            if "kt273" in analyses:
                needed = (
                    "KT273GRAUPEL",
                    "KT273DD_OMEGA",
                    "KT273DD_MESH_FRA",
                    "KT273RAIN",
                    "KT273HUMI.SPECIF",
                )
                if all(k in resolved[exp] for k in needed):
                    cache_file = build_cache_file(
                        intermediate_dir=intermediate_dir,
                        analysis_name="kt273",
                        period_subdir=period.output_subdir,
                        experiment=exp,
                        spatial_tag=spatial_tag,
                    )
                    if cache_file.exists() and not args.overwrite_intermediate:
                        payload = load_cache(cache_file)
                    else:
                        payload = compute_kt273_lines(
                            experiment=exp,
                            experiment_dir=experiment_dirs[exp],
                            records=records,
                            names={k: resolved[exp][k] for k in needed},
                            spatial_window=spatial_window,
                        )
                        if args.max_days is None:
                            save_cache(cache_file, payload)
                    for diag, arr in payload.items():
                        line_results[(period.key, diag, exp)] = arr
                else:
                    missing = [k for k in needed if k not in resolved[exp]]
                    print(
                        f"[warn] {period.key}/{exp}: skipping kt273 analysis; missing {missing}",
                        flush=True,
                    )

            # Analysis 5: column diagnostics
            if "column" in analyses:
                needed = ("PRESSURE", "CV_PREC_FLUX", "ST_PREC_FLUX")
                condensate_keys = ("LIQUID_WATER", "RAIN", "SNOW", "SOLID_WATER", "GRAUPEL")
                available_cond = [resolved[exp][k] for k in condensate_keys if k in resolved[exp]]
                if all(k in resolved[exp] for k in needed) and available_cond:
                    cache_file = build_cache_file(
                        intermediate_dir=intermediate_dir,
                        analysis_name="column",
                        period_subdir=period.output_subdir,
                        experiment=exp,
                        spatial_tag=spatial_tag,
                    )
                    if cache_file.exists() and not args.overwrite_intermediate:
                        payload = load_cache(cache_file)
                    else:
                        payload = compute_column_lines(
                            experiment=exp,
                            experiment_dir=experiment_dirs[exp],
                            records=records,
                            names={
                                "PRESSURE": resolved[exp]["PRESSURE"],
                                "CV_PREC_FLUX": resolved[exp]["CV_PREC_FLUX"],
                                "ST_PREC_FLUX": resolved[exp]["ST_PREC_FLUX"],
                            },
                            available_condensate_names=available_cond,
                            spatial_window=spatial_window,
                        )
                        if args.max_days is None:
                            save_cache(cache_file, payload)
                    for diag, arr in payload.items():
                        line_results[(period.key, diag, exp)] = arr
                else:
                    missing = [k for k in needed if k not in resolved[exp]]
                    if not available_cond:
                        missing.append("all condensate vars")
                    print(
                        f"[warn] {period.key}/{exp}: skipping column analysis; missing {missing}",
                        flush=True,
                    )

    # Derived profile: d(total_prec_flux)/dz after height alignment.
    if "precip" in analyses:
        for period in periods:
            key = period.key
            needed = [(key, "TOTAL_PREC_FLUX", exp) for exp in EXPERIMENTS]
            if not all(k in vertical_results for k in needed):
                continue
            axis_ctrl = axis_by_period_exp[(key, "control")]
            total_profiles: dict[str, np.ndarray] = {}
            for exp in EXPERIMENTS:
                axis_exp = axis_by_period_exp[(key, exp)]
                prof = vertical_results[(key, "TOTAL_PREC_FLUX", exp)]
                axis_exp, prof = align_axis_and_profile(axis_exp, prof)
                if exp == "control":
                    total_profiles[exp] = prof
                    continue
                total_profiles[exp] = interpolate_profile_to_target_height(
                    source_height_km=axis_exp.values,
                    source_profile=prof,
                    target_height_km=axis_ctrl.values[: total_profiles.get("control", prof).shape[0]],
                )
            nlev = min(total_profiles[exp].shape[0] for exp in EXPERIMENTS)
            target_h = axis_ctrl.values[:nlev]
            for exp in EXPERIMENTS:
                total_profiles[exp] = total_profiles[exp][:nlev, :]
                vertical_results[(key, "TOTAL_PREC_FLUX_DFDZ", exp)] = compute_vertical_derivative(
                    total_profiles[exp], target_h
                )

    # Plotting metadata.
    diag_meta = {
        "DD_EXTENT": {"label": "Downdraft extent", "unit": "1", "abs_linear": True},
        "DD_FLUX": {"label": "Downdraft mass flux", "unit": "kg/m²/s", "abs_linear": True},
        "DD_INTENSITY": {"label": "Downdraft intensity |DD_OMEGA|", "unit": "Pa/s", "abs_linear": True},
        "NET_CONVECTIVE_FLUX": {"label": "Net convective mass flux (UD−DD)", "unit": "kg/m²/s", "abs_linear": True},
        "CV_PREC_FLUX": {"label": "Convective precip flux", "unit": "", "abs_linear": True},
        "ST_PREC_FLUX": {"label": "Stratiform precip flux", "unit": "", "abs_linear": True},
        "TOTAL_PREC_FLUX": {"label": "Total precip flux", "unit": "", "abs_linear": True},
        "CONVECTIVE_PRECIP_FRACTION": {"label": "Convective precip fraction", "unit": "1", "abs_linear": True},
        "TOTAL_PREC_FLUX_DFDZ": {"label": "d(Total precip flux)/dz", "unit": "m-1", "abs_linear": True},
        "THETA_E": {"label": "Equivalent potential temperature", "unit": "K", "abs_linear": True},
        "MSE": {"label": "Moist static energy", "unit": "J/kg", "abs_linear": True},
        "DD_TO_UD_FLUX_RATIO": {"label": "Downdraft-to-updraft flux ratio", "unit": "1", "abs_linear": True},
    }

    # 3-panel anomaly plots.
    three_panel_diags = []
    if "downdraft" in analyses:
        three_panel_diags.extend(("DD_EXTENT", "DD_FLUX", "DD_INTENSITY", "NET_CONVECTIVE_FLUX"))
    if "precip" in analyses:
        three_panel_diags.extend(
            ("CV_PREC_FLUX", "ST_PREC_FLUX", "TOTAL_PREC_FLUX", "CONVECTIVE_PRECIP_FRACTION", "TOTAL_PREC_FLUX_DFDZ")
        )
    if "thermo" in analyses:
        three_panel_diags.extend(("THETA_E", "MSE"))

    for period in periods:
        key = period.key
        print(f"\n===== Plotting {period.label} ({period.key}) =====", flush=True)

        freezing_lines: dict[str, np.ndarray] = {}
        for exp in EXPERIMENTS:
            axis_exp = axis_by_period_exp[(key, exp)]
            temp_profile = None
            # Try loading from temperature npz cache first
            temp_cache_file = intermediate_dir / "temperature" / period.output_subdir / f"{exp}_{spatial_tag}_diurnal_profile.npz"
            if temp_cache_file.exists():
                try:
                    temp_profile = np.load(temp_cache_file, allow_pickle=True)["mean"]
                except Exception:
                    temp_profile = None
            # Fallback to vertical_results
            if temp_profile is None:
                temp_profile = vertical_results.get((key, "TEMPERATURE", exp))
            if temp_profile is not None:
                fl = compute_freezing_line_km(axis_exp, [temp_profile])
                if fl is not None:
                    freezing_lines[exp] = fl

        for diag in three_panel_diags:
            if not all((key, diag, exp) in vertical_results for exp in EXPERIMENTS):
                continue
            axis_ctrl = axis_by_period_exp[(key, "control")]
            axis_profiles: dict[str, np.ndarray] = {}
            for exp in EXPERIMENTS:
                axis_exp = axis_by_period_exp[(key, exp)]
                arr = vertical_results[(key, diag, exp)]
                axis_exp, arr = align_axis_and_profile(axis_exp, arr)
                if exp == "control":
                    axis_profiles[exp] = arr
                else:
                    axis_profiles[exp] = interpolate_profile_to_target_height(
                        source_height_km=axis_exp.values,
                        source_profile=arr,
                        target_height_km=axis_ctrl.values,
                    )

            nlev = min(axis_profiles[exp].shape[0] for exp in EXPERIMENTS)
            axis = VerticalAxis(values=axis_ctrl.values[:nlev], label="Height (km)", is_height_km=True)
            c = axis_profiles["control"][:nlev, :]
            g = axis_profiles["graupel"][:nlev, :]
            t = axis_profiles["2mom"][:nlev, :]

            out = (
                output_dir
                / "height_hour_panels"
                / safe_name(diag)
                / period.output_subdir
                / (
                    f"{safe_name(diag)}_panel_"
                    "c1m_g1m-c1m_g2m-g1m"
                    + (f"_{spatial_tag}" if spatial_tag != "full-domain" else "")
                    + ".png"
                )
            )
            meta = diag_meta[diag]
            plot_three_panel_anomaly(
                title_label=meta["label"],
                unit=meta["unit"],
                control=c,
                graupel=g,
                twomom=t,
                axis=axis,
                output_file=out,
                period_label=period.label,
                max_height_km=args.max_height_km,
                abs_linear=bool(meta.get("abs_linear", True)),
                freezing_lines=freezing_lines,
            )

        # Ratio absolute panels by experiment.
        if "downdraft" in analyses and all(
            (key, "DD_TO_UD_FLUX_RATIO", exp) in vertical_results for exp in EXPERIMENTS
        ):
            axis_ctrl = axis_by_period_exp[(key, "control")]
            ratio_profiles: dict[str, np.ndarray] = {}
            for exp in EXPERIMENTS:
                axis_exp = axis_by_period_exp[(key, exp)]
                arr = vertical_results[(key, "DD_TO_UD_FLUX_RATIO", exp)]
                axis_exp, arr = align_axis_and_profile(axis_exp, arr)
                if exp == "control":
                    ratio_profiles[exp] = arr
                else:
                    ratio_profiles[exp] = interpolate_profile_to_target_height(
                        source_height_km=axis_exp.values,
                        source_profile=arr,
                        target_height_km=axis_ctrl.values,
                    )
            nlev = min(ratio_profiles[exp].shape[0] for exp in EXPERIMENTS)
            axis = VerticalAxis(values=axis_ctrl.values[:nlev], label="Height (km)", is_height_km=True)
            ratio_profiles = {exp: arr[:nlev, :] for exp, arr in ratio_profiles.items()}
            out = (
                output_dir
                / "ratio_panels"
                / "dd_to_ud_flux_ratio"
                / period.output_subdir
                / (
                    "dd_to_ud_flux_ratio_experiment_panels"
                    + (f"_{spatial_tag}" if spatial_tag != "full-domain" else "")
                    + ".png"
                )
            )
            plot_three_experiment_absolute(
                title_label="Downdraft-to-updraft flux ratio",
                unit="1",
                profiles=ratio_profiles,
                axis=axis,
                output_file=out,
                period_label=period.label,
                max_height_km=args.max_height_km,
                diverging=True,
            )

        # KT273 stacked lines.
        if "kt273" in analyses:
            order = ("KT273GRAUPEL", "KT273DD_FLUX", "KT273RAIN", "KT273HUMI_SPECIF")
            if all((key, d, exp) in line_results for d in order for exp in EXPERIMENTS):
                line_data = {
                    d: {exp: line_results[(key, d, exp)] for exp in EXPERIMENTS}
                    for d in order
                }
                out = (
                    output_dir
                    / "line_panels"
                    / "kt273"
                    / period.output_subdir
                    / (
                        "kt273_stacked_diurnal"
                        + (f"_{spatial_tag}" if spatial_tag != "full-domain" else "")
                        + ".png"
                    )
                )
                plot_lines_stacked(
                    title="Freezing-level diagnostics (KT273)",
                    period_label=period.label,
                    diag_order=order,
                    diag_labels={
                        "KT273GRAUPEL": "KT273GRAUPEL",
                        "KT273DD_FLUX": "KT273DD_OMEGA x KT273DD_MESH_FRA",
                        "KT273RAIN": "KT273RAIN",
                        "KT273HUMI_SPECIF": "KT273HUMI.SPECIF",
                    },
                    unit_map={
                        "KT273GRAUPEL": "",
                        "KT273DD_FLUX": "Pa/s",
                        "KT273RAIN": "",
                        "KT273HUMI_SPECIF": "kg/kg",
                    },
                    line_data=line_data,
                    output_file=out,
                )

        # Column lines.
        if "column" in analyses:
            order = ("COLUMN_CONDENSATE", "SURFACE_PRECIP_FLUX", "RESIDENCE_TIME")
            if all((key, d, exp) in line_results for d in order for exp in EXPERIMENTS):
                line_data = {
                    d: {exp: line_results[(key, d, exp)] for exp in EXPERIMENTS}
                    for d in order
                }
                out = (
                    output_dir
                    / "line_panels"
                    / "column"
                    / period.output_subdir
                    / (
                        "column_condensate_residence_diurnal"
                        + (f"_{spatial_tag}" if spatial_tag != "full-domain" else "")
                        + ".png"
                    )
                )
                plot_lines_stacked(
                    title="Column condensate and residence diagnostics",
                    period_label=period.label,
                    diag_order=order,
                    diag_labels={
                        "COLUMN_CONDENSATE": "Column condensate",
                        "SURFACE_PRECIP_FLUX": "Lowest-level precip flux",
                        "RESIDENCE_TIME": "Residence time (proxy)",
                    },
                    unit_map={
                        "COLUMN_CONDENSATE": "kg/m2",
                        "SURFACE_PRECIP_FLUX": "",
                        "RESIDENCE_TIME": "s",
                    },
                    line_data=line_data,
                    output_file=out,
                )

        # BL theta-e gradient line.
        if "thermo" in analyses and all(
            (key, "THETA_E", exp) in vertical_results for exp in EXPERIMENTS
        ):
            axis_ctrl = axis_by_period_exp[(key, "control")]
            theta_profiles: dict[str, np.ndarray] = {}
            for exp in EXPERIMENTS:
                axis_exp = axis_by_period_exp[(key, exp)]
                arr = vertical_results[(key, "THETA_E", exp)]
                axis_exp, arr = align_axis_and_profile(axis_exp, arr)
                if exp == "control":
                    theta_profiles[exp] = arr
                else:
                    theta_profiles[exp] = interpolate_profile_to_target_height(
                        source_height_km=axis_exp.values,
                        source_profile=arr,
                        target_height_km=axis_ctrl.values,
                    )
            nlev = min(theta_profiles[exp].shape[0] for exp in EXPERIMENTS)
            z = axis_ctrl.values[:nlev]
            bl_lines = {
                exp: compute_bl_gradient_line(theta_profiles[exp][:nlev, :], z)
                for exp in EXPERIMENTS
            }
            out = (
                output_dir
                / "line_panels"
                / "thermo"
                / period.output_subdir
                / (
                    "thetae_bl_gradient_diurnal"
                    + (f"_{spatial_tag}" if spatial_tag != "full-domain" else "")
                    + ".png"
                )
            )
            plot_lines_stacked(
                title="Boundary-layer theta-e gradient (0-2 km)",
                period_label=period.label,
                diag_order=("THETAE_BL_GRADIENT",),
                diag_labels={"THETAE_BL_GRADIENT": "d(theta-e)/dz (0-2 km)"},
                unit_map={"THETAE_BL_GRADIENT": "K/km"},
                line_data={"THETAE_BL_GRADIENT": bl_lines},
                output_file=out,
            )

    print("\nCompleted extended diagnostics.", flush=True)


if __name__ == "__main__":
    main()
