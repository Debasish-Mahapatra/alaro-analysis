"""Surface."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from alaro_analysis.common.constants import CP_D, G
from alaro_analysis.common.naming import normalize_var_token
from alaro_analysis.common.surface_settings import HEIGHT_LEVEL_RE, PBLH_VAR_TOKEN
from alaro_analysis.common.timeparse import parse_month_from_day_name
from alaro_analysis.data.dataset_io import read_time_level_yx


def safe_scalar_mean(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def is_pblh_variable(name: str) -> bool:
    return normalize_var_token(name) == PBLH_VAR_TOKEN


def infer_lcl_start_height_m(variable_names: Sequence[str], override: float | None) -> float:
    if override is not None:
        return float(override)
    for name in variable_names:
        match = HEIGHT_LEVEL_RE.match(normalize_var_token(name))
        if match:
            return float(int(match.group(1)))
    return 0.0


def coerce_pressure_to_pa(values: np.ndarray) -> np.ndarray:
    pressure = np.asarray(values, dtype=np.float64)
    finite = pressure[np.isfinite(pressure)]
    if finite.size == 0:
        return pressure

    p01 = float(np.nanpercentile(finite, 1))
    p99 = float(np.nanpercentile(finite, 99))
    if 5.0 <= p01 <= 15.0 and 5.0 <= p99 <= 20.0:
        pressure = np.exp(pressure)  # SURFPRESSION is log(surface pressure in Pa).
    elif 100.0 <= p01 <= 1200.0 and 100.0 <= p99 <= 2000.0:
        pressure = pressure * 100.0

    return np.where(pressure > 0.0, pressure, np.nan)


def coerce_specific_humidity(values: np.ndarray) -> np.ndarray:
    q = np.asarray(values, dtype=np.float64)
    finite = q[np.isfinite(q)]
    if finite.size > 0:
        q99 = float(np.nanpercentile(finite, 99))
        if 1.0 < q99 <= 80.0:
            q = q / 1000.0
    return np.where((q > 0.0) & (q < 0.08), q, np.nan)


def calculate_lcl_height_m(
    *,
    temperature_k: np.ndarray,
    specific_humidity: np.ndarray,
    pressure_pa: np.ndarray,
    start_height_m: float,
) -> np.ndarray:
    try:
        import metpy.calc as mpcalc
        from metpy.units import units
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MetPy is required for --lcl-overlay. Install the HPC/full extras "
            "or add metpy to the active environment."
        ) from exc

    t = np.asarray(temperature_k, dtype=np.float64)
    q = coerce_specific_humidity(specific_humidity)
    p = coerce_pressure_to_pa(pressure_pa)
    valid = np.isfinite(t) & np.isfinite(q) & np.isfinite(p)

    t_units = np.where(valid, t, np.nan) * units.kelvin
    q_units = np.where(valid, q, np.nan) * units.dimensionless
    p_units = np.where(valid, p, np.nan) * units.pascal

    try:
        dewpoint = mpcalc.dewpoint_from_specific_humidity(p_units, q_units)
    except TypeError:
        dewpoint = mpcalc.dewpoint_from_specific_humidity(p_units, t_units, q_units)

    _lcl_pressure, lcl_temperature = mpcalc.lcl(p_units, t_units, dewpoint)
    delta_t = (t_units.to("kelvin") - lcl_temperature.to("kelvin")).to("kelvin").magnitude
    dry_lapse_rate = G / CP_D
    lift_height = np.maximum(delta_t / dry_lapse_rate, 0.0)
    lcl_height = lift_height + float(start_height_m)
    return np.where(valid & np.isfinite(lcl_height), lcl_height, np.nan)


def compute_surface_line(
    records: list[tuple[int, Path]],
    variable_name: str,
    spatial_window,
) -> tuple[np.ndarray, np.ndarray, int]:
    sums = np.zeros((24,), dtype=np.float64)
    counts = np.zeros((24,), dtype=np.int64)
    used = 0

    for idx, (hour, file_path) in enumerate(records, start=1):
        arr = read_time_level_yx(
            file_path,
            variable_name,
            spatial_window=spatial_window,
            token_normalizer=normalize_var_token,
        )
        value = safe_scalar_mean(arr)
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


def compute_lcl_line(
    records: list[tuple[int, Path]],
    experiment_dir: Path,
    *,
    temperature_name: str,
    specific_humidity_name: str,
    pressure_name: str,
    spatial_window,
    start_height_m: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    sums = np.zeros((24,), dtype=np.float64)
    counts = np.zeros((24,), dtype=np.int64)
    used = 0

    for idx, (hour, base_file) in enumerate(records, start=1):
        day_name = base_file.parent.name
        file_name = base_file.name
        t_file = experiment_dir / temperature_name / day_name / file_name
        q_file = experiment_dir / specific_humidity_name / day_name / file_name
        p_file = experiment_dir / pressure_name / day_name / file_name
        if not (t_file.exists() and q_file.exists() and p_file.exists()):
            continue

        temperature = read_time_level_yx(
            t_file,
            temperature_name,
            spatial_window=spatial_window,
            token_normalizer=normalize_var_token,
        )
        specific_humidity = read_time_level_yx(
            q_file,
            specific_humidity_name,
            spatial_window=spatial_window,
            token_normalizer=normalize_var_token,
        )
        pressure = read_time_level_yx(
            p_file,
            pressure_name,
            spatial_window=spatial_window,
            token_normalizer=normalize_var_token,
        )
        lcl_height = calculate_lcl_height_m(
            temperature_k=temperature,
            specific_humidity=specific_humidity,
            pressure_pa=pressure,
            start_height_m=start_height_m,
        )
        value = safe_scalar_mean(lcl_height)
        used += 1
        if np.isfinite(value):
            sums[hour] += value
            counts[hour] += 1

        if idx % 2000 == 0 or idx == len(records):
            print(f"[LCL] {idx}/{len(records)} files", flush=True)

    mean = np.full((24,), np.nan, dtype=np.float64)
    valid = counts > 0
    mean[valid] = sums[valid] / counts[valid]
    return mean, counts, used


def compute_lcl_lines_for_periods(
    records: list[tuple[int, Path]],
    period_specs,
    experiment_dir: Path,
    *,
    temperature_name: str,
    specific_humidity_name: str,
    pressure_name: str,
    spatial_window,
    start_height_m: float,
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    sums = {period.key: np.zeros((24,), dtype=np.float64) for period in period_specs}
    counts = {period.key: np.zeros((24,), dtype=np.int64) for period in period_specs}
    used = {period.key: 0 for period in period_specs}
    allowed_months = {
        period.key: set(period.allowed_months) if period.allowed_months is not None else None
        for period in period_specs
    }

    for idx, (hour, base_file) in enumerate(records, start=1):
        month = parse_month_from_day_name(base_file.parent.name)
        matched_keys = [
            period.key
            for period in period_specs
            if allowed_months[period.key] is None
            or (month is not None and month in allowed_months[period.key])
        ]
        if not matched_keys:
            continue

        day_name = base_file.parent.name
        file_name = base_file.name
        t_file = experiment_dir / temperature_name / day_name / file_name
        q_file = experiment_dir / specific_humidity_name / day_name / file_name
        p_file = experiment_dir / pressure_name / day_name / file_name
        if not (t_file.exists() and q_file.exists() and p_file.exists()):
            continue

        temperature = read_time_level_yx(
            t_file,
            temperature_name,
            spatial_window=spatial_window,
            token_normalizer=normalize_var_token,
        )
        specific_humidity = read_time_level_yx(
            q_file,
            specific_humidity_name,
            spatial_window=spatial_window,
            token_normalizer=normalize_var_token,
        )
        pressure = read_time_level_yx(
            p_file,
            pressure_name,
            spatial_window=spatial_window,
            token_normalizer=normalize_var_token,
        )
        lcl_height = calculate_lcl_height_m(
            temperature_k=temperature,
            specific_humidity=specific_humidity,
            pressure_pa=pressure,
            start_height_m=start_height_m,
        )
        value = safe_scalar_mean(lcl_height)
        for key in matched_keys:
            used[key] += 1
            if np.isfinite(value):
                sums[key][hour] += value
                counts[key][hour] += 1

        if idx % 2000 == 0 or idx == len(records):
            print(f"[LCL] {idx}/{len(records)} files", flush=True)

    out: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for period in period_specs:
        mean = np.full((24,), np.nan, dtype=np.float64)
        valid = counts[period.key] > 0
        mean[valid] = sums[period.key][valid] / counts[period.key][valid]
        out[period.key] = (mean, counts[period.key], used[period.key])
    return out


def lcl_cache_analysis_name(
    variable: str,
    temperature_name: str,
    specific_humidity_name: str,
    pressure_name: str,
    start_height_m: float,
) -> str:
    return (
        f"{variable}_lcl_{temperature_name}_"
        f"{specific_humidity_name}_{pressure_name}_z{start_height_m:g}"
    )
