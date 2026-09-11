"""Radiation compare."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alaro_analysis.common.constants import DAY_RE, FILE_HOUR_RE
from alaro_analysis.common.naming import normalize_var_token
from alaro_analysis.common.timeparse import parse_month_from_day_name
from alaro_analysis.data.dataset_io import read_time_level_yx


@dataclass(frozen=True)
class AlaroDaySteps:
    day_name: str
    step_files: tuple[tuple[int, Path], ...]


def safe_scalar_mean(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def collect_alaro_day_steps(
    *,
    variable_dir: Path,
    max_days: int | None,
    allowed_months: tuple[int, ...] | None,
) -> list[AlaroDaySteps]:
    if not variable_dir.exists():
        raise FileNotFoundError(f"Missing directory: {variable_dir}")

    allowed_set = set(allowed_months) if allowed_months is not None else None
    day_dirs = sorted(
        path
        for path in variable_dir.iterdir()
        if path.is_dir() and DAY_RE.match(path.name) is not None
    )
    if max_days is not None:
        day_dirs = day_dirs[:max_days]

    day_steps: list[AlaroDaySteps] = []
    missing_boundary_days: list[str] = []
    for day_dir in day_dirs:
        month = parse_month_from_day_name(day_dir.name)
        if allowed_set is not None and (month is None or month not in allowed_set):
            continue

        files_by_step: dict[int, Path] = {}
        for file_path in sorted(day_dir.glob("*.nc")):
            match = FILE_HOUR_RE.search(file_path.name)
            if match is None:
                continue
            step_hour = int(match.group(1))
            if step_hour < 0 or step_hour > 24:
                continue
            files_by_step.setdefault(step_hour, file_path)

        missing_boundary = [hour for hour in (0, 24) if hour not in files_by_step]
        if missing_boundary:
            missing_text = ", ".join(f"+{hour:04d}" for hour in missing_boundary)
            missing_boundary_days.append(f"{day_dir.name} ({missing_text})")
            continue

        ordered = tuple(sorted(files_by_step.items()))
        if len(ordered) < 2:
            continue
        day_steps.append(AlaroDaySteps(day_name=day_dir.name, step_files=ordered))

    if missing_boundary_days:
        preview = "; ".join(missing_boundary_days[:5])
        extra = len(missing_boundary_days) - min(len(missing_boundary_days), 5)
        if extra > 0:
            preview = f"{preview}; ... and {extra} more days"
        raise RuntimeError(
            "ALARO radiation comparison requires converted +0000 and +0024 files for every selected day. "
            f"Missing boundary steps in: {preview}. Reconvert the ALARO radiation variables with "
            "--include-hour24 and rerun the comparison."
        )

    return day_steps


def get_peer_file(base_file: Path, experiment_dir: Path, var_name: str) -> Path:
    return experiment_dir / var_name / base_file.parent.name / base_file.name


def read_mean_scalar(
    file_path: Path,
    variable_name: str,
    spatial_window,
) -> float:
    arr = read_time_level_yx(
        file_path,
        variable_name,
        spatial_window=spatial_window,
        token_normalizer=normalize_var_token,
    )
    return safe_scalar_mean(arr)


def finalize_line_means(
    sums: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, sum_arr in sums.items():
        cnt_arr = counts[key]
        mean = np.full(sum_arr.shape, np.nan, dtype=np.float64)
        valid = cnt_arr > 0
        mean[valid] = sum_arr[valid] / cnt_arr[valid]
        out[key] = mean
    return out


def read_alaro_step_scalars(
    *,
    experiment: str,
    experiment_dir: Path,
    base_file: Path,
    names: dict[str, str | None],
    spatial_window,
) -> dict[str, float] | None:
    sw_net_name = names["SW_NET"]
    lw_net_name = names["LW_NET"]
    if sw_net_name is None or lw_net_name is None:
        raise ValueError("SW_NET and LW_NET are required ALARO variables.")

    step_values: dict[str, float] = {}
    required_files = {
        "SW_NET": (base_file, sw_net_name),
        "LW_NET": (get_peer_file(base_file, experiment_dir, lw_net_name), lw_net_name),
    }
    for key, (file_path, variable_name) in required_files.items():
        if not file_path.exists():
            print(
                f"[warn] {experiment}/alaro: missing required file {file_path.name} for {variable_name}",
                flush=True,
            )
            return None
        try:
            step_values[key] = read_mean_scalar(file_path, variable_name, spatial_window)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[warn] {experiment}/alaro: skipping unreadable step {file_path.name} ({exc})",
                flush=True,
            )
            return None

    for key in ("SW_DOWN", "LW_DOWN"):
        variable_name = names[key]
        if variable_name is None:
            step_values[key] = np.nan
            continue
        file_path = get_peer_file(base_file, experiment_dir, variable_name)
        if not file_path.exists():
            step_values[key] = np.nan
            continue
        try:
            step_values[key] = read_mean_scalar(file_path, variable_name, spatial_window)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[warn] {experiment}/alaro: could not read optional {file_path.name} ({exc})",
                flush=True,
            )
            step_values[key] = np.nan

    return step_values


def deaccumulate_alaro_pair(
    previous_values: dict[str, float],
    current_values: dict[str, float],
    delta_seconds: float,
) -> dict[str, float]:
    if delta_seconds <= 0.0:
        raise ValueError("delta_seconds must be positive for ALARO de-accumulation.")

    fluxes: dict[str, float] = {}
    for key in ("SW_DOWN", "LW_DOWN", "SW_NET", "LW_NET"):
        previous = previous_values.get(key, np.nan)
        current = current_values.get(key, np.nan)
        if np.isfinite(previous) and np.isfinite(current):
            fluxes[key] = float(current - previous) / float(delta_seconds)
        else:
            fluxes[key] = np.nan

    fluxes["SW_UP"] = (
        fluxes["SW_DOWN"] - fluxes["SW_NET"]
        if np.isfinite(fluxes["SW_DOWN"]) and np.isfinite(fluxes["SW_NET"])
        else np.nan
    )
    fluxes["LW_UP"] = (
        fluxes["LW_DOWN"] - fluxes["LW_NET"]
        if np.isfinite(fluxes["LW_DOWN"]) and np.isfinite(fluxes["LW_NET"])
        else np.nan
    )
    fluxes["ALARO_RN"] = (
        fluxes["SW_NET"] + fluxes["LW_NET"]
        if np.isfinite(fluxes["SW_NET"]) and np.isfinite(fluxes["LW_NET"])
        else np.nan
    )
    return fluxes


def compute_alaro_lines(
    *,
    experiment: str,
    experiment_dir: Path,
    day_steps: list[AlaroDaySteps],
    names: dict[str, str | None],
    spatial_window,
    utc_offset_hours: int,
) -> dict[str, np.ndarray]:
    sums = {
        "SW_DOWN": np.zeros((24,), dtype=np.float64),
        "LW_DOWN": np.zeros((24,), dtype=np.float64),
        "SW_NET": np.zeros((24,), dtype=np.float64),
        "LW_NET": np.zeros((24,), dtype=np.float64),
        "SW_UP": np.zeros((24,), dtype=np.float64),
        "LW_UP": np.zeros((24,), dtype=np.float64),
        "ALARO_RN": np.zeros((24,), dtype=np.float64),
    }
    counts = {key: np.zeros((24,), dtype=np.int64) for key in sums}

    used_intervals = 0
    skipped_bad_steps = 0
    non_hour_intervals = 0
    print(
        f"[{experiment}/alaro] De-accumulating accumulated step fields to interval-mean W m-2 "
        "and assigning each increment to the ending local hour.",
        flush=True,
    )
    print(f"[{experiment}/alaro] +0024 found for all {len(day_steps)} selected days.", flush=True)

    for idx, day_record in enumerate(day_steps, start=1):
        previous_step_hour: int | None = None
        previous_values: dict[str, float] | None = None
        for step_hour, base_file in day_record.step_files:
            current_values = read_alaro_step_scalars(
                experiment=experiment,
                experiment_dir=experiment_dir,
                base_file=base_file,
                names=names,
                spatial_window=spatial_window,
            )
            if current_values is None:
                skipped_bad_steps += 1
                continue
            if previous_step_hour is None or previous_values is None:
                previous_step_hour = step_hour
                previous_values = current_values
                continue

            delta_hours = step_hour - previous_step_hour
            if delta_hours <= 0:
                previous_step_hour = step_hour
                previous_values = current_values
                continue
            if delta_hours != 1:
                non_hour_intervals += 1

            fluxes = deaccumulate_alaro_pair(
                previous_values,
                current_values,
                delta_seconds=float(delta_hours) * 3600.0,
            )
            local_hour = (step_hour + utc_offset_hours) % 24
            for key, value in fluxes.items():
                if np.isfinite(value):
                    sums[key][local_hour] += float(value)
                    counts[key][local_hour] += 1
            used_intervals += 1
            previous_step_hour = step_hour
            previous_values = current_values

        if idx % 100 == 0 or idx == len(day_steps):
            print(f"[{experiment}/alaro] {idx}/{len(day_steps)} days", flush=True)

    print(
        f"[{experiment}/alaro] used intervals: {used_intervals} | unreadable skipped: {skipped_bad_steps} "
        f"| non-hour intervals: {non_hour_intervals}",
        flush=True,
    )
    return finalize_line_means(sums, counts)


def compute_single_variable_line(
    *,
    experiment: str,
    variable_dir: Path,
    variable_name: str,
    records: list[tuple[int, Path]],
    spatial_window,
) -> dict[str, np.ndarray]:
    sums = {"SURFEX_RN": np.zeros((24,), dtype=np.float64)}
    counts = {"SURFEX_RN": np.zeros((24,), dtype=np.int64)}
    used = 0
    skipped_bad = 0

    for idx, (hour, file_path) in enumerate(records, start=1):
        try:
            value = read_mean_scalar(file_path, variable_name, spatial_window)
        except Exception as exc:  # noqa: BLE001
            skipped_bad += 1
            print(
                f"[warn] {experiment}/surfex: skipping unreadable file {file_path.name} ({exc})",
                flush=True,
            )
            continue
        used += 1
        if np.isfinite(value):
            sums["SURFEX_RN"][hour] += float(value)
            counts["SURFEX_RN"][hour] += 1

        if idx % 2000 == 0 or idx == len(records):
            print(f"[{experiment}/surfex] {idx}/{len(records)} files", flush=True)

    print(
        f"[{experiment}/surfex] used files: {used}/{len(records)} | unreadable skipped: {skipped_bad}",
        flush=True,
    )
    return finalize_line_means(sums, counts)


def compute_mae(left: np.ndarray, right: np.ndarray) -> float | None:
    mask = np.isfinite(left) & np.isfinite(right)
    if not np.any(mask):
        return None
    return float(np.mean(np.abs(left[mask] - right[mask])))
