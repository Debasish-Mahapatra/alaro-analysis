from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import xarray as xr

from alaro_analysis.data.dataset_io import resolve_data_var_name

DAY_DIR_RE = re.compile(r"^(?:pf|sfx)(\d{8})$")
HOUR_FILE_RE = re.compile(r"^.+\+(\d{4})(?:\.[^.]+)?$")
DEFAULT_PRECIP_VARS = ("SURFPREC.EAU.GEC", "SURFPREC.EAU.CON")


@dataclass(frozen=True)
class DeaccumConfig:
    input_root: Path
    output_root: Path
    variables: tuple[str, ...]
    workers: int = 1
    overwrite: bool = False
    compress: str = "zlib"
    compress_level: int = 1
    start_date: str | None = None
    end_date: str | None = None
    strict_hours: bool = True
    negative_tolerance: float = 1.0e-2
    clip_negative: bool = False
    quiet: bool = False


@dataclass(frozen=True)
class DayTask:
    variable: str
    input_day_dir: Path
    output_day_dir: Path


def _split_vars(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_PRECIP_VARS

    variables: list[str] = []
    for value in values:
        variables.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(dict.fromkeys(variables))


def _parse_date_token(value: str | None, name: str) -> datetime.date | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid {name}='{value}', expected YYYYMMDD") from exc


def _day_date(day_dir: Path) -> datetime.date | None:
    match = DAY_DIR_RE.fullmatch(day_dir.name)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def _lead_hour(path: Path) -> int | None:
    match = HOUR_FILE_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _discover_day_dirs(
    variable_root: Path,
    *,
    start_date: datetime.date | None,
    end_date: datetime.date | None,
) -> list[Path]:
    days: list[tuple[datetime.date, Path]] = []
    if not variable_root.is_dir():
        return []

    for path in variable_root.iterdir():
        if not path.is_dir():
            continue
        day = _day_date(path)
        if day is None:
            continue
        if start_date is not None and day < start_date:
            continue
        if end_date is not None and day > end_date:
            continue
        days.append((day, path))

    days.sort(key=lambda item: item[0])
    return [path for _, path in days]


def _discover_hourly_files(day_dir: Path) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in day_dir.iterdir():
        if not path.is_file():
            continue
        hour = _lead_hour(path)
        if hour is None:
            continue
        files.append((hour, path))
    files.sort(key=lambda item: item[0])
    return files


def _validate_hours(hourly_files: Sequence[tuple[int, Path]]) -> None:
    if not hourly_files:
        raise ValueError("no hourly NetCDF files found")

    hours = [hour for hour, _ in hourly_files]
    duplicated = sorted({hour for hour in hours if hours.count(hour) > 1})
    if duplicated:
        labels = ", ".join(f"+{hour:04d}" for hour in duplicated)
        raise ValueError(f"duplicate lead hours found: {labels}")

    first = hours[0]
    if first > 1:
        raise ValueError(
            f"first lead hour is +{first:04d}; cannot make an hourly first increment"
        )

    gaps = [
        (prev_hour, next_hour)
        for prev_hour, next_hour in zip(hours, hours[1:])
        if next_hour != prev_hour + 1
    ]
    if gaps:
        labels = ", ".join(
            f"+{prev_hour:04d}->+{next_hour:04d}" for prev_hour, next_hour in gaps
        )
        raise ValueError(f"non-consecutive lead hours found: {labels}")


def _prepare_increment(
    current: xr.DataArray,
    previous: xr.DataArray | None,
    *,
    tolerance: float,
    clip_negative: bool,
) -> tuple[xr.DataArray, dict[str, int]]:
    current_values = np.asarray(current.values, dtype=np.float64)
    if previous is None:
        increment_values = current_values.copy()
    else:
        previous_values = np.asarray(previous.values, dtype=np.float64)
        increment_values = current_values - previous_values

    finite = np.isfinite(increment_values)
    small_negative = finite & (increment_values < 0.0) & (increment_values >= -tolerance)
    large_negative = finite & (increment_values < -tolerance)

    small_negative_count = int(np.count_nonzero(small_negative))
    large_negative_count = int(np.count_nonzero(large_negative))

    if small_negative_count:
        increment_values[small_negative] = 0.0
    if clip_negative and large_negative_count:
        increment_values[large_negative] = 0.0

    increment = xr.DataArray(
        increment_values,
        dims=current.dims,
        coords=current.coords,
        attrs=dict(current.attrs),
        name=current.name,
    )
    return increment, {
        "small_negative_clipped": small_negative_count,
        "large_negative": large_negative_count,
        "negative_clipped": large_negative_count if clip_negative else 0,
    }


def _output_attrs(
    original_attrs: dict,
    *,
    source_file: Path,
    previous_file: Path | None,
    interval_hours: int,
) -> dict:
    attrs = dict(original_attrs)
    long_name = attrs.get("long_name")
    if long_name:
        attrs["long_name"] = f"Deaccumulated hourly {long_name}"
    attrs["accumulation"] = "deaccumulated"
    attrs["deaccumulation_method"] = "current accumulated value minus previous lead accumulated value"
    attrs["deaccumulation_interval_hours"] = int(interval_hours)
    attrs["source_accumulated_file"] = str(source_file)
    attrs["previous_accumulated_file"] = str(previous_file) if previous_file else ""
    return attrs


def _dataset_attrs(
    original_attrs: dict,
    *,
    source_file: Path,
    previous_file: Path | None,
    interval_hours: int,
) -> dict:
    attrs = dict(original_attrs)
    attrs["deaccumulated"] = "true"
    attrs["deaccumulation_method"] = (
        "Hourly increment from accumulated precipitation. "
        "For the first available lead, the accumulated value is kept as the interval amount."
    )
    attrs["deaccumulation_interval_hours"] = int(interval_hours)
    attrs["source_accumulated_file"] = str(source_file)
    attrs["previous_accumulated_file"] = str(previous_file) if previous_file else ""
    attrs["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return attrs


def process_day(task: DayTask, cfg: DeaccumConfig) -> dict[str, object]:
    try:
        hourly_files = _discover_hourly_files(task.input_day_dir)
        if cfg.strict_hours:
            _validate_hours(hourly_files)
        elif not hourly_files:
            raise ValueError("no hourly NetCDF files found")

        task.output_day_dir.mkdir(parents=True, exist_ok=True)

        previous_da: xr.DataArray | None = None
        previous_file: Path | None = None
        previous_hour: int | None = None
        written = 0
        skipped_existing = 0
        small_negative_clipped = 0
        large_negative = 0
        negative_clipped = 0

        for hour, source_file in hourly_files:
            out_file = task.output_day_dir / source_file.name
            interval_hours = hour if previous_hour is None else hour - previous_hour
            previous_for_increment = previous_file

            with xr.open_dataset(source_file, decode_times=False, engine="netcdf4") as ds:
                ds.load()
                var_name = resolve_data_var_name(ds, task.variable)
                current_da = ds[var_name]
                increment, stats = _prepare_increment(
                    current_da,
                    previous_da,
                    tolerance=cfg.negative_tolerance,
                    clip_negative=cfg.clip_negative,
                )

                previous_da = current_da.copy(deep=True)
                previous_file = source_file
                previous_hour = hour

                if out_file.exists() and not cfg.overwrite:
                    skipped_existing += 1
                    small_negative_clipped += int(stats["small_negative_clipped"])
                    large_negative += int(stats["large_negative"])
                    negative_clipped += int(stats["negative_clipped"])
                    continue

                out_ds = ds.copy(deep=True)
                increment.attrs = _output_attrs(
                    current_da.attrs,
                    source_file=source_file,
                    previous_file=previous_for_increment,
                    interval_hours=interval_hours,
                )
                out_ds[var_name] = increment
                out_ds.attrs = _dataset_attrs(
                    ds.attrs,
                    source_file=source_file,
                    previous_file=previous_for_increment,
                    interval_hours=interval_hours,
                )

                encoding = {}
                if cfg.compress == "zlib":
                    encoding[var_name] = {"zlib": True, "complevel": cfg.compress_level}
                out_ds.to_netcdf(out_file, mode="w", encoding=encoding)
                out_ds.close()

                written += 1
                small_negative_clipped += int(stats["small_negative_clipped"])
                large_negative += int(stats["large_negative"])
                negative_clipped += int(stats["negative_clipped"])

        return {
            "status": "ok",
            "variable": task.variable,
            "day": task.input_day_dir.name,
            "files": len(hourly_files),
            "written": written,
            "skipped_existing": skipped_existing,
            "small_negative_clipped": small_negative_clipped,
            "large_negative": large_negative,
            "negative_clipped": negative_clipped,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "variable": task.variable,
            "day": task.input_day_dir.name,
            "error": str(exc),
        }


def build_tasks(cfg: DeaccumConfig) -> tuple[list[DayTask], list[str]]:
    start_date = _parse_date_token(cfg.start_date, "start_date")
    end_date = _parse_date_token(cfg.end_date, "end_date")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    tasks: list[DayTask] = []
    missing_variables: list[str] = []
    for variable in cfg.variables:
        variable_root = cfg.input_root / variable
        if not variable_root.is_dir():
            missing_variables.append(variable)
            continue
        for day_dir in _discover_day_dirs(
            variable_root,
            start_date=start_date,
            end_date=end_date,
        ):
            tasks.append(
                DayTask(
                    variable=variable,
                    input_day_dir=day_dir,
                    output_day_dir=cfg.output_root / variable / day_dir.name,
                )
            )
    return tasks, missing_variables


def run_deaccumulation(cfg: DeaccumConfig) -> dict[str, object]:
    if not cfg.input_root.is_dir():
        raise FileNotFoundError(f"Input root not found or not a directory: {cfg.input_root}")
    if cfg.input_root.resolve() == cfg.output_root.resolve():
        raise ValueError("output_root must differ from input_root to avoid replacing accumulated data")
    if cfg.compress not in {"zlib", "none"}:
        raise ValueError("compress must be one of: zlib, none")
    if not (0 <= cfg.compress_level <= 9):
        raise ValueError("compress_level must be in range [0, 9]")

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    tasks, missing_variables = build_tasks(cfg)

    if not cfg.quiet:
        print(f"Input root: {cfg.input_root}", flush=True)
        print(f"Output root: {cfg.output_root}", flush=True)
        print(f"Variables: {', '.join(cfg.variables)}", flush=True)
        print(f"Scheduled variable-days: {len(tasks)}", flush=True)
        if missing_variables:
            print("Missing variable folders: " + ", ".join(missing_variables), flush=True)

    processed_days = 0
    failed_days = 0
    written_files = 0
    skipped_existing_files = 0
    scanned_files = 0
    small_negative_clipped = 0
    large_negative = 0
    negative_clipped = 0
    failures: list[str] = []

    if cfg.workers > 1 and tasks:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            future_map = {pool.submit(process_day, task, cfg): task for task in tasks}
            total = len(future_map)
            for i, future in enumerate(as_completed(future_map), start=1):
                task = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "status": "failed",
                        "variable": task.variable,
                        "day": task.input_day_dir.name,
                        "error": f"Worker crashed: {exc}",
                    }
                (
                    processed_days,
                    failed_days,
                    scanned_files,
                    written_files,
                    skipped_existing_files,
                    small_negative_clipped,
                    large_negative,
                    negative_clipped,
                ) = _accumulate_result(
                    result,
                    failures,
                    processed_days,
                    failed_days,
                    scanned_files,
                    written_files,
                    skipped_existing_files,
                    small_negative_clipped,
                    large_negative,
                    negative_clipped,
                )
                if not cfg.quiet and (i % 25 == 0 or i == total):
                    print(
                        f"Progress {i}/{total}: processed_days={processed_days}, "
                        f"failed_days={failed_days}, written={written_files}",
                        flush=True,
                    )
    else:
        total = len(tasks)
        for i, task in enumerate(tasks, start=1):
            result = process_day(task, cfg)
            (
                processed_days,
                failed_days,
                scanned_files,
                written_files,
                skipped_existing_files,
                small_negative_clipped,
                large_negative,
                negative_clipped,
            ) = _accumulate_result(
                result,
                failures,
                processed_days,
                failed_days,
                scanned_files,
                written_files,
                skipped_existing_files,
                small_negative_clipped,
                large_negative,
                negative_clipped,
            )
            if not cfg.quiet and (i % 25 == 0 or i == total):
                print(
                    f"Progress {i}/{total}: processed_days={processed_days}, "
                    f"failed_days={failed_days}, written={written_files}",
                    flush=True,
                )

    failures_log = cfg.output_root / "deaccumulation_failures.log"
    if failures:
        failures_log.write_text("\n".join(failures) + "\n")
    else:
        failures_log.write_text("")

    summary = {
        "input_root": str(cfg.input_root),
        "output_root": str(cfg.output_root),
        "variables": list(cfg.variables),
        "missing_variable_folders": missing_variables,
        "scheduled_variable_days": len(tasks),
        "processed_days": processed_days,
        "failed_days": failed_days,
        "scanned_netcdf_files": scanned_files,
        "written_netcdf_files": written_files,
        "skipped_existing_files": skipped_existing_files,
        "small_negative_clipped": small_negative_clipped,
        "large_negative": large_negative,
        "negative_clipped": negative_clipped,
        "negative_tolerance": cfg.negative_tolerance,
        "clip_negative": cfg.clip_negative,
        "strict_hours": cfg.strict_hours,
        "workers": cfg.workers,
        "overwrite": cfg.overwrite,
        "compression": cfg.compress,
        "compression_level": cfg.compress_level,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "failures_log": str(failures_log),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_file = cfg.output_root / "deaccumulation_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if not cfg.quiet:
        print(f"Done. Summary written to {summary_file}", flush=True)

    return summary


def _accumulate_result(
    result: dict[str, object],
    failures: list[str],
    processed_days: int,
    failed_days: int,
    scanned_files: int,
    written_files: int,
    skipped_existing_files: int,
    small_negative_clipped: int,
    large_negative: int,
    negative_clipped: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    if result.get("status") == "ok":
        processed_days += 1
        scanned_files += int(result.get("files", 0))
        written_files += int(result.get("written", 0))
        skipped_existing_files += int(result.get("skipped_existing", 0))
        small_negative_clipped += int(result.get("small_negative_clipped", 0))
        large_negative += int(result.get("large_negative", 0))
        negative_clipped += int(result.get("negative_clipped", 0))
    else:
        failed_days += 1
        failures.append(
            f"{result.get('variable', 'unknown')}/{result.get('day', 'unknown')}: "
            f"{result.get('error', 'unknown error')}"
        )
    return (
        processed_days,
        failed_days,
        scanned_files,
        written_files,
        skipped_existing_files,
        small_negative_clipped,
        large_negative,
        negative_clipped,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert accumulated precipitation NetCDF outputs to hourly "
            "deaccumulated increments."
        )
    )
    parser.add_argument(
        "input_root",
        help="Converted NetCDF root with layout <root>/<VAR>/<pfYYYYMMDD>/<file>.nc",
    )
    parser.add_argument(
        "output_root",
        help="Destination root for deaccumulated NetCDF files with the same layout",
    )
    parser.add_argument(
        "--vars",
        nargs="+",
        default=None,
        help=(
            "Precipitation variables to deaccumulate. Comma-separated values are accepted. "
            f"Default: {' '.join(DEFAULT_PRECIP_VARS)}"
        ),
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker count")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=False)
    parser.add_argument("--compress", choices=["zlib", "none"], default="zlib")
    parser.add_argument("--level", type=int, default=1, help="Compression level for zlib")
    parser.add_argument("--start-date", metavar="YYYYMMDD", help="Process days on/after this date")
    parser.add_argument("--end-date", metavar="YYYYMMDD", help="Process days on/before this date")
    parser.add_argument(
        "--strict-hours",
        dest="strict_hours",
        action="store_true",
        help="Require consecutive hourly lead files (default)",
    )
    parser.add_argument(
        "--allow-gaps",
        dest="strict_hours",
        action="store_false",
        help="Allow gaps and compute increments between available lead files",
    )
    parser.set_defaults(strict_hours=True)
    parser.add_argument(
        "--negative-tolerance",
        type=float,
        default=1.0e-2,
        help="Small negative differences within this tolerance are set to zero (default: 0.01)",
    )
    parser.add_argument(
        "--clip-negative",
        action="store_true",
        help="Also set larger negative deaccumulated values to zero",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = DeaccumConfig(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        variables=_split_vars(args.vars),
        workers=max(1, int(args.workers)),
        overwrite=bool(args.overwrite),
        compress=str(args.compress),
        compress_level=int(args.level),
        start_date=args.start_date,
        end_date=args.end_date,
        strict_hours=bool(args.strict_hours),
        negative_tolerance=float(args.negative_tolerance),
        clip_negative=bool(args.clip_negative),
        quiet=bool(args.quiet),
    )
    run_deaccumulation(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
