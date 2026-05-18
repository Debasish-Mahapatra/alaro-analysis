#!/usr/bin/env python3
"""
Surface-only diurnal diagnostics for masked NetCDF outputs.

Typical use case:
- Convert SURFEX hourly files such as SFX.RN to masked NetCDF first.
- Then aggregate the masked 2D field over the kept lat/lon points.
- Plot the 24-hour diurnal cycle for control / graupel / 2mom.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from alaro_analysis.common.constants import CP_D, EXPERIMENTS, EXPERIMENT_LABELS, G, SEASONS
from alaro_analysis.common.cli_config import add_config_argument, parse_configured_args
from alaro_analysis.common.naming import safe_name
from alaro_analysis.common.seasons import build_period_specs, resolve_seasons
from alaro_analysis.common.spatial import build_spatial_window, spatial_window_tag
from alaro_analysis.common.timeparse import has_pf_subdirs, parse_month_from_day_name
from alaro_analysis.data.cache import build_cache_file, load_cache, save_cache
from alaro_analysis.data.dataset_io import read_time_level_yx
from alaro_analysis.data.discovery import collect_file_records

DEFAULT_CONTROL_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/SURFEX/control/masked-netcdf"
)
DEFAULT_GRAUPEL_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/SURFEX/graupel/masked-netcdf"
)
DEFAULT_2MOM_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/SURFEX/2mom/masked-netcdf"
)

DEFAULT_OUTPUT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/surface")
DEFAULT_INTERMEDIATE_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/surface"
)

VAR_TOKEN_RE = re.compile(r"[^A-Za-z0-9]+")
HEIGHT_LEVEL_RE = re.compile(r"^H(\d{5})")
PBLH_VAR_TOKEN = "CLPMHAUTMODXFU"
DEFAULT_LCL_TEMPERATURE_VARS = (
    "H00100TEMPERATUR",
    "SURFTEMPERATURE",
    "CLSTEMPERATURE",
)
DEFAULT_LCL_SPECIFIC_HUMIDITY_VARS = ("H00100HUMI.SPECI", "CLSHUMI.SPECIFIQ")
DEFAULT_LCL_PRESSURE_VARS = ("H00100PRESSURE", "SURFPRESSION")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot diurnal cycles for masked 2D surface fields such as SURFEX SFX.RN."
    )
    add_config_argument(parser)
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL_DIR)
    parser.add_argument("--graupel-dir", type=Path, default=DEFAULT_GRAUPEL_DIR)
    parser.add_argument("--twomom-dir", type=Path, default=DEFAULT_2MOM_DIR)
    parser.add_argument(
        "--variable",
        default="SFX.RN",
        help="Masked NetCDF variable directory/name to analyze (default: SFX.RN).",
    )
    parser.add_argument(
        "--variable-label",
        default="Net radiation",
        help="Plot label for the selected variable.",
    )
    parser.add_argument(
        "--variable-unit",
        default="W m-2",
        help="Unit label for the selected variable.",
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
    parser.add_argument(
        "--lcl-overlay",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Overlay MetPy lifted-condensation-level height. "
            "'auto' enables it for CLPMHAUT.MOD.XFU only."
        ),
    )
    parser.add_argument(
        "--lcl-temperature-vars",
        nargs="+",
        default=list(DEFAULT_LCL_TEMPERATURE_VARS),
        help="Candidate converted variables for the LCL parcel temperature.",
    )
    parser.add_argument(
        "--lcl-specific-humidity-vars",
        nargs="+",
        default=list(DEFAULT_LCL_SPECIFIC_HUMIDITY_VARS),
        help="Candidate converted variables for the LCL parcel specific humidity.",
    )
    parser.add_argument(
        "--lcl-pressure-vars",
        nargs="+",
        default=list(DEFAULT_LCL_PRESSURE_VARS),
        help="Candidate converted variables for the LCL parcel pressure.",
    )
    parser.add_argument(
        "--lcl-start-height-m",
        type=float,
        default=None,
        help=(
            "Parcel starting height in metres AGL. By default this is inferred "
            "from Hxxxxx input variables, e.g. H00100 -> 100 m."
        ),
    )
    return parse_configured_args(parser, "surface", argv=argv)


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


def plot_surface_diurnal(
    *,
    variable_label: str,
    variable_unit: str,
    period_label: str,
    line_data: dict[str, np.ndarray],
    lcl_data: dict[str, np.ndarray] | None,
    output_file: Path,
    utc_offset_hours: int,
) -> None:
    colors = {
        "control": "#d62728",
        "graupel": "#1f77b4",
        "2mom": "#2ca02c",
    }
    hours = np.arange(24, dtype=np.float64)
    ylabel = variable_label
    if variable_unit:
        ylabel = f"{ylabel} [{variable_unit}]"
    has_lcl = lcl_data is not None and any(
        lcl_data.get(exp) is not None and np.isfinite(lcl_data[exp]).any()
        for exp in EXPERIMENTS
    )
    if has_lcl and variable_unit:
        ylabel = f"{variable_label} and LCL [{variable_unit}]"

    def draw_lines(ax: plt.Axes, *, configure_main_axis: bool) -> None:
        for exp in EXPERIMENTS:
            arr = line_data.get(exp)
            if arr is None:
                continue
            ax.plot(
                hours,
                arr,
                linewidth=2.4,
                marker="o",
                markersize=3.5,
                color=colors[exp],
                label=f"{EXPERIMENT_LABELS[exp]} PBLH" if has_lcl else EXPERIMENT_LABELS[exp],
            )
            if has_lcl and lcl_data is not None:
                lcl_arr = lcl_data.get(exp)
                if lcl_arr is not None:
                    ax.plot(
                        hours,
                        lcl_arr,
                        linewidth=2.0,
                        linestyle="--",
                        marker="s",
                        markersize=3.0,
                        color=colors[exp],
                        alpha=0.85,
                        label=f"{EXPERIMENT_LABELS[exp]} LCL",
                    )
        ax.grid(alpha=0.25, linestyle="--")
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        if configure_main_axis:
            ax.set_xticks(np.arange(0, 24, 3))
            ax.set_xlim(0.0, 23.0)

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    draw_lines(ax, configure_main_axis=True)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(f"Hour (local UTC{utc_offset_hours:+d})", fontsize=12)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    title_label = f"{variable_label} and LCL" if has_lcl else variable_label
    ax.set_title(f"{period_label} - {title_label} diurnal cycle", fontsize=14, fontweight="bold")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def main() -> None:
    args = parse_args()
    spatial_window = build_spatial_window(args.y_slice, args.x_slice)
    spatial_tag = spatial_window_tag(spatial_window)

    experiment_dirs = {
        "control": args.control_dir.resolve(),
        "graupel": args.graupel_dir.resolve(),
        "2mom": args.twomom_dir.resolve(),
    }
    for exp, d in experiment_dirs.items():
        if not d.exists():
            raise FileNotFoundError(f"{exp} data dir not found: {d}")

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

    resolved = {
        exp: resolve_var_name(variable_maps, exp, (args.variable,))
        for exp in EXPERIMENTS
    }
    lcl_enabled = args.lcl_overlay == "on" or (
        args.lcl_overlay == "auto" and is_pblh_variable(args.variable)
    )
    lcl_resolved: dict[str, dict[str, str | None]] = {
        exp: {
            "temperature": resolve_var_name(variable_maps, exp, args.lcl_temperature_vars),
            "specific_humidity": resolve_var_name(
                variable_maps, exp, args.lcl_specific_humidity_vars
            ),
            "pressure": resolve_var_name(variable_maps, exp, args.lcl_pressure_vars),
        }
        for exp in EXPERIMENTS
    }

    print("\nInput data directories:", flush=True)
    for exp in EXPERIMENTS:
        print(f"- {exp}: {experiment_dirs[exp]}", flush=True)
    print("\nResolved variable names:", flush=True)
    for exp in EXPERIMENTS:
        print(f"- {exp}: {resolved[exp]}", flush=True)
    if lcl_enabled:
        print("\nResolved LCL input variables:", flush=True)
        for exp in EXPERIMENTS:
            names = lcl_resolved[exp]
            print(
                f"- {exp}: T={names['temperature']}, "
                f"q={names['specific_humidity']}, p={names['pressure']}",
                flush=True,
            )
    print("\nOutput directory:", args.output_dir.resolve(), flush=True)
    print("Intermediate directory:", args.intermediate_dir.resolve(), flush=True)
    print("Periods:", [p.key for p in periods], flush=True)
    print("Spatial averaging tag:", spatial_tag, flush=True)
    print(
        "Spatial Y slice:",
        f"{spatial_window.y_start}:{spatial_window.y_end}",
        "| X slice:",
        f"{spatial_window.x_start}:{spatial_window.x_end}",
        flush=True,
    )
    print("Ignoring +0024 files by design.", flush=True)

    output_dir = args.output_dir.resolve()
    intermediate_dir = args.intermediate_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    lcl_lines_by_period_exp: dict[tuple[str, str], np.ndarray] = {}
    if lcl_enabled:
        for exp in EXPERIMENTS:
            variable_name = resolved[exp]
            lcl_inputs = lcl_resolved[exp]
            temperature_name = lcl_inputs["temperature"]
            specific_humidity_name = lcl_inputs["specific_humidity"]
            pressure_name = lcl_inputs["pressure"]
            if variable_name is None:
                continue
            if not (temperature_name and specific_humidity_name and pressure_name):
                print(
                    f"[warn] {exp}: LCL inputs not found; convert H00100TEMPERATUR, "
                    "H00100HUMI.SPECI, and H00100PRESSURE.",
                    flush=True,
                )
                continue

            start_height_m = infer_lcl_start_height_m(
                (temperature_name, specific_humidity_name, pressure_name),
                args.lcl_start_height_m,
            )
            analysis_name = lcl_cache_analysis_name(
                args.variable,
                temperature_name,
                specific_humidity_name,
                pressure_name,
                start_height_m,
            )
            missing_periods = []
            cache_files = {}
            for period in periods:
                cache_file = build_cache_file(
                    intermediate_dir=intermediate_dir,
                    analysis_name=analysis_name,
                    period_subdir=period.output_subdir,
                    experiment=exp,
                    spatial_tag=spatial_tag,
                )
                cache_files[period.key] = cache_file
                if cache_file.exists() and not args.overwrite_intermediate:
                    payload = load_cache(cache_file)
                    lcl_lines_by_period_exp[(period.key, exp)] = np.asarray(
                        payload["mean"], dtype=np.float64
                    )
                else:
                    missing_periods.append(period)

            if not missing_periods:
                continue

            variable_dir = experiment_dirs[exp] / variable_name
            all_records = collect_file_records(
                variable_dir=variable_dir,
                max_days=args.max_days,
                allowed_months=None,
                utc_offset_hours=args.utc_offset_hours,
            )
            if not all_records:
                print(f"[warn] {exp}: no records found in {variable_dir}", flush=True)
                continue

            print(
                f"\n===== Computing LCL for {exp} across "
                f"{len(missing_periods)} period(s) in one pass =====",
                flush=True,
            )
            computed = compute_lcl_lines_for_periods(
                records=all_records,
                period_specs=missing_periods,
                experiment_dir=experiment_dirs[exp],
                temperature_name=temperature_name,
                specific_humidity_name=specific_humidity_name,
                pressure_name=pressure_name,
                spatial_window=spatial_window,
                start_height_m=start_height_m,
            )
            for period in missing_periods:
                lcl_mean, lcl_counts, lcl_used = computed[period.key]
                if args.max_days is None:
                    save_cache(
                        cache_files[period.key],
                        {
                            "mean": lcl_mean,
                            "counts": lcl_counts,
                            "n_files": np.array([lcl_used], dtype=np.int64),
                            "start_height_m": np.array([start_height_m], dtype=np.float64),
                        },
                    )
                if np.isfinite(lcl_mean).any():
                    lcl_lines_by_period_exp[(period.key, exp)] = lcl_mean
                else:
                    print(f"[warn] {period.key}/{exp}: no finite LCL values.", flush=True)

    for period in periods:
        print(f"\n===== Computing {args.variable} for {period.label} ({period.key}) =====", flush=True)
        lines_by_exp: dict[str, np.ndarray] = {}
        lcl_by_exp: dict[str, np.ndarray] = {}

        for exp in EXPERIMENTS:
            variable_name = resolved[exp]
            if variable_name is None:
                print(
                    f"[warn] {period.key}/{exp}: variable '{args.variable}' not found.",
                    flush=True,
                )
                continue

            variable_dir = experiment_dirs[exp] / variable_name
            records = collect_file_records(
                variable_dir=variable_dir,
                max_days=args.max_days,
                allowed_months=period.allowed_months,
                utc_offset_hours=args.utc_offset_hours,
            )
            if not records:
                print(f"[warn] {period.key}/{exp}: no records found in {variable_dir}", flush=True)
                continue

            cache_file = build_cache_file(
                intermediate_dir=intermediate_dir,
                analysis_name=f"{args.variable}_surface_diurnal",
                period_subdir=period.output_subdir,
                experiment=exp,
                spatial_tag=spatial_tag,
            )
            if cache_file.exists() and not args.overwrite_intermediate:
                payload = load_cache(cache_file)
                mean = np.asarray(payload["mean"], dtype=np.float64)
            else:
                mean, counts, used = compute_surface_line(
                    records=records,
                    variable_name=variable_name,
                    spatial_window=spatial_window,
                )
                if args.max_days is None:
                    save_cache(
                        cache_file,
                        {
                            "mean": mean,
                            "counts": counts,
                            "n_files": np.array([used], dtype=np.int64),
                        },
                    )
            lines_by_exp[exp] = mean

            if lcl_enabled:
                lcl_mean = lcl_lines_by_period_exp.get((period.key, exp))
                if lcl_mean is not None and np.isfinite(lcl_mean).any():
                    lcl_by_exp[exp] = lcl_mean

        if not lines_by_exp:
            print(f"[warn] {period.key}: no experiments produced a line.", flush=True)
            continue

        output_file = (
            output_dir
            / safe_name(args.variable)
            / period.output_subdir
            / (
                f"{safe_name(args.variable)}_diurnal_cycle"
                + (f"_{spatial_tag}" if spatial_tag != "full-domain" else "")
                + ".png"
            )
        )
        plot_surface_diurnal(
            variable_label=args.variable_label,
            variable_unit=args.variable_unit,
            period_label=period.label,
            line_data=lines_by_exp,
            lcl_data=lcl_by_exp if lcl_by_exp else None,
            output_file=output_file,
            utc_offset_hours=args.utc_offset_hours,
        )

    print("\nCompleted surface diurnal diagnostics.", flush=True)


if __name__ == "__main__":
    main()
