"""
Surface-only diurnal diagnostics for masked NetCDF outputs.

Typical use case:
- Convert SURFEX hourly files such as SFX.RN to masked NetCDF first.
- Then aggregate the masked 2D field over the kept lat/lon points.
- Plot the 24-hour diurnal cycle for control / graupel / 2mom.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from alaro_analysis.analysis.surface import (
    compute_lcl_lines_for_periods,
    compute_surface_line,
    infer_lcl_start_height_m,
    is_pblh_variable,
    lcl_cache_analysis_name,
)
from alaro_analysis.common.cli_config import add_config_argument, parse_configured_args
from alaro_analysis.common.constants import EXPERIMENTS, SEASONS
from alaro_analysis.common.naming import safe_name
from alaro_analysis.common.paths import runs_path
from alaro_analysis.common.seasons import build_period_specs, resolve_seasons
from alaro_analysis.common.spatial import build_spatial_window, spatial_window_tag
from alaro_analysis.data.cache import build_cache_file
from alaro_analysis.data.cache_context import CacheContext
from alaro_analysis.data.discovery import (
    collect_file_records,
    discover_variable_maps,
    resolve_var_name,
)
from alaro_analysis.plotting.surface import plot_surface_diurnal

DEFAULT_CONTROL_DIR = runs_path("SURFEX/control/masked-netcdf")
DEFAULT_GRAUPEL_DIR = runs_path("SURFEX/graupel/masked-netcdf")
DEFAULT_2MOM_DIR = runs_path("SURFEX/2mom/masked-netcdf")

DEFAULT_OUTPUT_DIR = runs_path("figures/surface")
DEFAULT_INTERMEDIATE_DIR = runs_path("processed-data/surface")

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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
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

    resolved = {exp: resolve_var_name(variable_maps, exp, (args.variable,)) for exp in EXPERIMENTS}
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
    source_roots = [
        experiment_dirs[exp] / name
        for exp in EXPERIMENTS
        for name in (resolved[exp], *lcl_resolved[exp].values())
        if name
    ]
    cache = CacheContext.from_netcdf("surface", vars(args), source_roots)

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
                if cache.valid(cache_file) and not args.overwrite_intermediate:
                    payload = cache.load(cache_file)
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
                    cache.save(
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
        print(
            f"\n===== Computing {args.variable} for {period.label} ({period.key}) =====", flush=True
        )
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
            if cache.valid(cache_file) and not args.overwrite_intermediate:
                payload = cache.load(cache_file)
                mean = np.asarray(payload["mean"], dtype=np.float64)
            else:
                mean, counts, used = compute_surface_line(
                    records=records,
                    variable_name=variable_name,
                    spatial_window=spatial_window,
                )
                if args.max_days is None:
                    cache.save(
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


def __getattr__(name):
    """Compatibility for moved functions; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    main()
