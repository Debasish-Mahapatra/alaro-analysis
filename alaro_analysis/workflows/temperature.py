"""
Temperature-only diurnal-height panel plots for ALARO masked-netcdf runs.

This script mirrors the main hydrometeor workflow but focuses on TEMPERATURE:
1) Builds full 2-year and seasonal diurnal profiles.
2) Caches intermediate .npz files (resume-friendly by default).
3) Uses GEOPOTENTIEL (meters) for vertical axis.
4) Plots 3 panels:
   (a) C1M absolute temperature
   (b) G1M - C1M anomaly
   (c) G2M - G1M anomaly
5) Saves figures at 450 dpi.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alaro_analysis.common.cli_config import add_config_argument, parse_configured_args
from alaro_analysis.common.constants import EXPERIMENTS, SEASONS
from alaro_analysis.common.models import PeriodSpec, VerticalAxis
from alaro_analysis.common.naming import safe_name
from alaro_analysis.common.parallel import execute_jobs, resolve_workers
from alaro_analysis.common.paths import runs_path
from alaro_analysis.common.seasons import build_period_specs, resolve_seasons
from alaro_analysis.common.vertical import compute_freezing_line_km
from alaro_analysis.data.cache import build_diurnal_cache_file, build_height_cache_file
from alaro_analysis.data.discovery import discover_variables
from alaro_analysis.data.temperature_profiles import (
    align_vertical_shapes,
    load_or_compute_height,
    run_profile_job,
)

DEFAULT_CONTROL_DIR = runs_path("control/masked-netcdf-2")
DEFAULT_GRAUPEL_DIR = runs_path("graupel/masked-netcdf-2")
DEFAULT_2MOM_DIR = runs_path("2mom/masked-netcdf-2")

DEFAULT_CONTROL_GEO_DIR = runs_path("control/masked-netcdf/GEOPOTENTIEL")
DEFAULT_GRAUPEL_GEO_DIR = runs_path("graupel/masked-netcdf/GEOPOTENTIEL")
DEFAULT_2MOM_GEO_DIR = runs_path("2mom/masked-netcdf/GEOPOTENTIEL")

DEFAULT_OUTPUT_DIR = runs_path("figures")
DEFAULT_INTERMEDIATE_DIR = runs_path("processed-data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full+seasonal temperature diurnal panels from masked-netcdf-2."
    )
    add_config_argument(parser)
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL_DIR)
    parser.add_argument("--graupel-dir", type=Path, default=DEFAULT_GRAUPEL_DIR)
    parser.add_argument("--twomom-dir", type=Path, default=DEFAULT_2MOM_DIR)

    parser.add_argument("--control-geopotential-dir", type=Path, default=DEFAULT_CONTROL_GEO_DIR)
    parser.add_argument("--graupel-geopotential-dir", type=Path, default=DEFAULT_GRAUPEL_GEO_DIR)
    parser.add_argument("--twomom-geopotential-dir", type=Path, default=DEFAULT_2MOM_GEO_DIR)

    parser.add_argument(
        "--temperature-variable",
        default="TEMPERATURE",
        help="Temperature variable name (default: TEMPERATURE).",
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
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=DEFAULT_INTERMEDIATE_DIR,
        help="Directory for cached/intermediate .npz outputs.",
    )
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--overwrite-intermediate",
        action="store_true",
        help="Overwrite existing intermediate .npz files.",
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
        "--height-variable",
        default="GEOPOTENTIEL",
        help="Variable name used inside geopotential files.",
    )
    parser.add_argument(
        "--height-aggregate",
        choices=("first", "mean-all"),
        default="first",
        help="How to aggregate geopotential profiles by period.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=32,
        help="Multiprocessing worker count (capped to 32).",
    )
    parser.add_argument(
        "--max-height-km",
        type=float,
        default=20.0,
        help="Upper y-limit for height-based plots.",
    )
    parser.add_argument(
        "--anomaly-percentile",
        type=float,
        default=98.0,
        help="Percentile for fixed symmetric anomaly scale (default: 98).",
    )
    parser.add_argument(
        "--absolute-low-percentile",
        type=float,
        default=2.0,
        help="Lower percentile for fixed absolute colorbar range (default: 2).",
    )
    parser.add_argument(
        "--absolute-high-percentile",
        type=float,
        default=98.0,
        help="Upper percentile for fixed absolute colorbar range (default: 98).",
    )
    return parse_configured_args(parser, "temperature", argv=argv)


@dataclass(frozen=True)
class TemperatureRun:
    """Resolved source locations and settings for one command invocation."""

    args: argparse.Namespace
    experiment_dirs: dict[str, Path]
    geopotential_dirs: dict[str, Path]
    periods: list[PeriodSpec]
    output_dir: Path
    intermediate_dir: Path
    n_workers: int


def _prepare_run(args: argparse.Namespace):
    """Validate inputs and resolve the sources used by every subsequent stage."""
    if not 0.0 < args.anomaly_percentile <= 100.0:
        raise ValueError("--anomaly-percentile must be in (0, 100].")
    if not 0.0 <= args.absolute_low_percentile < 100.0:
        raise ValueError("--absolute-low-percentile must be in [0, 100).")
    if not 0.0 < args.absolute_high_percentile <= 100.0:
        raise ValueError("--absolute-high-percentile must be in (0, 100].")
    if args.absolute_low_percentile >= args.absolute_high_percentile:
        raise ValueError("--absolute-low-percentile must be lower than --absolute-high-percentile.")
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
            print(f"[warn] {exp} geopotential dir not found: {d}")
    available = discover_variables(experiment_dirs)
    common = set.intersection(*(available[exp] for exp in EXPERIMENTS))
    if args.temperature_variable not in common:
        raise RuntimeError(
            f"Temperature variable '{args.temperature_variable}' is not common across experiments. Common: {sorted(common)}"
        )
    selected_seasons = resolve_seasons(args.seasons)
    periods = build_period_specs(set(args.analysis_modes), selected_seasons)
    if not periods:
        raise RuntimeError("No analysis periods selected.")
    output_dir = args.output_dir.resolve()
    intermediate_dir = args.intermediate_dir.resolve()
    n_workers = resolve_workers(args.n_workers)
    print("\nInput data directories:")
    for exp in EXPERIMENTS:
        print(f"- {exp}: {experiment_dirs[exp]}")
    print("\nGeopotential directories:")
    for exp in EXPERIMENTS:
        print(f"- {exp}: {geopotential_dirs[exp]}")
    print("\nOutput directory:", output_dir)
    print("Intermediate directory:", intermediate_dir)
    print("Workers:", n_workers)
    print("Periods:", [p.key for p in periods])
    print("Overwrite intermediate:", args.overwrite_intermediate)
    print("Ignoring +0024 files by design.")
    print("Variable to plot:", args.temperature_variable)
    return TemperatureRun(
        args=args,
        experiment_dirs=experiment_dirs,
        geopotential_dirs=geopotential_dirs,
        periods=periods,
        output_dir=output_dir,
        intermediate_dir=intermediate_dir,
        n_workers=n_workers,
    )


def _build_jobs(plan: TemperatureRun):
    """Build one job per experiment and requested period."""
    jobs: list[tuple[str, str, str, str, str, tuple[int, ...] | None]] = []
    for period in plan.periods:
        for exp in EXPERIMENTS:
            cache_file = build_diurnal_cache_file(
                intermediate_dir=plan.intermediate_dir,
                variable=plan.args.temperature_variable,
                period_subdir=period.output_subdir,
                experiment=exp,
            )
            jobs.append(
                (
                    period.key,
                    exp,
                    str(cache_file),
                    str(plan.experiment_dirs[exp]),
                    plan.args.temperature_variable,
                    period.allowed_months,
                )
            )
    print(f"\nComputing {len(jobs)} profile jobs with up to {plan.n_workers} processes...")
    return jobs


def _compute_profiles(plan: TemperatureRun, jobs):
    """Execute profile jobs and collect failures explicitly."""
    tasks = [
        (
            (period, exp),
            (
                cache,
                root,
                variable,
                plan.args.max_days,
                plan.args.overwrite_intermediate,
                months,
                plan.args.utc_offset_hours,
            ),
        )
        for period, exp, cache, root, variable, months in jobs
    ]
    results, errors = {}, {}
    for key, value, error in execute_jobs(run_profile_job, tasks, workers=plan.n_workers):
        if error is not None:
            errors[key] = error
        else:
            mean, sample_file = value
            results[key] = (mean, Path(sample_file))
    return results, errors


def _choose_scales(plan: TemperatureRun, results):
    """Choose common color scales across every requested period."""
    from alaro_analysis.plotting.temperature import (
        compute_temperature_scales,
    )

    scale_data: dict[tuple[str, str], np.ndarray] = {}
    for period in plan.periods:
        for exp in EXPERIMENTS:
            key = (period.key, exp)
            if key in results:
                scale_data[key] = results[key][0]
    abs_limits, anom_scale = compute_temperature_scales(
        periods=plan.periods,
        results=scale_data,
        low_pct=plan.args.absolute_low_percentile,
        high_pct=plan.args.absolute_high_percentile,
        anom_pct=plan.args.anomaly_percentile,
    )
    print(
        f"\nFixed temperature scales: abs=[{abs_limits[0]:.3f}, {abs_limits[1]:.3f}], anom=±{anom_scale:.3f}"
    )
    return (abs_limits, anom_scale)


def _render_results(plan: TemperatureRun, results, errors, abs_limits, anom_scale):
    """Render each complete comparison from its computed profiles."""
    from alaro_analysis.plotting.temperature import (
        plot_temperature_panels,
    )

    total_saved = 0
    for period in plan.periods:
        print(f"\n==================== {period.label} ({period.key}) ====================")
        profiles: dict[str, np.ndarray] = {}
        for exp in EXPERIMENTS:
            key = (period.key, exp)
            if key not in results:
                print(f"Skipping {period.key}: {exp} unavailable ({errors.get(key, 'missing')})")
                profiles = {}
                break
            profiles[exp] = results[key][0]
        if not profiles:
            continue
        height_cache = build_height_cache_file(
            intermediate_dir=plan.intermediate_dir,
            period_subdir=period.output_subdir,
            experiment="control",
            aggregate=plan.args.height_aggregate,
        )
        height_m = load_or_compute_height(
            cache_file=height_cache,
            geopotential_dir=plan.geopotential_dirs["control"],
            height_variable=plan.args.height_variable,
            max_days=plan.args.max_days,
            overwrite_intermediate=plan.args.overwrite_intermediate,
            allowed_months=period.allowed_months,
            utc_offset_hours=plan.args.utc_offset_hours,
            aggregate=plan.args.height_aggregate,
        )
        axis = VerticalAxis(
            values=np.asarray(height_m, dtype=np.float64) / 1000.0,
            label="Height (km)",
            is_height_km=True,
        )
        axis, profiles = align_vertical_shapes(axis=axis, profiles=profiles)
        freezing_line_km = compute_freezing_line_km(
            axis=axis, temperature_profiles=[profiles[exp] for exp in EXPERIMENTS]
        )
        output_file = (
            plan.output_dir
            / safe_name(plan.args.temperature_variable)
            / period.output_subdir
            / f"{safe_name(plan.args.temperature_variable)}_panel_c1m_g1m-c1m_g2m-g1m.png"
        )
        plot_temperature_panels(
            control_mean=profiles["control"],
            graupel_mean=profiles["graupel"],
            twomom_mean=profiles["2mom"],
            axis=axis,
            output_file=output_file,
            max_height_km=plan.args.max_height_km,
            period_label=period.label,
            variable=plan.args.temperature_variable,
            freezing_line_km=freezing_line_km,
            abs_limits=abs_limits,
            anom_scale=anom_scale,
        )
        total_saved += 1
    print(f"\nCompleted. Total figures generated: {total_saved}")
    if errors:
        print(f"Completed with {len(errors)} profile-job errors.")


def main(argv: list[str] | None = None) -> None:
    plan = _prepare_run(parse_args(argv))
    if plan is None:
        return
    jobs = _build_jobs(plan)
    results, errors = _compute_profiles(plan, jobs)
    abs_limits, anom_scale = _choose_scales(plan, results)
    _render_results(plan, results, errors, abs_limits, anom_scale)
    if errors:
        details = "; ".join(f"{key}: {message}" for key, message in errors.items())
        raise RuntimeError(f"{len(errors)} profile job(s) failed: {details}")


def __getattr__(name):
    """Compatibility for pre-refactor imports; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    main()
