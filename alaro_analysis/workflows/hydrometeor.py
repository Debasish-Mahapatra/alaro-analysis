"""
Hydrometeor diurnal-height 3-panel plots for ALARO masked-netcdf-2 runs.

What this script does:
1) Builds full 2-year and seasonal diurnal profiles for each variable.
2) Saves figures with structure:
     figures/<variable>/2years/*.png
     figures/<variable>/seasonal/<season>/*.png
3) Saves intermediate data (.npz) with matching structure:
     processed-data/<variable>/2years/*.npz
     processed-data/<variable>/seasonal/<season>/*.npz
4) Uses up to 32 processes for profile computation.
5) Uses GEOPOTENTIEL (meters) for vertical axis and overlays a freezing-level
   line derived from TEMPERATURE (not as separate temperature plots).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alaro_analysis.common.cli_config import add_config_argument, parse_configured_args
from alaro_analysis.common.constants import EXPERIMENTS, SEASONS
from alaro_analysis.common.models import PeriodSpec, SpatialWindow, VerticalAxis
from alaro_analysis.common.naming import safe_name
from alaro_analysis.common.parallel import execute_jobs, resolve_workers
from alaro_analysis.common.paths import runs_path
from alaro_analysis.common.seasons import build_period_specs, resolve_seasons
from alaro_analysis.common.spatial import build_spatial_window, spatial_window_tag
from alaro_analysis.common.vertical import (
    compute_freezing_line_km,
    interpolate_profile_to_target_height,
)
from alaro_analysis.data.cache import build_diurnal_cache_file
from alaro_analysis.data.discovery import discover_variables
from alaro_analysis.data.hydrometeor_profiles import (
    align_axis_and_profile,
    align_vertical_shapes,
    build_vertical_axis,
    run_profile_job,
)
from alaro_analysis.data.profile_sources import (
    UPDRAFT_DERIVED_VARIABLES,
    UPDRAFT_MESH_VAR,
    UPDRAFT_OMEGA_VAR,
    build_experiment_sources,
    discover_variable_roots,
    discover_variables_from_roots,
    is_updraft_derived_variable,
    resolve_plot_variables,
    resolve_temperature_variable,
    resolve_updraft_source_dirs,
    temperature_cache_available,
    updraft_derived_available,
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
        description="Build full+seasonal hydrometeor diurnal panels from masked-netcdf-2."
    )
    add_config_argument(parser)
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL_DIR)
    parser.add_argument("--graupel-dir", type=Path, default=DEFAULT_GRAUPEL_DIR)
    parser.add_argument("--twomom-dir", type=Path, default=DEFAULT_2MOM_DIR)
    parser.add_argument(
        "--control-dir-2",
        type=Path,
        default=None,
        help="Optional secondary control source directory (used for missing vars).",
    )
    parser.add_argument(
        "--graupel-dir-2",
        type=Path,
        default=None,
        help="Optional secondary graupel source directory (used for missing vars).",
    )
    parser.add_argument(
        "--twomom-dir-2",
        type=Path,
        default=None,
        help="Optional secondary 2mom source directory (used for missing vars).",
    )

    parser.add_argument("--control-geopotential-dir", type=Path, default=DEFAULT_CONTROL_GEO_DIR)
    parser.add_argument("--graupel-geopotential-dir", type=Path, default=DEFAULT_GRAUPEL_GEO_DIR)
    parser.add_argument("--twomom-geopotential-dir", type=Path, default=DEFAULT_2MOM_GEO_DIR)

    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help=(
            "Variables to plot. Default: common variables across all experiments, "
            "excluding TEMPERATURE. Supports derived updraft variables: "
            "UPDRAFT_FLUX, UPDRAFT_EXTENT, UPDRAFT_INTENSITY."
        ),
    )
    parser.add_argument(
        "--temperature-variable",
        default="TEMPERATURE",
        help="Variable used to derive freezing-level line (default: TEMPERATURE).",
    )
    parser.add_argument(
        "--temperature-control-dir",
        type=Path,
        default=None,
        help="Optional control directory for temperature source; defaults to --control-dir.",
    )
    parser.add_argument(
        "--temperature-graupel-dir",
        type=Path,
        default=None,
        help="Optional graupel directory for temperature source; defaults to --graupel-dir.",
    )
    parser.add_argument(
        "--temperature-twomom-dir",
        type=Path,
        default=None,
        help="Optional 2mom directory for temperature source; defaults to --twomom-dir.",
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
    parser.add_argument(
        "--temperature-intermediate-dir",
        type=Path,
        default=None,
        help=(
            "Optional intermediate-cache root for temperature profiles. "
            "Defaults to --intermediate-dir."
        ),
    )

    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--overwrite-intermediate",
        action="store_true",
        help="Overwrite existing intermediate .npz files and recompute from raw data.",
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
        "--height-mode",
        choices=("geopotential", "auto", "coord", "index"),
        default="geopotential",
        help="Vertical axis source. Default: geopotential.",
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
    parser.add_argument(
        "--disable-default-shared-groups",
        action="store_true",
        help=(
            "Disable default shared-scale groups "
            "(LIQUID_WATER+RAD_LIQUID_W, SOLID_WATER+RAD_SOLID_WA)."
        ),
    )
    parser.add_argument(
        "--list-variables",
        action="store_true",
        help="List common variables and exit.",
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
    return parse_configured_args(parser, "hydrometeor", argv=argv)


def _figure_subdir(period: PeriodSpec) -> Path:
    """Figure output subdir for a period.

    Seasonal figures share a single flat `seasonal/` directory per variable
    (season name goes into the filename instead).  All other periods use
    their native output_subdir.
    """
    parts = period.output_subdir.parts
    if parts and parts[0] == "seasonal" and len(parts) > 1:
        return Path("seasonal")
    return period.output_subdir


def _figure_filename(variable: str, period: PeriodSpec) -> str:
    base = safe_name(variable)
    parts = period.output_subdir.parts
    if parts and parts[0] == "seasonal" and len(parts) > 1:
        return f"{base}_panel_{period.key}_c1m_g1m-c1m_g2m-g1m.png"
    return f"{base}_panel_c1m_g1m-c1m_g2m-g1m.png"


def ensure_output_tree(
    output_dir: Path,
    intermediate_dir: Path,
    variables: list[str],
    periods: list[PeriodSpec],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    for variable in variables:
        for period in periods:
            (output_dir / safe_name(variable) / _figure_subdir(period)).mkdir(
                parents=True, exist_ok=True
            )
            (intermediate_dir / safe_name(variable) / period.output_subdir).mkdir(
                parents=True, exist_ok=True
            )

    for period in periods:
        (intermediate_dir / "geopotential" / period.output_subdir).mkdir(
            parents=True, exist_ok=True
        )


@dataclass(frozen=True)
class HydrometeorRun:
    """Resolved source locations and settings for one command invocation."""

    args: argparse.Namespace
    spatial_window: SpatialWindow
    spatial_tag: str
    geopotential_dirs: dict[str, Path]
    periods: list[PeriodSpec]
    output_dir: Path
    intermediate_dir: Path
    temperature_intermediate_dir: Path
    n_workers: int
    variable_roots_by_experiment: dict[str, dict[str, Path]]
    updraft_source_dirs: dict[str, Path] | None
    plot_variables: list[str]
    temperature_source_dirs: dict[str, Path]
    temperature_variable: str | None


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
    spatial_window = build_spatial_window(args.y_slice, args.x_slice)
    spatial_tag = spatial_window_tag(spatial_window)
    experiment_primary_dirs = {
        "control": args.control_dir.resolve(),
        "graupel": args.graupel_dir.resolve(),
        "2mom": args.twomom_dir.resolve(),
    }
    experiment_secondary_dirs: dict[str, Path | None] = {
        "control": args.control_dir_2,
        "graupel": args.graupel_dir_2,
        "2mom": args.twomom_dir_2,
    }
    experiment_sources = build_experiment_sources(
        primary_dirs=experiment_primary_dirs, secondary_dirs=experiment_secondary_dirs
    )
    geopotential_dirs = {
        "control": args.control_geopotential_dir.resolve(),
        "graupel": args.graupel_geopotential_dir.resolve(),
        "2mom": args.twomom_geopotential_dir.resolve(),
    }
    for exp, source_dirs in experiment_sources.items():
        for source_dir in source_dirs:
            if not source_dir.exists():
                raise FileNotFoundError(f"{exp} data dir not found: {source_dir}")
    for exp, geo_dir in geopotential_dirs.items():
        if not geo_dir.exists():
            print(f"[warn] {exp} geopotential dir not found: {geo_dir}")
    selected_seasons = resolve_seasons(args.seasons)
    mode_set = set(args.analysis_modes)
    periods = build_period_specs(mode_set, selected_seasons)
    if not periods:
        raise RuntimeError("No analysis periods selected.")
    output_dir = args.output_dir.resolve()
    intermediate_dir = args.intermediate_dir.resolve()
    temperature_intermediate_dir = (
        args.temperature_intermediate_dir.resolve()
        if args.temperature_intermediate_dir is not None
        else intermediate_dir
    )
    n_workers = resolve_workers(args.n_workers)
    variable_roots_by_experiment = discover_variable_roots(experiment_sources)
    available = discover_variables_from_roots(variable_roots_by_experiment)
    updraft_source_dirs = resolve_updraft_source_dirs(experiment_sources)
    derived_available = updraft_derived_available(
        available_by_experiment=available, updraft_source_dirs=updraft_source_dirs
    )
    plot_variables = resolve_plot_variables(
        available_by_experiment=available,
        requested=args.variables,
        temperature_var=args.temperature_variable,
        derived_available_override=derived_available,
    )
    temperature_source_dirs = {
        "control": args.temperature_control_dir.resolve()
        if args.temperature_control_dir is not None
        else experiment_primary_dirs["control"],
        "graupel": args.temperature_graupel_dir.resolve()
        if args.temperature_graupel_dir is not None
        else experiment_primary_dirs["graupel"],
        "2mom": args.temperature_twomom_dir.resolve()
        if args.temperature_twomom_dir is not None
        else experiment_primary_dirs["2mom"],
    }
    temp_dirs_ok = all((p.exists() for p in temperature_source_dirs.values()))
    temperature_variable: str | None = None
    temperature_from_cache_only = False
    temp_available_common: list[str] = []
    if temp_dirs_ok:
        temp_available = discover_variables(temperature_source_dirs)
        temp_available_common = sorted(
            set.intersection(*(temp_available[exp] for exp in EXPERIMENTS))
        )
        temperature_variable = resolve_temperature_variable(
            available_by_experiment=temp_available,
            requested_temperature_var=args.temperature_variable,
        )
    if temperature_variable is None:
        cache_hit = temperature_cache_available(
            temperature_intermediate_dir=temperature_intermediate_dir,
            temperature_variable=args.temperature_variable,
            periods=periods,
            spatial_tag=spatial_tag,
        )
        if cache_hit:
            temperature_variable = args.temperature_variable
            temperature_from_cache_only = True
    ensure_output_tree(
        output_dir=output_dir,
        intermediate_dir=intermediate_dir,
        variables=plot_variables,
        periods=periods,
    )
    common_vars = sorted(set.intersection(*(available[exp] for exp in EXPERIMENTS)))
    print("\nInput data directories:")
    for exp in EXPERIMENTS:
        print(f"- {exp}:")
        for idx, source_dir in enumerate(experiment_sources[exp], start=1):
            suffix = " (primary)" if idx == 1 else f" (secondary #{idx - 1})"
            print(f"    - {source_dir}{suffix}")
    print("\nGeopotential directories:")
    for exp in EXPERIMENTS:
        print(f"- {exp}: {geopotential_dirs[exp]}")
    print("\nOutput directory:", output_dir)
    print("Intermediate directory:", intermediate_dir)
    print("Temperature intermediate directory:", temperature_intermediate_dir)
    print("Spatial averaging tag:", spatial_tag)
    print(
        "Spatial Y slice:",
        f"{spatial_window.y_start}:{spatial_window.y_end}",
        "| X slice:",
        f"{spatial_window.x_start}:{spatial_window.x_end}",
    )
    print("Workers:", n_workers)
    print("Analysis modes:", sorted(mode_set))
    print("Periods:", [p.key for p in periods])
    print("Overwrite intermediate:", args.overwrite_intermediate)
    print(
        f"Global scales: anomaly p{args.anomaly_percentile:g}, absolute p{args.absolute_low_percentile:g}-p{args.absolute_high_percentile:g}"
    )
    print("Default shared groups enabled:", not args.disable_default_shared_groups)
    print("Ignoring +0024 files by design.")
    print("\nCommon variables:", ", ".join(common_vars) if common_vars else "(none)")
    if derived_available:
        print("Derived updraft variables available:", ", ".join(UPDRAFT_DERIVED_VARIABLES))
        if updraft_source_dirs is not None:
            print("Derived updraft source roots:")
            for exp in EXPERIMENTS:
                print(f"    - {exp}: {updraft_source_dirs[exp]}")
    else:
        print(
            f"[info] Derived updraft variables disabled until both {UPDRAFT_OMEGA_VAR} and {UPDRAFT_MESH_VAR} exist together in at least one source root per experiment."
        )
    if temp_dirs_ok:
        print(
            "Temperature-source common variables:",
            ", ".join(temp_available_common) if temp_available_common else "(none)",
        )
    else:
        print("[warn] One or more temperature-source directories are missing.")
    if args.list_variables:
        return
    if temperature_variable is None:
        print(
            f"[warn] Temperature variable '{args.temperature_variable}' not found in temperature source and no complete cached profile set was found. Freezing-level overlay disabled."
        )
    elif temperature_from_cache_only:
        print(
            f"Freezing level source variable: {temperature_variable} (using cached temperature profiles)"
        )
    else:
        print(f"Freezing level source variable: {temperature_variable}")
    print("\nVariables to plot:", plot_variables)
    return HydrometeorRun(
        args=args,
        spatial_window=spatial_window,
        spatial_tag=spatial_tag,
        geopotential_dirs=geopotential_dirs,
        periods=periods,
        output_dir=output_dir,
        intermediate_dir=intermediate_dir,
        temperature_intermediate_dir=temperature_intermediate_dir,
        n_workers=n_workers,
        variable_roots_by_experiment=variable_roots_by_experiment,
        updraft_source_dirs=updraft_source_dirs,
        plot_variables=plot_variables,
        temperature_source_dirs=temperature_source_dirs,
        temperature_variable=temperature_variable,
    )


def _build_jobs(plan: HydrometeorRun):
    """Choose the source and cache for each experiment, variable and period."""
    profile_variables_needed = list(plan.plot_variables)
    if (
        plan.temperature_variable is not None
        and plan.temperature_variable not in profile_variables_needed
    ):
        profile_variables_needed.append(plan.temperature_variable)
    jobs: list[tuple[str, str, str, str, str, tuple[int, ...] | None]] = []
    for period in plan.periods:
        for variable in profile_variables_needed:
            for exp in EXPERIMENTS:
                if plan.temperature_variable is not None and variable == plan.temperature_variable:
                    source_dir = plan.temperature_source_dirs[exp]
                elif is_updraft_derived_variable(variable):
                    if plan.updraft_source_dirs is None or exp not in plan.updraft_source_dirs:
                        raise RuntimeError(
                            f"Derived updraft source root unresolved for {exp}. Check that UD_OMEGA and UD_MESH_FRAC are available."
                        )
                    source_dir = plan.updraft_source_dirs[exp]
                else:
                    source_dir = plan.variable_roots_by_experiment[exp].get(variable)
                    if source_dir is None:
                        raise RuntimeError(
                            f"Variable '{variable}' source root unresolved for {exp}."
                        )
                cache_file = build_diurnal_cache_file(
                    intermediate_dir=plan.temperature_intermediate_dir
                    if plan.temperature_variable is not None
                    and variable == plan.temperature_variable
                    else plan.intermediate_dir,
                    variable=variable,
                    period_subdir=period.output_subdir,
                    experiment=exp,
                    spatial_tag=plan.spatial_tag,
                )
                jobs.append(
                    (
                        period.key,
                        variable,
                        exp,
                        str(cache_file),
                        str(source_dir),
                        period.allowed_months,
                    )
                )
    print(f"\nComputing {len(jobs)} profile jobs with up to {plan.n_workers} processes...")
    return jobs


def _compute_profiles(plan: HydrometeorRun, jobs):
    """Execute jobs with separate values, sample counts and failure records."""
    window = plan.spatial_window
    tasks = [
        (
            (period, variable, exp),
            (
                cache,
                root,
                variable,
                plan.args.max_days,
                plan.args.overwrite_intermediate,
                months,
                plan.args.utc_offset_hours,
                window.y_start,
                window.y_end,
                window.x_start,
                window.x_end,
            ),
        )
        for period, variable, exp, cache, root, months in jobs
    ]
    results, count_stats, errors = {}, {}, {}
    for key, value, error in execute_jobs(run_profile_job, tasks, workers=plan.n_workers):
        if error is not None:
            errors[key] = error
        else:
            mean, sample_file, count_min, count_max = value
            results[key] = (mean, Path(sample_file))
            count_stats[key] = (count_min, count_max)
    return results, count_stats, errors


def _choose_scales(plan: HydrometeorRun, results):
    """Choose common absolute and anomaly scales for comparable fields."""
    from alaro_analysis.plotting.hydrometeor import (
        build_scale_group_map,
        compute_global_plot_scales,
    )

    scale_group_map = build_scale_group_map(
        plot_variables=plan.plot_variables,
        disable_default_groups=plan.args.disable_default_shared_groups,
    )
    global_scales = compute_global_plot_scales(
        plot_variables=plan.plot_variables,
        periods=plan.periods,
        results=results,
        group_map=scale_group_map,
        anomaly_percentile=plan.args.anomaly_percentile,
        absolute_low_percentile=plan.args.absolute_low_percentile,
        absolute_high_percentile=plan.args.absolute_high_percentile,
    )
    print("\nFixed colorbar scale groups:")
    for group_id in sorted(set(scale_group_map.values())):
        vars_in_group = [v for v in plan.plot_variables if scale_group_map[v] == group_id]
        scale = global_scales[group_id]
        abs_limits = scale["abs_limits"]
        anom_scale = scale["anom_scale"]
        abs_mode = "linear" if bool(scale.get("abs_linear", False)) else "log"
        print(
            f"- {group_id}: vars={vars_in_group}, abs=[{abs_limits[0]:.3e}, {abs_limits[1]:.3e}], abs_mode={abs_mode}, anom=±{anom_scale:.3e}"
        )
    return (scale_group_map, global_scales)


def _render_results(
    plan: HydrometeorRun, results, count_stats, errors, scale_group_map, global_scales
):
    """Align heights and render complete experiment comparisons."""
    from alaro_analysis.plotting.hydrometeor import (
        plot_three_panels,
    )

    total_saved = 0
    for period in plan.periods:
        print(f"\n==================== {period.label} ({period.key}) ====================")
        for variable in plan.plot_variables:
            print(f"\n=== {period.key} | {variable} ===", flush=True)
            profiles: dict[str, np.ndarray] = {}
            sample_files: dict[str, Path] = {}
            missing = False
            for exp in EXPERIMENTS:
                key = (period.key, variable, exp)
                if key not in results:
                    err = errors.get(key, "missing result")
                    print(f"Skipping {variable}: {exp} profile unavailable ({err})")
                    missing = True
                    break
                mean, sample = results[key]
                profiles[exp] = mean
                sample_files[exp] = sample
            if missing:
                continue
            count_msgs: list[str] = []
            count_ranges: list[tuple[int, int]] = []
            for exp in EXPERIMENTS:
                stats = count_stats.get((period.key, variable, exp))
                if stats is None:
                    continue
                count_min, count_max = stats
                count_msgs.append(f"{exp}: min={count_min}, max={count_max}")
                count_ranges.append((count_min, count_max))
            if count_msgs:
                print("Count diagnostics:", "; ".join(count_msgs))
                if len(set(count_ranges)) > 1:
                    print(
                        "[warn] Count ranges differ across experiments; check NaN-vs-zero masking consistency."
                    )
            axis_by_exp: dict[str, VerticalAxis] = {}
            for exp in EXPERIMENTS:
                axis_exp = build_vertical_axis(
                    args=plan.args,
                    intermediate_dir=plan.intermediate_dir,
                    period=period,
                    experiment=exp,
                    geopotential_dir=plan.geopotential_dirs[exp],
                    sample_file=sample_files[exp],
                    variable=variable,
                    spatial_window=plan.spatial_window,
                    spatial_tag=plan.spatial_tag,
                )
                axis_exp, profile_exp = align_axis_and_profile(
                    axis=axis_exp,
                    profile=profiles[exp],
                    variable=variable,
                    period_key=period.key,
                    experiment=exp,
                )
                axis_by_exp[exp] = axis_exp
                profiles[exp] = profile_exp
            if all((axis_by_exp[exp].is_height_km for exp in EXPERIMENTS)):
                axis = axis_by_exp["control"]
                target_height_km = np.asarray(axis.values, dtype=np.float64)
                profiles_plot = {"control": profiles["control"]}
                for exp in ("graupel", "2mom"):
                    profiles_plot[exp] = interpolate_profile_to_target_height(
                        source_height_km=axis_by_exp[exp].values,
                        source_profile=profiles[exp],
                        target_height_km=target_height_km,
                    )
            else:
                axis, profiles_plot = align_vertical_shapes(
                    axis=axis_by_exp["control"],
                    profiles=profiles,
                    variable=variable,
                    period_key=period.key,
                )
            freezing_lines_km: dict[str, np.ndarray | None] | None = None
            if plan.temperature_variable is not None:
                freezing_lines_km = {}
                for exp in EXPERIMENTS:
                    t_key = (period.key, plan.temperature_variable, exp)
                    if t_key in results:
                        temp_profile = results[t_key][0]
                        freezing_lines_km[exp] = compute_freezing_line_km(
                            axis=axis_by_exp[exp], temperature_profiles=temp_profile
                        )
                    else:
                        if t_key in errors:
                            print(
                                f"[warn] Missing {plan.temperature_variable} for {exp} in {period.key}: {errors[t_key]}"
                            )
                        freezing_lines_km[exp] = None
            output_base = plan.output_dir / safe_name(variable) / _figure_subdir(period)
            if plan.spatial_tag != "full-domain":
                output_base = output_base / plan.spatial_tag
            output_file = output_base / _figure_filename(variable, period)
            scale_group = scale_group_map[variable]
            fixed_scale = global_scales[scale_group]
            plot_three_panels(
                variable=variable,
                control_mean=profiles_plot["control"],
                graupel_mean=profiles_plot["graupel"],
                twomom_mean=profiles_plot["2mom"],
                axis=axis,
                output_file=output_file,
                max_height_km=plan.args.max_height_km,
                period_label="" if period.key == "full_2yr" else period.label,
                freezing_lines_km=freezing_lines_km,
                fixed_abs_limits=fixed_scale["abs_limits"],
                fixed_anom_scale=fixed_scale["anom_scale"],
                fixed_abs_linear=bool(fixed_scale.get("abs_linear", False)),
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
    results, count_stats, errors = _compute_profiles(plan, jobs)
    scale_group_map, global_scales = _choose_scales(plan, results)
    _render_results(plan, results, count_stats, errors, scale_group_map, global_scales)
    if errors:
        details = "; ".join(f"{key}: {message}" for key, message in errors.items())
        raise RuntimeError(f"{len(errors)} profile job(s) failed: {details}")


def __getattr__(name):
    """Compatibility for pre-refactor imports; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    main()
