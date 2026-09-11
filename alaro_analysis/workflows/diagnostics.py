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
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from alaro_analysis.analysis.diagnostic_profiles import (
    align_axis_and_profile,
    compute_bl_gradient_line,
    compute_column_lines,
    compute_downdraft_profiles,
    compute_geopotential_height_profile,
    compute_kt273_lines,
    compute_precip_profiles,
    compute_thermo_profiles,
    compute_vertical_derivative,
)
from alaro_analysis.common.cli_config import add_config_argument, parse_configured_args
from alaro_analysis.common.constants import EXPERIMENTS, SEASONS
from alaro_analysis.common.models import PeriodSpec, SpatialWindow, VerticalAxis
from alaro_analysis.common.naming import safe_name
from alaro_analysis.common.paths import runs_path
from alaro_analysis.common.seasons import build_period_specs, resolve_seasons
from alaro_analysis.common.spatial import build_spatial_window, spatial_window_tag
from alaro_analysis.common.vertical import (
    compute_freezing_line_km,
    interpolate_profile_to_target_height,
)
from alaro_analysis.data.cache import build_cache_file
from alaro_analysis.data.cache_context import CacheContext
from alaro_analysis.data.discovery import (
    collect_file_records,
    discover_variable_maps,
    resolve_var_name,
)

DEFAULT_CONTROL_DIR = runs_path("control/masked-netcdf")
DEFAULT_GRAUPEL_DIR = runs_path("graupel/masked-netcdf")
DEFAULT_2MOM_DIR = runs_path("2mom/masked-netcdf")

DEFAULT_CONTROL_GEO_DIR = DEFAULT_CONTROL_DIR / "GEOPOTENTIEL"
DEFAULT_GRAUPEL_GEO_DIR = DEFAULT_GRAUPEL_DIR / "GEOPOTENTIEL"
DEFAULT_2MOM_GEO_DIR = DEFAULT_2MOM_DIR / "GEOPOTENTIEL"

DEFAULT_OUTPUT_DIR = runs_path("figures/diagnostics")
DEFAULT_INTERMEDIATE_DIR = runs_path("processed-data/diagnostics")

ANALYSIS_CHOICES = ("downdraft", "precip", "thermo", "kt273", "column")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extended diagnostics (downdraft/precip/thermo/KT273/column) from masked NetCDF."
    )
    add_config_argument(parser)
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
    return parse_configured_args(parser, "diagnostics", argv=argv)


@dataclass(frozen=True)
class DiagnosticsRun:
    """Resolved source locations and settings for one command invocation."""

    args: argparse.Namespace
    analyses: list[str]
    spatial_window: SpatialWindow
    spatial_tag: str
    experiment_dirs: dict[str, Path]
    geopotential_dirs: dict[str, Path]
    periods: list[PeriodSpec]
    output_dir: Path
    intermediate_dir: Path
    resolved: dict[str, dict[str, str]]
    cache: CacheContext


def _prepare_run(args: argparse.Namespace):
    """Validate inputs and resolve the sources used by every subsequent stage."""
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
    source_roots = [
        experiment_dirs[exp] / name for exp in EXPERIMENTS for name in resolved[exp].values()
    ]
    source_roots += list(geopotential_dirs.values())
    cache = CacheContext.from_netcdf("diagnostics", vars(args), source_roots)
    return DiagnosticsRun(
        args=args,
        analyses=analyses,
        spatial_window=spatial_window,
        spatial_tag=spatial_tag,
        experiment_dirs=experiment_dirs,
        geopotential_dirs=geopotential_dirs,
        periods=periods,
        output_dir=output_dir,
        intermediate_dir=intermediate_dir,
        resolved=resolved,
        cache=cache,
    )


def _build_height_axes(plan: DiagnosticsRun):
    """Read or compute the vertical coordinate for each experiment and period."""
    axis_by_period_exp: dict[tuple[str, str], VerticalAxis] = {}
    for period in plan.periods:
        for exp in EXPERIMENTS:
            cache_file = build_cache_file(
                intermediate_dir=plan.intermediate_dir,
                analysis_name="geopotential",
                period_subdir=period.output_subdir,
                experiment=exp,
                spatial_tag=plan.spatial_tag,
            )
            if plan.cache.valid(cache_file) and (not plan.args.overwrite_intermediate):
                payload = plan.cache.load(cache_file)
                height_m = payload.get("height_m")
                if height_m is None:
                    raise RuntimeError(f"Corrupt height cache: {cache_file}")
            else:
                height_m, _ = compute_geopotential_height_profile(
                    geopotential_dir=plan.geopotential_dirs[exp],
                    height_variable=plan.args.height_variable,
                    max_days=plan.args.max_days,
                    allowed_months=period.allowed_months,
                    utc_offset_hours=plan.args.utc_offset_hours,
                    aggregate=plan.args.height_aggregate,
                    spatial_window=plan.spatial_window,
                )
                if plan.args.max_days is None:
                    plan.cache.save(
                        cache_file, {"height_m": np.asarray(height_m, dtype=np.float64)}
                    )
            axis_by_period_exp[period.key, exp] = VerticalAxis(
                values=np.asarray(height_m, dtype=np.float64) / 1000.0,
                label="Height (km)",
                is_height_km=True,
            )
    return axis_by_period_exp


def _compute_diagnostics(plan: DiagnosticsRun, axis_by_period_exp):
    """Compute requested diagnostic families and preserve their separate products."""
    vertical_results: dict[tuple[str, str, str], np.ndarray] = {}
    line_results: dict[tuple[str, str, str], np.ndarray] = {}
    for period in plan.periods:
        print(f"\n===== Computing diagnostics for {period.label} ({period.key}) =====", flush=True)
        for exp in EXPERIMENTS:
            print(f"[{period.key}/{exp}] collecting records...", flush=True)
            ref_candidates = ("PRESSURE", "CV_PREC_FLUX", "DD_OMEGA", "KT273GRAUPEL")
            ref_var = next(
                (plan.resolved[exp].get(v) for v in ref_candidates if plan.resolved[exp].get(v)),
                None,
            )
            if ref_var is None:
                print(
                    f"[warn] {period.key}/{exp}: no reference variable found; skipping.", flush=True
                )
                continue
            ref_dir = plan.experiment_dirs[exp] / ref_var
            records = collect_file_records(
                variable_dir=ref_dir,
                max_days=plan.args.max_days,
                allowed_months=period.allowed_months,
                utc_offset_hours=plan.args.utc_offset_hours,
            )
            if not records:
                print(f"[warn] {period.key}/{exp}: no records found in {ref_dir}", flush=True)
                continue
            freezing_lines: dict[str, np.ndarray] = {}
            for fexp in EXPERIMENTS:
                axis_fexp = axis_by_period_exp[period.key, fexp]
                temp_cache_file = (
                    plan.intermediate_dir
                    / "temperature"
                    / period.output_subdir
                    / f"{fexp}_{plan.spatial_tag}_diurnal_profile.npz"
                )
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
            if "downdraft" in plan.analyses:
                needed = ("UD_OMEGA", "UD_MESH_FRAC", "DD_OMEGA", "DD_MESH_FRAC")
                if all((k in plan.resolved[exp] for k in needed)):
                    cache_file = build_cache_file(
                        intermediate_dir=plan.intermediate_dir,
                        analysis_name="downdraft",
                        period_subdir=period.output_subdir,
                        experiment=exp,
                        spatial_tag=plan.spatial_tag,
                    )
                    if plan.cache.valid(cache_file) and (not plan.args.overwrite_intermediate):
                        payload = plan.cache.load(cache_file)
                    else:
                        payload = compute_downdraft_profiles(
                            experiment=exp,
                            experiment_dir=plan.experiment_dirs[exp],
                            records=records,
                            names={
                                k: plan.resolved[exp][k]
                                for k in ("UD_OMEGA", "UD_MESH_FRAC", "DD_OMEGA", "DD_MESH_FRAC")
                            },
                            spatial_window=plan.spatial_window,
                        )
                        if plan.args.max_days is None:
                            plan.cache.save(cache_file, payload)
                    for diag, arr in payload.items():
                        vertical_results[period.key, diag, exp] = arr
                else:
                    missing = [k for k in needed if k not in plan.resolved[exp]]
                    print(
                        f"[warn] {period.key}/{exp}: skipping downdraft analysis; missing {missing}",
                        flush=True,
                    )
            if "precip" in plan.analyses:
                needed = ("CV_PREC_FLUX", "ST_PREC_FLUX")
                if all((k in plan.resolved[exp] for k in needed)):
                    cache_file = build_cache_file(
                        intermediate_dir=plan.intermediate_dir,
                        analysis_name="precip",
                        period_subdir=period.output_subdir,
                        experiment=exp,
                        spatial_tag=plan.spatial_tag,
                    )
                    if plan.cache.valid(cache_file) and (not plan.args.overwrite_intermediate):
                        payload = plan.cache.load(cache_file)
                    else:
                        payload = compute_precip_profiles(
                            experiment=exp,
                            experiment_dir=plan.experiment_dirs[exp],
                            records=records,
                            names={
                                "CV_PREC_FLUX": plan.resolved[exp]["CV_PREC_FLUX"],
                                "ST_PREC_FLUX": plan.resolved[exp]["ST_PREC_FLUX"],
                            },
                            spatial_window=plan.spatial_window,
                        )
                        if plan.args.max_days is None:
                            plan.cache.save(cache_file, payload)
                    for diag, arr in payload.items():
                        vertical_results[period.key, diag, exp] = arr
                else:
                    missing = [k for k in needed if k not in plan.resolved[exp]]
                    print(
                        f"[warn] {period.key}/{exp}: skipping precip analysis; missing {missing}",
                        flush=True,
                    )
            if "thermo" in plan.analyses:
                needed = ("TEMPERATURE", "HUMI.SPECIFI", "PRESSURE")
                if all((k in plan.resolved[exp] for k in needed)):
                    z_km = axis_by_period_exp[period.key, exp].values
                    cache_file = build_cache_file(
                        intermediate_dir=plan.intermediate_dir,
                        analysis_name="thermo",
                        period_subdir=period.output_subdir,
                        experiment=exp,
                        spatial_tag=plan.spatial_tag,
                    )
                    if plan.cache.valid(cache_file) and (not plan.args.overwrite_intermediate):
                        payload = plan.cache.load(cache_file)
                    else:
                        payload = compute_thermo_profiles(
                            experiment=exp,
                            experiment_dir=plan.experiment_dirs[exp],
                            records=records,
                            names={
                                "TEMPERATURE": plan.resolved[exp]["TEMPERATURE"],
                                "HUMI.SPECIFI": plan.resolved[exp]["HUMI.SPECIFI"],
                                "PRESSURE": plan.resolved[exp]["PRESSURE"],
                            },
                            z_m=np.asarray(z_km, dtype=np.float64) * 1000.0,
                            spatial_window=plan.spatial_window,
                        )
                        if plan.args.max_days is None:
                            plan.cache.save(cache_file, payload)
                    for diag, arr in payload.items():
                        vertical_results[period.key, diag, exp] = arr
                else:
                    missing = [k for k in needed if k not in plan.resolved[exp]]
                    print(
                        f"[warn] {period.key}/{exp}: skipping thermo analysis; missing {missing}",
                        flush=True,
                    )
            if "kt273" in plan.analyses:
                needed = (
                    "KT273GRAUPEL",
                    "KT273DD_OMEGA",
                    "KT273DD_MESH_FRA",
                    "KT273RAIN",
                    "KT273HUMI.SPECIF",
                )
                if all((k in plan.resolved[exp] for k in needed)):
                    cache_file = build_cache_file(
                        intermediate_dir=plan.intermediate_dir,
                        analysis_name="kt273",
                        period_subdir=period.output_subdir,
                        experiment=exp,
                        spatial_tag=plan.spatial_tag,
                    )
                    if plan.cache.valid(cache_file) and (not plan.args.overwrite_intermediate):
                        payload = plan.cache.load(cache_file)
                    else:
                        payload = compute_kt273_lines(
                            experiment=exp,
                            experiment_dir=plan.experiment_dirs[exp],
                            records=records,
                            names={k: plan.resolved[exp][k] for k in needed},
                            spatial_window=plan.spatial_window,
                        )
                        if plan.args.max_days is None:
                            plan.cache.save(cache_file, payload)
                    for diag, arr in payload.items():
                        line_results[period.key, diag, exp] = arr
                else:
                    missing = [k for k in needed if k not in plan.resolved[exp]]
                    print(
                        f"[warn] {period.key}/{exp}: skipping kt273 analysis; missing {missing}",
                        flush=True,
                    )
            if "column" in plan.analyses:
                needed = ("PRESSURE", "CV_PREC_FLUX", "ST_PREC_FLUX")
                condensate_keys = ("LIQUID_WATER", "RAIN", "SNOW", "SOLID_WATER", "GRAUPEL")
                available_cond = [
                    plan.resolved[exp][k] for k in condensate_keys if k in plan.resolved[exp]
                ]
                if all((k in plan.resolved[exp] for k in needed)) and available_cond:
                    cache_file = build_cache_file(
                        intermediate_dir=plan.intermediate_dir,
                        analysis_name="column",
                        period_subdir=period.output_subdir,
                        experiment=exp,
                        spatial_tag=plan.spatial_tag,
                    )
                    if plan.cache.valid(cache_file) and (not plan.args.overwrite_intermediate):
                        payload = plan.cache.load(cache_file)
                    else:
                        payload = compute_column_lines(
                            experiment=exp,
                            experiment_dir=plan.experiment_dirs[exp],
                            records=records,
                            names={
                                "PRESSURE": plan.resolved[exp]["PRESSURE"],
                                "CV_PREC_FLUX": plan.resolved[exp]["CV_PREC_FLUX"],
                                "ST_PREC_FLUX": plan.resolved[exp]["ST_PREC_FLUX"],
                            },
                            available_condensate_names=available_cond,
                            spatial_window=plan.spatial_window,
                        )
                        if plan.args.max_days is None:
                            plan.cache.save(cache_file, payload)
                    for diag, arr in payload.items():
                        line_results[period.key, diag, exp] = arr
                else:
                    missing = [k for k in needed if k not in plan.resolved[exp]]
                    if not available_cond:
                        missing.append("all condensate vars")
                    print(
                        f"[warn] {period.key}/{exp}: skipping column analysis; missing {missing}",
                        flush=True,
                    )
    return (vertical_results, line_results)


def _derive_precipitation(plan: DiagnosticsRun, axis_by_period_exp, vertical_results):
    """Derive evaporation profiles from the precipitation-flux profiles."""
    if "precip" in plan.analyses:
        for period in plan.periods:
            key = period.key
            needed = [(key, "TOTAL_PREC_FLUX", exp) for exp in EXPERIMENTS]
            if not all((k in vertical_results for k in needed)):
                continue
            axis_ctrl = axis_by_period_exp[key, "control"]
            total_profiles: dict[str, np.ndarray] = {}
            for exp in EXPERIMENTS:
                axis_exp = axis_by_period_exp[key, exp]
                prof = vertical_results[key, "TOTAL_PREC_FLUX", exp]
                axis_exp, prof = align_axis_and_profile(axis_exp, prof)
                if exp == "control":
                    total_profiles[exp] = prof
                    continue
                total_profiles[exp] = interpolate_profile_to_target_height(
                    source_height_km=axis_exp.values,
                    source_profile=prof,
                    target_height_km=axis_ctrl.values[
                        : total_profiles.get("control", prof).shape[0]
                    ],
                )
            nlev = min((total_profiles[exp].shape[0] for exp in EXPERIMENTS))
            target_h = axis_ctrl.values[:nlev]
            for exp in EXPERIMENTS:
                total_profiles[exp] = total_profiles[exp][:nlev, :]
                vertical_results[key, "TOTAL_PREC_FLUX_DFDZ", exp] = compute_vertical_derivative(
                    total_profiles[exp], target_h
                )


def _render_diagnostics(plan: DiagnosticsRun, axis_by_period_exp, vertical_results, line_results):
    """Render cached diagnostic products with shared experiment scales."""
    from alaro_analysis.plotting.diagnostics import (
        plot_lines_stacked,
        plot_three_experiment_absolute,
        plot_three_panel_anomaly,
    )

    diag_meta = {
        "DD_EXTENT": {"label": "Downdraft extent", "unit": "1", "abs_linear": True},
        "DD_FLUX": {"label": "Downdraft mass flux", "unit": "kg/m²/s", "abs_linear": True},
        "DD_INTENSITY": {
            "label": "Downdraft intensity |DD_OMEGA|",
            "unit": "Pa/s",
            "abs_linear": True,
        },
        "NET_CONVECTIVE_FLUX": {
            "label": "Net convective mass flux (UD−DD)",
            "unit": "kg/m²/s",
            "abs_linear": True,
        },
        "CV_PREC_FLUX": {"label": "Convective precip flux", "unit": "", "abs_linear": True},
        "ST_PREC_FLUX": {"label": "Stratiform precip flux", "unit": "", "abs_linear": True},
        "TOTAL_PREC_FLUX": {"label": "Total precip flux", "unit": "", "abs_linear": True},
        "CONVECTIVE_PRECIP_FRACTION": {
            "label": "Convective precip fraction",
            "unit": "1",
            "abs_linear": True,
        },
        "TOTAL_PREC_FLUX_DFDZ": {
            "label": "d(Total precip flux)/dz",
            "unit": "m-1",
            "abs_linear": True,
        },
        "THETA_E": {"label": "Equivalent potential temperature", "unit": "K", "abs_linear": True},
        "MSE": {"label": "Moist static energy", "unit": "J/kg", "abs_linear": True},
        "DD_TO_UD_FLUX_RATIO": {
            "label": "Downdraft-to-updraft flux ratio",
            "unit": "1",
            "abs_linear": True,
        },
    }
    three_panel_diags = []
    if "downdraft" in plan.analyses:
        three_panel_diags.extend(("DD_EXTENT", "DD_FLUX", "DD_INTENSITY", "NET_CONVECTIVE_FLUX"))
    if "precip" in plan.analyses:
        three_panel_diags.extend(
            (
                "CV_PREC_FLUX",
                "ST_PREC_FLUX",
                "TOTAL_PREC_FLUX",
                "CONVECTIVE_PRECIP_FRACTION",
                "TOTAL_PREC_FLUX_DFDZ",
            )
        )
    if "thermo" in plan.analyses:
        three_panel_diags.extend(("THETA_E", "MSE"))
    for period in plan.periods:
        key = period.key
        print(f"\n===== Plotting {period.label} ({period.key}) =====", flush=True)
        freezing_lines: dict[str, np.ndarray] = {}
        for exp in EXPERIMENTS:
            axis_exp = axis_by_period_exp[key, exp]
            temp_profile = None
            temp_cache_file = (
                plan.intermediate_dir
                / "temperature"
                / period.output_subdir
                / f"{exp}_{plan.spatial_tag}_diurnal_profile.npz"
            )
            if temp_cache_file.exists():
                try:
                    temp_profile = np.load(temp_cache_file, allow_pickle=True)["mean"]
                except Exception:
                    temp_profile = None
            if temp_profile is None:
                temp_profile = vertical_results.get((key, "TEMPERATURE", exp))
            if temp_profile is not None:
                fl = compute_freezing_line_km(axis_exp, [temp_profile])
                if fl is not None:
                    freezing_lines[exp] = fl
        for diag in three_panel_diags:
            if not all(((key, diag, exp) in vertical_results for exp in EXPERIMENTS)):
                continue
            axis_ctrl = axis_by_period_exp[key, "control"]
            axis_profiles: dict[str, np.ndarray] = {}
            for exp in EXPERIMENTS:
                axis_exp = axis_by_period_exp[key, exp]
                arr = vertical_results[key, diag, exp]
                axis_exp, arr = align_axis_and_profile(axis_exp, arr)
                if exp == "control":
                    axis_profiles[exp] = arr
                else:
                    axis_profiles[exp] = interpolate_profile_to_target_height(
                        source_height_km=axis_exp.values,
                        source_profile=arr,
                        target_height_km=axis_ctrl.values,
                    )
            nlev = min((axis_profiles[exp].shape[0] for exp in EXPERIMENTS))
            axis = VerticalAxis(
                values=axis_ctrl.values[:nlev], label="Height (km)", is_height_km=True
            )
            c = axis_profiles["control"][:nlev, :]
            g = axis_profiles["graupel"][:nlev, :]
            t = axis_profiles["2mom"][:nlev, :]
            out = (
                plan.output_dir
                / "height_hour_panels"
                / safe_name(diag)
                / period.output_subdir
                / (
                    f"{safe_name(diag)}_panel_c1m_g1m-c1m_g2m-g1m"
                    + (f"_{plan.spatial_tag}" if plan.spatial_tag != "full-domain" else "")
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
                max_height_km=plan.args.max_height_km,
                abs_linear=bool(meta.get("abs_linear", True)),
                freezing_lines=freezing_lines,
            )
        if "downdraft" in plan.analyses and all(
            ((key, "DD_TO_UD_FLUX_RATIO", exp) in vertical_results for exp in EXPERIMENTS)
        ):
            axis_ctrl = axis_by_period_exp[key, "control"]
            ratio_profiles: dict[str, np.ndarray] = {}
            for exp in EXPERIMENTS:
                axis_exp = axis_by_period_exp[key, exp]
                arr = vertical_results[key, "DD_TO_UD_FLUX_RATIO", exp]
                axis_exp, arr = align_axis_and_profile(axis_exp, arr)
                if exp == "control":
                    ratio_profiles[exp] = arr
                else:
                    ratio_profiles[exp] = interpolate_profile_to_target_height(
                        source_height_km=axis_exp.values,
                        source_profile=arr,
                        target_height_km=axis_ctrl.values,
                    )
            nlev = min((ratio_profiles[exp].shape[0] for exp in EXPERIMENTS))
            axis = VerticalAxis(
                values=axis_ctrl.values[:nlev], label="Height (km)", is_height_km=True
            )
            ratio_profiles = {exp: arr[:nlev, :] for exp, arr in ratio_profiles.items()}
            out = (
                plan.output_dir
                / "ratio_panels"
                / "dd_to_ud_flux_ratio"
                / period.output_subdir
                / (
                    "dd_to_ud_flux_ratio_experiment_panels"
                    + (f"_{plan.spatial_tag}" if plan.spatial_tag != "full-domain" else "")
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
                max_height_km=plan.args.max_height_km,
                diverging=True,
            )
        if "kt273" in plan.analyses:
            order = ("KT273GRAUPEL", "KT273DD_FLUX", "KT273RAIN", "KT273HUMI_SPECIF")
            if all(((key, d, exp) in line_results for d in order for exp in EXPERIMENTS)):
                line_data = {
                    d: {exp: line_results[key, d, exp] for exp in EXPERIMENTS} for d in order
                }
                out = (
                    plan.output_dir
                    / "line_panels"
                    / "kt273"
                    / period.output_subdir
                    / (
                        "kt273_stacked_diurnal"
                        + (f"_{plan.spatial_tag}" if plan.spatial_tag != "full-domain" else "")
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
                        "KT273GRAUPEL": "kg/kg",
                        "KT273DD_FLUX": "kg m$^{-2}$ s$^{-1}$",
                        "KT273RAIN": "kg/kg",
                        "KT273HUMI_SPECIF": "kg/kg",
                    },
                    line_data=line_data,
                    output_file=out,
                )
        if "column" in plan.analyses:
            order = ("COLUMN_CONDENSATE", "SURFACE_PRECIP_FLUX", "RESIDENCE_TIME")
            if all(((key, d, exp) in line_results for d in order for exp in EXPERIMENTS)):
                line_data = {
                    d: {exp: line_results[key, d, exp] for exp in EXPERIMENTS} for d in order
                }
                out = (
                    plan.output_dir
                    / "line_panels"
                    / "column"
                    / period.output_subdir
                    / (
                        "column_condensate_residence_diurnal"
                        + (f"_{plan.spatial_tag}" if plan.spatial_tag != "full-domain" else "")
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
        if "thermo" in plan.analyses and all(
            ((key, "THETA_E", exp) in vertical_results for exp in EXPERIMENTS)
        ):
            axis_ctrl = axis_by_period_exp[key, "control"]
            theta_profiles: dict[str, np.ndarray] = {}
            for exp in EXPERIMENTS:
                axis_exp = axis_by_period_exp[key, exp]
                arr = vertical_results[key, "THETA_E", exp]
                axis_exp, arr = align_axis_and_profile(axis_exp, arr)
                if exp == "control":
                    theta_profiles[exp] = arr
                else:
                    theta_profiles[exp] = interpolate_profile_to_target_height(
                        source_height_km=axis_exp.values,
                        source_profile=arr,
                        target_height_km=axis_ctrl.values,
                    )
            nlev = min((theta_profiles[exp].shape[0] for exp in EXPERIMENTS))
            z = axis_ctrl.values[:nlev]
            bl_lines = {
                exp: compute_bl_gradient_line(theta_profiles[exp][:nlev, :], z)
                for exp in EXPERIMENTS
            }
            out = (
                plan.output_dir
                / "line_panels"
                / "thermo"
                / period.output_subdir
                / (
                    "thetae_bl_gradient_diurnal"
                    + (f"_{plan.spatial_tag}" if plan.spatial_tag != "full-domain" else "")
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


def main(argv: list[str] | None = None) -> None:
    plan = _prepare_run(parse_args(argv))
    if plan is None:
        return
    axis_by_period_exp = _build_height_axes(plan)
    vertical_results, line_results = _compute_diagnostics(plan, axis_by_period_exp)
    _derive_precipitation(plan, axis_by_period_exp, vertical_results)
    _render_diagnostics(plan, axis_by_period_exp, vertical_results, line_results)


def __getattr__(name):
    """Compatibility for pre-refactor imports; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    main()
