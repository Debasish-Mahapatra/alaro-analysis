#!/usr/bin/env python3
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
4) Uses up to 16 processes for profile computation.
5) Uses GEOPOTENTIEL (meters) for vertical axis and overlays a freezing-level
   line derived from TEMPERATURE (not as separate temperature plots).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cmaps

from alaro_analysis.common.constants import (
    EXPERIMENTS,
    EXPERIMENT_LABELS,
    SEASONS,
)
from alaro_analysis.common.models import PeriodSpec, SpatialWindow, VerticalAxis
from alaro_analysis.common.naming import safe_name
from alaro_analysis.common.seasons import build_period_specs, resolve_seasons
from alaro_analysis.common.spatial import (
    apply_spatial_window_to_array,
    build_spatial_window,
    spatial_window_tag,
)
from alaro_analysis.common.timeparse import has_pf_subdirs
from alaro_analysis.common.vertical import (
    centers_to_edges,
    compute_freezing_line_km,
    interpolate_profile_to_target_height,
)
from alaro_analysis.data.cache import (
    build_diurnal_cache_file,
    build_height_cache_file,
    load_diurnal_profile_cache,
    load_height_profile_cache,
    save_diurnal_profile_cache,
    save_height_profile_cache,
)
from alaro_analysis.data.dataset_io import (
    nanmean_with_count,
    read_vertical_profile,
    resolve_data_var_name,
)
from alaro_analysis.data.discovery import collect_file_records, discover_variables
from alaro_analysis.plotting.scales import robust_anomaly_scale, robust_log_limits
from alaro_analysis.plotting.style import resolve_workers

DEFAULT_CONTROL_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/masked-netcdf-2"
)
DEFAULT_GRAUPEL_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/graupel/masked-netcdf-2"
)
DEFAULT_2MOM_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/2mom/masked-netcdf-2"
)

DEFAULT_CONTROL_GEO_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/masked-netcdf/GEOPOTENTIEL"
)
DEFAULT_GRAUPEL_GEO_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/graupel/masked-netcdf/GEOPOTENTIEL"
)
DEFAULT_2MOM_GEO_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/2mom/masked-netcdf/GEOPOTENTIEL"
)

DEFAULT_OUTPUT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures")
DEFAULT_INTERMEDIATE_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data"
)

UPDRAFT_OMEGA_VAR = "UD_OMEGA"
UPDRAFT_MESH_VAR = "UD_MESH_FRAC"
UPDRAFT_DERIVED_VARIABLES = (
    "UPDRAFT_FLUX",
    "UPDRAFT_EXTENT",
    "UPDRAFT_INTENSITY",
)
UPDRAFT_DERIVED_SET = set(UPDRAFT_DERIVED_VARIABLES)
DEFAULT_SHARED_SCALE_GROUPS = (
    ("LIQUID_WATER", "RAD_LIQUID_W"),
    ("SOLID_WATER", "RAD_SOLID_WA"),
)
LINEAR_ABSOLUTE_VARIABLE_KEYS = {
    "HUMI.RELATIVE",
    "CLOUD_FRACTI",
    "SURFNEBUL.BASSE",
    "SURFNEBUL.HAUTE",
    "SURFNEBUL.MOYENN",
    "SURFNEBUL.TOTALE",
    "UD_MESH_FRAC",
    "UPDRAFT_EXTENT",
    "UPDRAFT_INTENSITY",
    "UPDRAFT_FLUX",
    "TEMPERATURE",
    "HUMI.SPECIFI",
    "RAIN",
    "SNOW",
    "GRAUPEL",
    "LIQUID_WATER",
    "SOLID_WATER",
    "RAD_LIQUID_W",
    "RAD_SOLID_WA",
}

# The variables that should strictly be bounded [0.0, 1.0] when linear
UNIT_INTERVAL_VARIABLE_KEYS = {
    "HUMI.RELATIVE",
    "CLOUD_FRACTI",
    "SURFNEBUL.BASSE",
    "SURFNEBUL.HAUTE",
    "SURFNEBUL.MOYENN",
    "SURFNEBUL.TOTALE",
    "UD_MESH_FRAC",
    "UPDRAFT_EXTENT",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full+seasonal hydrometeor diurnal panels from masked-netcdf-2."
    )
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

    parser.add_argument(
        "--control-geopotential-dir", type=Path, default=DEFAULT_CONTROL_GEO_DIR
    )
    parser.add_argument(
        "--graupel-geopotential-dir", type=Path, default=DEFAULT_GRAUPEL_GEO_DIR
    )
    parser.add_argument(
        "--twomom-geopotential-dir", type=Path, default=DEFAULT_2MOM_GEO_DIR
    )

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
        default=16,
        help="Multiprocessing worker count (capped to 16).",
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
    return parser.parse_args()


def variable_label(name: str) -> str:
    key = name.strip().upper()
    if key == "UPDRAFT_FLUX":
        return "Updraft Flux"
    if key == "UPDRAFT_EXTENT":
        return "Updraft Extent"
    if key == "UPDRAFT_INTENSITY":
        return "Updraft Intensity"
    return name


def variable_key(name: str) -> str:
    return name.strip().upper()


def variable_unit(name: str) -> str:
    key = variable_key(name)
    if key == "UPDRAFT_FLUX":
        return "kg/(m^2 s)"
    if key == "UPDRAFT_INTENSITY":
        return "Pa/s"
    if key == "UPDRAFT_EXTENT":
        return "Fraction"
    return ""


def uses_linear_absolute_scale(name: str) -> bool:
    return variable_key(name) in LINEAR_ABSOLUTE_VARIABLE_KEYS


def is_unit_interval_variable(name: str) -> bool:
    return variable_key(name) in UNIT_INTERVAL_VARIABLE_KEYS


def build_scale_group_map(
    plot_variables: list[str], disable_default_groups: bool
) -> dict[str, str]:
    group_map = {var: f"var:{variable_key(var)}" for var in plot_variables}
    if disable_default_groups:
        return group_map

    key_to_var = {variable_key(var): var for var in plot_variables}
    for idx, members in enumerate(DEFAULT_SHARED_SCALE_GROUPS):
        present = [key_to_var[m] for m in members if m in key_to_var]
        if len(present) < 2:
            continue
        group_id = f"default-group:{idx}"
        for var in present:
            group_map[var] = group_id
    return group_map


def compute_global_plot_scales(
    plot_variables: list[str],
    periods: list[PeriodSpec],
    results: dict[tuple[str, str, str], tuple[np.ndarray, Path]],
    group_map: dict[str, str],
    anomaly_percentile: float,
    absolute_low_percentile: float,
    absolute_high_percentile: float,
) -> dict[str, dict[str, object]]:
    scales: dict[str, dict[str, object]] = {}
    group_ids = sorted(set(group_map.values()))

    for group_id in group_ids:
        abs_chunks: list[np.ndarray] = []
        anom_chunks: list[np.ndarray] = []

        group_vars = [v for v in plot_variables if group_map[v] == group_id]
        abs_linear = all(uses_linear_absolute_scale(v) for v in group_vars)
        group_is_unit_interval = all(is_unit_interval_variable(v) for v in group_vars)
        for variable in group_vars:
            var_key = variable_key(variable)
            for period in periods:
                k_ctrl = (period.key, variable, "control")
                k_g1 = (period.key, variable, "graupel")
                k_g2 = (period.key, variable, "2mom")
                if k_ctrl not in results or k_g1 not in results or k_g2 not in results:
                    continue

                ctrl = results[k_ctrl][0]
                g1 = results[k_g1][0]
                g2 = results[k_g2][0]

                abs_field = g1 if var_key == "GRAUPEL" else ctrl
                if abs_linear:
                    abs_valid = abs_field[np.isfinite(abs_field)]
                else:
                    abs_valid = abs_field[np.isfinite(abs_field) & (abs_field > 0)]
                if abs_valid.size > 0:
                    abs_chunks.append(abs_valid)

                diff_g1 = g1 - ctrl
                diff_g2 = g2 - g1
                if var_key == "GRAUPEL":
                    diff_valid = diff_g2[np.isfinite(diff_g2)]
                    if diff_valid.size > 0:
                        anom_chunks.append(diff_valid)
                else:
                    d1 = diff_g1[np.isfinite(diff_g1)]
                    d2 = diff_g2[np.isfinite(diff_g2)]
                    if d1.size > 0:
                        anom_chunks.append(d1)
                    if d2.size > 0:
                        anom_chunks.append(d2)

        if abs_chunks:
            abs_all = np.concatenate(abs_chunks)
            vmin = float(np.percentile(abs_all, absolute_low_percentile))
            vmax = float(np.percentile(abs_all, absolute_high_percentile))
            if abs_linear:
                if group_is_unit_interval:
                    vmin = max(0.0, vmin)
                    vmax = min(1.0, vmax)
                if vmax <= vmin:
                    vmax = min(1.0, vmin + 1e-6)
            else:
                vmin = max(vmin, float(np.min(abs_all)))
                if vmin <= 0:
                    pos = abs_all[abs_all > 0]
                    vmin = float(np.min(pos)) if pos.size > 0 else 1e-12
                if vmax <= vmin:
                    vmax = vmin * 10.0
            abs_limits = (vmin, vmax)
        else:
            abs_limits = (1e-12, 1.0)

        if anom_chunks:
            anom_all = np.concatenate(anom_chunks)
            anom_abs = np.abs(anom_all[np.isfinite(anom_all)])
            if anom_abs.size > 0:
                scale = float(np.percentile(anom_abs, anomaly_percentile))
                if scale <= 0:
                    scale = float(np.max(anom_abs))
                if scale <= 0:
                    scale = 1.0
            else:
                scale = 1.0
        else:
            scale = 1.0

        scales[group_id] = {
            "abs_limits": abs_limits,
            "anom_scale": scale,
            "abs_linear": abs_linear,
        }

    return scales


def build_experiment_sources(
    primary_dirs: dict[str, Path],
    secondary_dirs: dict[str, Path | None],
) -> dict[str, list[Path]]:
    sources: dict[str, list[Path]] = {}
    for exp in EXPERIMENTS:
        srcs = [primary_dirs[exp].resolve()]
        second = secondary_dirs[exp]
        if second is not None:
            second_resolved = second.resolve()
            if second_resolved not in srcs:
                srcs.append(second_resolved)
        sources[exp] = srcs
    return sources


def discover_variable_roots(
    experiment_sources: dict[str, list[Path]]
) -> dict[str, dict[str, Path]]:
    roots: dict[str, dict[str, Path]] = {}
    for exp, source_dirs in experiment_sources.items():
        var_to_root: dict[str, Path] = {}
        for source_dir in source_dirs:
            for p in source_dir.iterdir():
                if not p.is_dir() or p.name.startswith(".") or not has_pf_subdirs(p):
                    continue
                if p.name not in var_to_root:
                    var_to_root[p.name] = source_dir
        roots[exp] = var_to_root
    return roots


def discover_variables_from_roots(
    variable_roots_by_experiment: dict[str, dict[str, Path]]
) -> dict[str, set[str]]:
    return {exp: set(var_roots.keys()) for exp, var_roots in variable_roots_by_experiment.items()}


def common_variable_map(available_by_experiment: dict[str, set[str]]) -> dict[str, str]:
    common = set.intersection(*(vars_set for vars_set in available_by_experiment.values()))
    mapping: dict[str, str] = {}
    for name in sorted(common):
        mapping[name.lower()] = name
    return mapping


def is_updraft_derived_variable(variable: str) -> bool:
    return variable.strip().upper() in UPDRAFT_DERIVED_SET


def resolve_updraft_source_dirs(
    experiment_sources: dict[str, list[Path]]
) -> dict[str, Path] | None:
    selected: dict[str, Path] = {}
    for exp, source_dirs in experiment_sources.items():
        match: Path | None = None
        for source_dir in source_dirs:
            if has_pf_subdirs(source_dir / UPDRAFT_OMEGA_VAR) and has_pf_subdirs(
                source_dir / UPDRAFT_MESH_VAR
            ):
                match = source_dir
                break
        if match is None:
            return None
        selected[exp] = match
    return selected


def updraft_derived_available(
    available_by_experiment: dict[str, set[str]],
    updraft_source_dirs: dict[str, Path] | None = None,
) -> bool:
    if updraft_source_dirs is not None:
        return all(exp in updraft_source_dirs for exp in EXPERIMENTS)
    common_map = common_variable_map(available_by_experiment)
    return (
        UPDRAFT_OMEGA_VAR.lower() in common_map
        and UPDRAFT_MESH_VAR.lower() in common_map
    )


def resolve_plot_variables(
    available_by_experiment: dict[str, set[str]],
    requested: list[str] | None,
    temperature_var: str,
    derived_available_override: bool | None = None,
) -> list[str]:
    common_map = common_variable_map(available_by_experiment)
    temp_key = temperature_var.lower()
    derived_map = {name.lower(): name for name in UPDRAFT_DERIVED_VARIABLES}
    derived_ok = (
        updraft_derived_available(available_by_experiment)
        if derived_available_override is None
        else derived_available_override
    )

    if requested is None:
        selected = [common_map[key] for key in sorted(common_map) if key != temp_key]
    else:
        selected = []
        seen: set[str] = set()
        for token in requested:
            value = token.strip()
            if not value:
                continue
            key = value.lower()
            if key in common_map:
                canonical = common_map[key]
            elif key in derived_map:
                if not derived_ok:
                    raise ValueError(
                        f"Derived variable '{value}' requires both "
                        f"{UPDRAFT_OMEGA_VAR} and {UPDRAFT_MESH_VAR} to be common "
                        "across experiments."
                    )
                canonical = derived_map[key]
            else:
                raise ValueError(
                    f"Variable '{value}' is not common across all experiments."
                )
            if canonical in seen:
                continue
            seen.add(canonical)
            selected.append(canonical)

    if not selected:
        raise RuntimeError("No plottable variables selected.")
    return selected


def resolve_temperature_variable(
    available_by_experiment: dict[str, set[str]], requested_temperature_var: str
) -> str | None:
    common_map = common_variable_map(available_by_experiment)
    return common_map.get(requested_temperature_var.lower())


def temperature_cache_available(
    temperature_intermediate_dir: Path,
    temperature_variable: str,
    periods: list[PeriodSpec],
    spatial_tag: str,
) -> bool:
    for period in periods:
        for exp in EXPERIMENTS:
            cache_file = build_diurnal_cache_file(
                intermediate_dir=temperature_intermediate_dir,
                variable=temperature_variable,
                period_subdir=period.output_subdir,
                experiment=exp,
                spatial_tag=spatial_tag,
            )
            if not cache_file.exists():
                return False
    return True


def read_field_array(file_path: Path, requested_variable: str) -> np.ndarray:
    with xr.open_dataset(file_path, decode_times=False) as ds:
        var_name = resolve_data_var_name(ds, requested_variable, compact_match=True)
        return np.asarray(ds[var_name].values, dtype=np.float64)


def as_time_level_yx(arr: np.ndarray, file_path: Path) -> np.ndarray:
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        return arr[np.newaxis, :, :, :]
    raise ValueError(f"Expected 3D/4D array in {file_path}, got shape {arr.shape}")


def compute_updraft_derived_profile_from_files(
    omega_file: Path,
    mesh_file: Path,
    derived_variable: str,
    spatial_window: SpatialWindow,
) -> np.ndarray:
    omega = as_time_level_yx(read_field_array(omega_file, UPDRAFT_OMEGA_VAR), omega_file)
    mesh = as_time_level_yx(read_field_array(mesh_file, UPDRAFT_MESH_VAR), mesh_file)
    omega = apply_spatial_window_to_array(omega, spatial_window, omega_file)
    mesh = apply_spatial_window_to_array(mesh, spatial_window, mesh_file)

    if omega.shape != mesh.shape:
        raise ValueError(
            f"Shape mismatch for derived updraft profile: "
            f"{omega_file} {omega.shape} vs {mesh_file} {mesh.shape}"
        )

    name = derived_variable.strip().upper()
    if name == "UPDRAFT_FLUX":
        # M_u = - \sigma_u * \omega_u / g
        # Since omega is negative for updrafts, `-omega` is positive.
        # We also enforce that we only care where mesh > 0.
        flux = np.where(mesh > 0, (-omega * mesh) / 9.80665, 0.0)
        profile, _ = nanmean_with_count(flux, axis=(0, 2, 3))
        return profile

    if name == "UPDRAFT_EXTENT":
        profile, _ = nanmean_with_count(mesh, axis=(0, 2, 3))
        return profile

    if name == "UPDRAFT_INTENSITY":
        # Intensity is just UD_OMEGA. We take the absolute value so it plots positively.
        # We only average where the updraft is actually active (mesh > 0).
        omega_abs = np.where(mesh > 0, np.abs(omega), np.nan)
        profile, _ = nanmean_with_count(omega_abs, axis=(0, 2, 3))
        return profile

    raise ValueError(f"Unknown derived updraft variable: {derived_variable}")


def compute_diurnal_profile(
    experiment_dir: Path,
    variable: str,
    max_days: int | None,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
    spatial_window: SpatialWindow,
) -> tuple[np.ndarray, np.ndarray, int, Path]:
    is_derived = is_updraft_derived_variable(variable)

    if is_derived:
        omega_dir = experiment_dir / UPDRAFT_OMEGA_VAR
        mesh_dir = experiment_dir / UPDRAFT_MESH_VAR
        records = collect_file_records(
            variable_dir=omega_dir,
            max_days=max_days,
            allowed_months=allowed_months,
            utc_offset_hours=utc_offset_hours,
        )
        if not records:
            raise RuntimeError(f"No valid +0000..+0023 files found in {omega_dir}")

        first_omega = records[0][1]
        first_mesh = mesh_dir / first_omega.parent.name / first_omega.name
        if not first_mesh.exists():
            raise FileNotFoundError(
                f"Missing paired {UPDRAFT_MESH_VAR} file for {first_omega}: {first_mesh}"
            )
        first_profile = compute_updraft_derived_profile_from_files(
            omega_file=first_omega,
            mesh_file=first_mesh,
            derived_variable=variable,
            spatial_window=spatial_window,
        )
    else:
        variable_dir = experiment_dir / variable
        records = collect_file_records(
            variable_dir=variable_dir,
            max_days=max_days,
            allowed_months=allowed_months,
            utc_offset_hours=utc_offset_hours,
        )
        if not records:
            raise RuntimeError(f"No valid +0000..+0023 files found in {variable_dir}")
        first_profile, _ = read_vertical_profile(
            records[0][1],
            variable,
            spatial_window=spatial_window,
            compact_match=True,
        )

    n_levels = first_profile.size
    sums = np.zeros((n_levels, 24), dtype=np.float64)
    counts = np.zeros((n_levels, 24), dtype=np.int64)

    for idx, (local_hour, file_path) in enumerate(records, start=1):
        if is_derived:
            mesh_path = (
                (experiment_dir / UPDRAFT_MESH_VAR) / file_path.parent.name / file_path.name
            )
            if not mesh_path.exists():
                raise FileNotFoundError(
                    f"Missing paired {UPDRAFT_MESH_VAR} file for {file_path}: {mesh_path}"
                )
            profile = compute_updraft_derived_profile_from_files(
                omega_file=file_path,
                mesh_file=mesh_path,
                derived_variable=variable,
                spatial_window=spatial_window,
            )
        else:
            profile, _ = read_vertical_profile(
                file_path,
                variable,
                spatial_window=spatial_window,
                compact_match=True,
            )

        if profile.size != n_levels:
            raise ValueError(
                f"Inconsistent vertical levels in {file_path}: "
                f"{profile.size} vs expected {n_levels}"
            )
        valid = np.isfinite(profile)
        sums[valid, local_hour] += profile[valid]
        counts[valid, local_hour] += 1

        if idx % 2000 == 0 or idx == len(records):
            print(
                f"[{experiment_dir.name}/{variable}] {idx}/{len(records)} files",
                flush=True,
            )

    mean = np.full_like(sums, np.nan)
    nonzero = counts > 0
    mean[nonzero] = sums[nonzero] / counts[nonzero]
    return mean, counts, len(records), records[0][1]


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
        profile, _ = read_vertical_profile(
            records[0][1],
            height_variable,
            spatial_window=spatial_window,
            compact_match=True,
        )
        return profile, 1

    first, _ = read_vertical_profile(
        records[0][1],
        height_variable,
        spatial_window=spatial_window,
        compact_match=True,
    )
    sums = np.zeros_like(first, dtype=np.float64)
    counts = np.zeros_like(first, dtype=np.int64)

    for idx, (_, file_path) in enumerate(records, start=1):
        profile, _ = read_vertical_profile(
            file_path,
            height_variable,
            spatial_window=spatial_window,
            compact_match=True,
        )
        valid = np.isfinite(profile)
        sums[valid] += profile[valid]
        counts[valid] += 1

        if idx % 4000 == 0 or idx == len(records):
            print(
                f"[{geopotential_dir.parent.name}/GEOPOTENTIEL] {idx}/{len(records)} files",
                flush=True,
            )

    mean = np.full_like(sums, np.nan)
    nonzero = counts > 0
    mean[nonzero] = sums[nonzero] / counts[nonzero]
    return mean, len(records)


def load_or_compute_diurnal(
    cache_file: Path,
    experiment_dir: Path,
    variable: str,
    max_days: int | None,
    overwrite_intermediate: bool,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
    spatial_window: SpatialWindow,
) -> tuple[np.ndarray, Path, int, int]:
    use_cache = cache_file.exists() and (not overwrite_intermediate)
    if use_cache:
        mean, counts, _, sample_file = load_diurnal_profile_cache(cache_file)
        if sample_file is None:
            raise ValueError(f"Missing sample_file in cache: {cache_file}")
        if counts is not None:
            positive = counts[counts > 0]
            if positive.size > 0:
                return mean, sample_file, int(np.min(positive)), int(np.max(positive))
        return mean, sample_file, 0, 0

    mean, counts, n_files, sample_file = compute_diurnal_profile(
        experiment_dir=experiment_dir,
        variable=variable,
        max_days=max_days,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
        spatial_window=spatial_window,
    )

    if max_days is None:
        save_diurnal_profile_cache(
            cache_file,
            mean=mean,
            counts=counts,
            n_files=n_files,
            sample_file=sample_file,
        )
    positive = counts[counts > 0]
    if positive.size == 0:
        return mean, sample_file, 0, 0
    return mean, sample_file, int(np.min(positive)), int(np.max(positive))


def load_or_compute_height(
    cache_file: Path,
    geopotential_dir: Path,
    height_variable: str,
    max_days: int | None,
    overwrite_intermediate: bool,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
    aggregate: str,
    spatial_window: SpatialWindow,
) -> np.ndarray:
    use_cache = cache_file.exists() and (not overwrite_intermediate)
    if use_cache:
        return load_height_profile_cache(cache_file)

    height_m, n_files = compute_geopotential_height_profile(
        geopotential_dir=geopotential_dir,
        height_variable=height_variable,
        max_days=max_days,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
        aggregate=aggregate,
        spatial_window=spatial_window,
    )
    if max_days is None:
        save_height_profile_cache(cache_file, height_m=height_m, n_files=n_files)
    return height_m


def infer_axis_from_coord(
    sample_file: Path, variable: str, spatial_window: SpatialWindow
) -> VerticalAxis:
    profile, coord = read_vertical_profile(
        sample_file,
        variable,
        spatial_window=spatial_window,
        compact_match=True,
    )
    n_levels = profile.size
    if coord is None or coord.size != n_levels:
        return VerticalAxis(
            values=np.arange(n_levels, dtype=np.float64),
            label="Model level",
            is_height_km=False,
        )

    finite = np.isfinite(coord)
    if np.sum(finite) < max(2, n_levels // 2):
        return VerticalAxis(
            values=np.arange(n_levels, dtype=np.float64),
            label="Model level",
            is_height_km=False,
        )

    cmax = float(np.nanmax(np.abs(coord[finite])))
    cmin = float(np.nanmin(np.abs(coord[finite])))
    is_integerish = np.allclose(coord[finite], np.round(coord[finite]), atol=1e-6)

    if cmax <= 60.0 and not is_integerish:
        return VerticalAxis(values=coord, label="Height (km)", is_height_km=True)
    if 10.0 <= cmin and cmax <= 1200.0:
        return VerticalAxis(values=coord, label="Pressure (hPa)", is_height_km=False)
    if 1200.0 < cmax <= 20000.0:
        return VerticalAxis(values=coord / 1000.0, label="Height (km)", is_height_km=True)
    return VerticalAxis(values=coord, label="Model level", is_height_km=False)


def align_vertical_shapes(
    axis: VerticalAxis, profiles: dict[str, np.ndarray], variable: str, period_key: str
) -> tuple[VerticalAxis, dict[str, np.ndarray]]:
    n_levels = min(
        axis.values.size,
        *(profile.shape[0] for profile in profiles.values()),
    )
    mismatch = (
        axis.values.size != n_levels
        or any(profile.shape[0] != n_levels for profile in profiles.values())
    )
    if mismatch:
        print(
            f"[warn] Vertical mismatch for {variable} ({period_key}); "
            f"truncating to {n_levels} levels.",
            flush=True,
        )
    axis_new = VerticalAxis(
        values=axis.values[:n_levels],
        label=axis.label,
        is_height_km=axis.is_height_km,
    )
    profiles_new = {exp: arr[:n_levels, :] for exp, arr in profiles.items()}
    return axis_new, profiles_new


def align_axis_and_profile(
    axis: VerticalAxis,
    profile: np.ndarray,
    variable: str,
    period_key: str,
    experiment: str,
) -> tuple[VerticalAxis, np.ndarray]:
    n_levels = min(axis.values.size, profile.shape[0])
    if axis.values.size != profile.shape[0]:
        print(
            f"[warn] Vertical mismatch for {variable} ({period_key}, {experiment}); "
            f"truncating to {n_levels} levels.",
            flush=True,
        )
    axis_new = VerticalAxis(
        values=axis.values[:n_levels],
        label=axis.label,
        is_height_km=axis.is_height_km,
    )
    return axis_new, profile[:n_levels, :]




def plot_three_panels(
    variable: str,
    control_mean: np.ndarray,
    graupel_mean: np.ndarray,
    twomom_mean: np.ndarray,
    axis: VerticalAxis,
    output_file: Path,
    max_height_km: float,
    period_label: str,
    freezing_lines_km: dict[str, np.ndarray | None] | None,
    fixed_abs_limits: tuple[float, float] | None = None,
    fixed_anom_scale: float | None = None,
    fixed_abs_linear: bool | None = None,
) -> None:
    axis_label_fs = 16
    tick_label_fs = 14
    legend_fs = 12
    panel_tag_fs = 14
    cbar_label_fs = 14
    cbar_tick_fs = 12
    var_label = variable_label(variable)
    unit = variable_unit(variable)
    is_updraft = is_updraft_derived_variable(variable)
    abs_linear = (
        uses_linear_absolute_scale(variable)
        if fixed_abs_linear is None
        else bool(fixed_abs_linear)
    )

    y = np.asarray(axis.values, dtype=np.float64)
    order = np.argsort(y)
    y = y[order]
    ctrl = control_mean[order, :]
    g1_abs = graupel_mean[order, :]
    diff_g1 = (graupel_mean - control_mean)[order, :]
    diff_g2_chain = (twomom_mean - graupel_mean)[order, :]

    if axis.is_height_km:
        keep = np.isfinite(y) & (y >= 0.0) & (y <= max_height_km)
        y = y[keep]
        ctrl = ctrl[keep, :]
        g1_abs = g1_abs[keep, :]
        diff_g1 = diff_g1[keep, :]
        diff_g2_chain = diff_g2_chain[keep, :]

    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    y_edges = centers_to_edges(y)

    is_special_graupel = variable.strip().upper() == "GRAUPEL"
    ctrl_plot = np.ma.masked_invalid(ctrl)
    g1_abs_plot = np.ma.masked_invalid(g1_abs)
    if not abs_linear:
        ctrl_plot = np.ma.masked_where(ctrl_plot <= 0, ctrl_plot)
        g1_abs_plot = np.ma.masked_where(g1_abs_plot <= 0, g1_abs_plot)
    diff_g1_plot = np.ma.masked_invalid(diff_g1)
    diff_g2_chain_plot = np.ma.masked_invalid(diff_g2_chain)

    if fixed_abs_limits is None:
        abs_ref = g1_abs_plot.filled(np.nan) if is_special_graupel else ctrl_plot.filled(np.nan)
        if abs_linear:
            valid = abs_ref[np.isfinite(abs_ref)]
            if valid.size > 0:
                vmin_abs = float(np.percentile(valid, 2.0))
                vmax_abs = float(np.percentile(valid, 98.0))
            else:
                vmin_abs, vmax_abs = 0.0, 1.0
            if is_unit_interval_variable(variable):
                vmin_abs = max(0.0, vmin_abs)
                vmax_abs = min(1.0, vmax_abs)
            if vmax_abs <= vmin_abs:
                vmax_abs = min(1.0, vmin_abs + 1e-6)
        else:
            vmin_abs, vmax_abs = robust_log_limits(abs_ref)
    else:
        vmin_abs, vmax_abs = fixed_abs_limits
        if abs_linear and is_unit_interval_variable(variable):
            vmin_abs = max(0.0, vmin_abs)
            vmax_abs = min(1.0, vmax_abs)
        if vmax_abs <= vmin_abs:
            if abs_linear:
                vmax_abs = vmin_abs + 1e-6
            else:
                vmax_abs = vmin_abs * 10.0

    if abs_linear:
        abs_norm = mcolors.Normalize(vmin=vmin_abs, vmax=vmax_abs)
    else:
        if vmax_abs <= vmin_abs:
            vmax_abs = vmin_abs * 10.0
        abs_norm = mcolors.LogNorm(vmin=vmin_abs, vmax=vmax_abs)

    if fixed_anom_scale is None:
        if is_special_graupel:
            anom_scale = robust_anomaly_scale(diff_g2_chain_plot.filled(np.nan))
        else:
            anom_scale = robust_anomaly_scale(
                diff_g1_plot.filled(np.nan),
                diff_g2_chain_plot.filled(np.nan),
            )
    else:
        anom_scale = float(fixed_anom_scale)
        if anom_scale <= 0:
            anom_scale = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    for idx, ax in enumerate(axes):
        ax.set_facecolor("#d3d3d3")
        ax.set_xlabel("Hour (Amazon UTC-4)", fontsize=axis_label_fs)
        ax.set_xticks(np.arange(0, 24, 6))
        ax.set_xlim(-0.5, 23.5)
        ax.tick_params(axis="both", labelsize=tick_label_fs)
        ax.set_ylabel(axis.label, fontsize=axis_label_fs)
        if axis.is_height_km:
            ax.set_ylim(0.0, max_height_km)
        ax.text(
            0.02,
            0.98,
            f"({chr(ord('a') + idx)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=panel_tag_fs,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2.0},
        )

    diff_norm = mcolors.TwoSlopeNorm(vmin=-anom_scale, vcenter=0.0, vmax=anom_scale)
    if is_special_graupel:
        axes[0].set_title(
            f"{EXPERIMENT_LABELS['control']} ({var_label}, Absolute)",
            fontsize=14,
            fontweight="bold",
        )
        axes[0].text(
            0.5,
            0.5,
            "No Data",
            transform=axes[0].transAxes,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.95, "pad": 4.0},
        )

        pcm_abs = axes[1].pcolormesh(
            hour_edges,
            y_edges,
            g1_abs_plot,
            cmap=cmaps.WhiteBlueGreenYellowRed,
            norm=abs_norm,
            shading="auto",
        )
        axes[1].set_title(
            f"{EXPERIMENT_LABELS['graupel']} ({var_label}, Absolute)",
            fontsize=14,
            fontweight="bold",
        )

        pcm_diff = axes[2].pcolormesh(
            hour_edges,
            y_edges,
            diff_g2_chain_plot,
            cmap="RdBu_r",
            norm=diff_norm,
            shading="auto",
        )
        axes[2].set_title(
            f"{EXPERIMENT_LABELS['2mom']} - {EXPERIMENT_LABELS['graupel']}",
            fontsize=14,
            fontweight="bold",
        )
    else:
        pcm_abs = axes[0].pcolormesh(
            hour_edges,
            y_edges,
            ctrl_plot,
            cmap=cmaps.WhiteBlueGreenYellowRed,
            norm=abs_norm,
            shading="auto",
        )
        axes[0].set_title(
            f"{EXPERIMENT_LABELS['control']} ({var_label}, Absolute)",
            fontsize=14,
            fontweight="bold",
        )

        axes[1].pcolormesh(
            hour_edges,
            y_edges,
            diff_g1_plot,
            cmap="RdBu_r",
            norm=diff_norm,
            shading="auto",
        )
        axes[1].set_title(
            f"{EXPERIMENT_LABELS['graupel']} - {EXPERIMENT_LABELS['control']}",
            fontsize=14,
            fontweight="bold",
        )

        rhs_plot = diff_g2_chain_plot
        pcm_diff = axes[2].pcolormesh(
            hour_edges,
            y_edges,
            rhs_plot,
            cmap="RdBu_r",
            norm=diff_norm,
            shading="auto",
        )
        axes[2].set_title(
            f"{EXPERIMENT_LABELS['2mom']} - {EXPERIMENT_LABELS['graupel']}",
            fontsize=14,
            fontweight="bold",
        )

    if freezing_lines_km:
        hours = np.arange(24, dtype=np.float64)
        panel_experiment = ("control", "graupel", "2mom")
        for idx, ax in enumerate(axes):
            exp = panel_experiment[idx]
            line = freezing_lines_km.get(exp)
            if line is None or not np.isfinite(line).any():
                continue
            ax.plot(
                hours,
                line,
                color="black",
                linewidth=1.8,
                linestyle="--",
                label=f"Freezing level ({EXPERIMENT_LABELS[exp]})",
                zorder=10,
            )
            ax.legend(loc="upper right", fontsize=legend_fs, framealpha=0.9)

    fig.suptitle(f"{period_label} - {var_label}", fontsize=16, fontweight="bold")

    cbar_abs = fig.colorbar(
        pcm_abs,
        ax=axes[1] if is_special_graupel else axes[0],
        orientation="horizontal",
        fraction=0.08,
        pad=0.16,
    )
    if is_updraft and variable.strip().upper() == "UPDRAFT_FLUX":
        abs_label = f"Mean Updraft Mass Flux [{unit}]"
    elif unit:
        abs_label = f"Mean {var_label} [{unit}]"
    else:
        abs_label = f"Mean {var_label} (Absolute)"
    cbar_abs.set_label(abs_label, fontsize=cbar_label_fs)
    cbar_abs.ax.tick_params(labelsize=cbar_tick_fs)

    cbar_diff = fig.colorbar(
        pcm_diff, ax=axes[2] if is_special_graupel else axes[1:], orientation="horizontal", fraction=0.08, pad=0.16
    )
    if is_special_graupel:
        diff_label = f"{var_label} anomaly ({EXPERIMENT_LABELS['2mom']}-{EXPERIMENT_LABELS['graupel']})"
    elif is_updraft:
        diff_label = f"{var_label} anomaly [{unit}]" if unit else f"{var_label} anomaly"
    else:
        diff_label = f"{var_label} anomaly"
    cbar_diff.set_label(diff_label, fontsize=cbar_label_fs)
    cbar_diff.ax.tick_params(labelsize=cbar_tick_fs)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def build_vertical_axis(
    args: argparse.Namespace,
    intermediate_dir: Path,
    period: PeriodSpec,
    experiment: str,
    geopotential_dir: Path,
    sample_file: Path,
    variable: str,
    spatial_window: SpatialWindow,
    spatial_tag: str,
) -> VerticalAxis:
    axis_source_variable = (
        UPDRAFT_OMEGA_VAR if is_updraft_derived_variable(variable) else variable
    )

    if args.height_mode in ("geopotential", "auto"):
        try:
            height_cache = build_height_cache_file(
                intermediate_dir=intermediate_dir,
                period_subdir=period.output_subdir,
                experiment=experiment,
                aggregate=args.height_aggregate,
                spatial_tag=spatial_tag,
            )
            height_m = load_or_compute_height(
                cache_file=height_cache,
                geopotential_dir=geopotential_dir,
                height_variable=args.height_variable,
                max_days=args.max_days,
                overwrite_intermediate=args.overwrite_intermediate,
                allowed_months=period.allowed_months,
                utc_offset_hours=args.utc_offset_hours,
                aggregate=args.height_aggregate,
                spatial_window=spatial_window,
            )
            return VerticalAxis(
                values=np.asarray(height_m, dtype=np.float64) / 1000.0,
                label="Height (km)",
                is_height_km=True,
            )
        except Exception as exc:
            if args.height_mode == "geopotential":
                raise
            print(
                f"[warn] Geopotential axis failed for {experiment} ({exc}); falling back.",
                flush=True,
            )

    if args.height_mode in ("coord", "auto"):
        try:
            return infer_axis_from_coord(
                sample_file, axis_source_variable, spatial_window=spatial_window
            )
        except Exception as exc:
            if args.height_mode == "coord":
                raise
            print(
                f"[warn] Coordinate axis failed for {experiment} ({exc}); falling back.",
                flush=True,
            )

    profile, _ = read_vertical_profile(
        sample_file, axis_source_variable, spatial_window=spatial_window
    )
    return VerticalAxis(
        values=np.arange(profile.size, dtype=np.float64),
        label="Model level",
        is_height_km=False,
    )


def run_profile_job(
    cache_file: str,
    experiment_dir: str,
    variable: str,
    max_days: int | None,
    overwrite_intermediate: bool,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
    y_start: int | None,
    y_end: int | None,
    x_start: int | None,
    x_end: int | None,
) -> tuple[np.ndarray, str, int, int]:
    spatial_window = SpatialWindow(
        y_start=y_start,
        y_end=y_end,
        x_start=x_start,
        x_end=x_end,
    )
    mean, sample_file, count_min, count_max = load_or_compute_diurnal(
        cache_file=Path(cache_file),
        experiment_dir=Path(experiment_dir),
        variable=variable,
        max_days=max_days,
        overwrite_intermediate=overwrite_intermediate,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
        spatial_window=spatial_window,
    )
    return mean, str(sample_file), count_min, count_max


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
            (output_dir / safe_name(variable) / period.output_subdir).mkdir(
                parents=True, exist_ok=True
            )
            (intermediate_dir / safe_name(variable) / period.output_subdir).mkdir(
                parents=True, exist_ok=True
            )

    for period in periods:
        (intermediate_dir / "geopotential" / period.output_subdir).mkdir(
            parents=True, exist_ok=True
        )


def main() -> None:
    args = parse_args()
    if not (0.0 < args.anomaly_percentile <= 100.0):
        raise ValueError("--anomaly-percentile must be in (0, 100].")
    if not (0.0 <= args.absolute_low_percentile < 100.0):
        raise ValueError("--absolute-low-percentile must be in [0, 100).")
    if not (0.0 < args.absolute_high_percentile <= 100.0):
        raise ValueError("--absolute-high-percentile must be in (0, 100].")
    if args.absolute_low_percentile >= args.absolute_high_percentile:
        raise ValueError(
            "--absolute-low-percentile must be lower than --absolute-high-percentile."
        )
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
        primary_dirs=experiment_primary_dirs,
        secondary_dirs=experiment_secondary_dirs,
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
        available_by_experiment=available,
        updraft_source_dirs=updraft_source_dirs,
    )
    plot_variables = resolve_plot_variables(
        available_by_experiment=available,
        requested=args.variables,
        temperature_var=args.temperature_variable,
        derived_available_override=derived_available,
    )
    temperature_source_dirs = {
        "control": (
            args.temperature_control_dir.resolve()
            if args.temperature_control_dir is not None
            else experiment_primary_dirs["control"]
        ),
        "graupel": (
            args.temperature_graupel_dir.resolve()
            if args.temperature_graupel_dir is not None
            else experiment_primary_dirs["graupel"]
        ),
        "2mom": (
            args.temperature_twomom_dir.resolve()
            if args.temperature_twomom_dir is not None
            else experiment_primary_dirs["2mom"]
        ),
    }

    temp_dirs_ok = all(p.exists() for p in temperature_source_dirs.values())
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
        f"Global scales: anomaly p{args.anomaly_percentile:g}, "
        f"absolute p{args.absolute_low_percentile:g}-p{args.absolute_high_percentile:g}"
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
            f"[info] Derived updraft variables disabled until both "
            f"{UPDRAFT_OMEGA_VAR} and {UPDRAFT_MESH_VAR} exist together "
            "in at least one source root per experiment."
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
            f"[warn] Temperature variable '{args.temperature_variable}' not found in "
            "temperature source and no complete cached profile set was found. "
            "Freezing-level overlay disabled."
        )
    else:
        if temperature_from_cache_only:
            print(
                f"Freezing level source variable: {temperature_variable} "
                "(using cached temperature profiles)"
            )
        else:
            print(f"Freezing level source variable: {temperature_variable}")

    print("\nVariables to plot:", plot_variables)

    profile_variables_needed = list(plot_variables)
    if temperature_variable is not None and temperature_variable not in profile_variables_needed:
        profile_variables_needed.append(temperature_variable)

    jobs: list[tuple[str, str, str, str, str, tuple[int, ...] | None]] = []
    for period in periods:
        for variable in profile_variables_needed:
            for exp in EXPERIMENTS:
                if temperature_variable is not None and variable == temperature_variable:
                    source_dir = temperature_source_dirs[exp]
                elif is_updraft_derived_variable(variable):
                    if updraft_source_dirs is None or exp not in updraft_source_dirs:
                        raise RuntimeError(
                            f"Derived updraft source root unresolved for {exp}. "
                            "Check that UD_OMEGA and UD_MESH_FRAC are available."
                        )
                    source_dir = updraft_source_dirs[exp]
                else:
                    source_dir = variable_roots_by_experiment[exp].get(variable)
                    if source_dir is None:
                        raise RuntimeError(
                            f"Variable '{variable}' source root unresolved for {exp}."
                        )

                cache_file = build_diurnal_cache_file(
                    intermediate_dir=(
                        temperature_intermediate_dir
                        if (temperature_variable is not None and variable == temperature_variable)
                        else intermediate_dir
                    ),
                    variable=variable,
                    period_subdir=period.output_subdir,
                    experiment=exp,
                    spatial_tag=spatial_tag,
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

    print(f"\nComputing {len(jobs)} profile jobs with up to {n_workers} processes...")

    results: dict[tuple[str, str, str], tuple[np.ndarray, Path]] = {}
    count_stats: dict[tuple[str, str, str], tuple[int, int]] = {}
    errors: dict[tuple[str, str, str], str] = {}

    if n_workers == 1:
        for idx, (period_key, variable, exp, cache_file, exp_dir, months) in enumerate(
            jobs, start=1
        ):
            key = (period_key, variable, exp)
            try:
                mean, sample_file, count_min, count_max = run_profile_job(
                    cache_file=cache_file,
                    experiment_dir=exp_dir,
                    variable=variable,
                    max_days=args.max_days,
                    overwrite_intermediate=args.overwrite_intermediate,
                    allowed_months=months,
                    utc_offset_hours=args.utc_offset_hours,
                    y_start=spatial_window.y_start,
                    y_end=spatial_window.y_end,
                    x_start=spatial_window.x_start,
                    x_end=spatial_window.x_end,
                )
                results[key] = (mean, Path(sample_file))
                count_stats[key] = (count_min, count_max)
            except Exception as exc:
                errors[key] = str(exc)
            if idx % 20 == 0 or idx == len(jobs):
                print(f"Finished {idx}/{len(jobs)} jobs", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_key: dict[object, tuple[str, str, str]] = {}
            for period_key, variable, exp, cache_file, exp_dir, months in jobs:
                key = (period_key, variable, exp)
                future = pool.submit(
                    run_profile_job,
                    cache_file,
                    exp_dir,
                    variable,
                    args.max_days,
                    args.overwrite_intermediate,
                    months,
                    args.utc_offset_hours,
                    spatial_window.y_start,
                    spatial_window.y_end,
                    spatial_window.x_start,
                    spatial_window.x_end,
                )
                future_to_key[future] = key

            done = 0
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                done += 1
                try:
                    mean, sample_file, count_min, count_max = future.result()
                    results[key] = (mean, Path(sample_file))
                    count_stats[key] = (count_min, count_max)
                except Exception as exc:
                    errors[key] = str(exc)
                if done % 20 == 0 or done == len(jobs):
                    print(f"Finished {done}/{len(jobs)} jobs", flush=True)

    scale_group_map = build_scale_group_map(
        plot_variables=plot_variables,
        disable_default_groups=args.disable_default_shared_groups,
    )
    global_scales = compute_global_plot_scales(
        plot_variables=plot_variables,
        periods=periods,
        results=results,
        group_map=scale_group_map,
        anomaly_percentile=args.anomaly_percentile,
        absolute_low_percentile=args.absolute_low_percentile,
        absolute_high_percentile=args.absolute_high_percentile,
    )
    print("\nFixed colorbar scale groups:")
    for group_id in sorted(set(scale_group_map.values())):
        vars_in_group = [v for v in plot_variables if scale_group_map[v] == group_id]
        scale = global_scales[group_id]
        abs_limits = scale["abs_limits"]
        anom_scale = scale["anom_scale"]
        abs_mode = "linear" if bool(scale.get("abs_linear", False)) else "log"
        print(
            f"- {group_id}: vars={vars_in_group}, "
            f"abs=[{abs_limits[0]:.3e}, {abs_limits[1]:.3e}], "
            f"abs_mode={abs_mode}, anom=±{anom_scale:.3e}"
        )

    total_saved = 0
    for period in periods:
        print(f"\n==================== {period.label} ({period.key}) ====================")
        for variable in plot_variables:
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
                        "[warn] Count ranges differ across experiments; "
                        "check NaN-vs-zero masking consistency."
                    )

            axis_by_exp: dict[str, VerticalAxis] = {}
            for exp in EXPERIMENTS:
                axis_exp = build_vertical_axis(
                    args=args,
                    intermediate_dir=intermediate_dir,
                    period=period,
                    experiment=exp,
                    geopotential_dir=geopotential_dirs[exp],
                    sample_file=sample_files[exp],
                    variable=variable,
                    spatial_window=spatial_window,
                    spatial_tag=spatial_tag,
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

            if all(axis_by_exp[exp].is_height_km for exp in EXPERIMENTS):
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
            if temperature_variable is not None:
                freezing_lines_km = {}
                for exp in EXPERIMENTS:
                    t_key = (period.key, temperature_variable, exp)
                    if t_key in results:
                        temp_profile = results[t_key][0]
                        freezing_lines_km[exp] = compute_freezing_line_km(
                            axis=axis_by_exp[exp],
                            temperature_profile=temp_profile,
                        )
                    else:
                        if t_key in errors:
                            print(
                                f"[warn] Missing {temperature_variable} for {exp} in {period.key}: "
                                f"{errors[t_key]}"
                            )
                        freezing_lines_km[exp] = None

            output_base = output_dir / safe_name(variable) / period.output_subdir
            if spatial_tag != "full-domain":
                output_base = output_base / spatial_tag
            output_file = output_base / f"{safe_name(variable)}_panel_c1m_g1m-c1m_g2m-g1m.png"
            scale_group = scale_group_map[variable]
            fixed_scale = global_scales[scale_group]
            plot_three_panels(
                variable=variable,
                control_mean=profiles_plot["control"],
                graupel_mean=profiles_plot["graupel"],
                twomom_mean=profiles_plot["2mom"],
                axis=axis,
                output_file=output_file,
                max_height_km=args.max_height_km,
                period_label=period.label,
                freezing_lines_km=freezing_lines_km,
                fixed_abs_limits=fixed_scale["abs_limits"],
                fixed_anom_scale=fixed_scale["anom_scale"],
                fixed_abs_linear=bool(fixed_scale.get("abs_linear", False)),
            )
            total_saved += 1

    print(f"\nCompleted. Total figures generated: {total_saved}")
    if errors:
        print(f"Completed with {len(errors)} profile-job errors.")


if __name__ == "__main__":
    main()
