"""Profile sources. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from pathlib import Path

from alaro_analysis.common.constants import EXPERIMENTS
from alaro_analysis.common.models import PeriodSpec
from alaro_analysis.common.timeparse import has_pf_subdirs
from alaro_analysis.data.cache import build_diurnal_cache_file

UPDRAFT_OMEGA_VAR = "UD_OMEGA"


UPDRAFT_MESH_VAR = "UD_MESH_FRAC"


UPDRAFT_DERIVED_VARIABLES = (
    "UPDRAFT_FLUX",
    "UPDRAFT_EXTENT",
    "UPDRAFT_INTENSITY",
)


UPDRAFT_DERIVED_SET = set(UPDRAFT_DERIVED_VARIABLES)


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
    experiment_sources: dict[str, list[Path]],
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
    variable_roots_by_experiment: dict[str, dict[str, Path]],
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
    experiment_sources: dict[str, list[Path]],
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
    return UPDRAFT_OMEGA_VAR.lower() in common_map and UPDRAFT_MESH_VAR.lower() in common_map


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
                raise ValueError(f"Variable '{value}' is not common across all experiments.")
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
