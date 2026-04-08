from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from alaro_analysis.common.naming import normalize_var_token
from alaro_analysis.common.timeparse import (
    has_pf_subdirs,
    is_supported_day_dir_name,
    parse_month_from_day_name,
    parse_utc_hour_from_name,
)


def discover_variables(experiment_dirs: dict[str, Path]) -> dict[str, set[str]]:
    """Return the set of variable-directory names present per experiment."""
    available: dict[str, set[str]] = {}
    for exp, exp_dir in experiment_dirs.items():
        vars_for_exp = {
            p.name
            for p in exp_dir.iterdir()
            if p.is_dir() and has_pf_subdirs(p) and not p.name.startswith(".")
        }
        available[exp] = vars_for_exp
    return available


def discover_variable_maps(
    experiment_dirs: dict[str, Path],
) -> dict[str, dict[str, str]]:
    """Build normalised-token -> directory-name maps per experiment.

    Example: if the directory is ``CLPMHAUT.MOD.XFU``, the token is
    ``CLPMHAUTMODXFU`` which can then be matched against user input.
    """
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
    """Resolve user-facing variable names to actual directory names.

    Tries each candidate in order, returning the first match.
    """
    token_map = variable_maps.get(experiment, {})
    for cand in candidates:
        token = normalize_var_token(cand)
        if token in token_map:
            return token_map[token]
    return None


def collect_file_records(
    variable_dir: Path,
    max_days: int | None,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
) -> list[tuple[int, Path]]:
    """Discover hourly NetCDF files and return (local_hour, path) records."""
    if not variable_dir.exists():
        raise FileNotFoundError(f"Missing directory: {variable_dir}")

    allowed_set = set(allowed_months) if allowed_months is not None else None
    day_dirs = sorted(
        path
        for path in variable_dir.iterdir()
        if path.is_dir() and is_supported_day_dir_name(path.name)
    )
    if max_days is not None:
        day_dirs = day_dirs[:max_days]

    records: list[tuple[int, Path]] = []
    for day_dir in day_dirs:
        month = parse_month_from_day_name(day_dir.name)
        if allowed_set is not None and (month is None or month not in allowed_set):
            continue

        for file_path in sorted(day_dir.glob("*.nc")):
            utc_hour = parse_utc_hour_from_name(file_path.name)
            if utc_hour is None:
                continue
            local_hour = (utc_hour + utc_offset_hours) % 24
            records.append((local_hour, file_path))

    return records
