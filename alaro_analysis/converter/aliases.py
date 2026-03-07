from __future__ import annotations

from pathlib import Path
import re
from typing import Optional, Sequence

import faxarray as fx

from .models import VariablePlan

MODEL_LEVEL_FIELD_RE = re.compile(r"^S(\d{3})(.+)$")
PRESSURE_LEVEL_FIELD_RE = re.compile(r"^P(\d{5})(.+)$")
VAR_TOKEN_SANITIZE_RE = re.compile(r"[^A-Za-z0-9]+")

REQUESTED_VAR_FALLBACK_ALIASES: dict[str, tuple[str, ...]] = {
    "KT273TEMPERATUR": (
        "KT273TEMPERATURE",
        "KT273TEMPERATU",
    ),
    "NC_LIQUID_WA": (
        "NC_LIQUID_WAT",
        "NC_LIQUID_WATER",
        "NC.LIQUID.WA",
        "NC.LIQUID.WAT",
        "NC.LIQUID.WATER",
    ),
}


def var_to_ds_name(name: str) -> str:
    return name.replace(".", "_")


def normalize_var_token(name: str) -> str:
    return VAR_TOKEN_SANITIZE_RE.sub("", name).upper()


def build_available_aliases(raw_field_names: Sequence[str]) -> set[str]:
    aliases: set[str] = set()
    for field in raw_field_names:
        aliases.add(field)
        aliases.add(var_to_ds_name(field))

    model_groups: dict[str, int] = {}
    pressure_groups: dict[str, int] = {}
    for field in raw_field_names:
        match = MODEL_LEVEL_FIELD_RE.match(field)
        if match:
            base = match.group(2)
            model_groups[base] = model_groups.get(base, 0) + 1
            continue
        match = PRESSURE_LEVEL_FIELD_RE.match(field)
        if match:
            base = match.group(2)
            pressure_groups[base] = pressure_groups.get(base, 0) + 1

    for base, count in model_groups.items():
        if count > 1:
            aliases.add(base)
            aliases.add(var_to_ds_name(base))

    for base, count in pressure_groups.items():
        if count > 1:
            p_name = f"P_{base}"
            aliases.add(p_name)
            aliases.add(var_to_ds_name(p_name))

    return aliases


def resolve_requested_vars(
    sample_file: Path,
    requested_vars: Sequence[str],
    *,
    derived_model_rh_var: str,
    model_rh_q_var: str,
    model_rh_t_var: str,
    model_rh_pressure_candidates: Sequence[str],
) -> VariablePlan:
    fa = fx.FADataset(str(sample_file))
    try:
        raw_field_names = tuple(fa.variables)
        aliases = build_available_aliases(raw_field_names)
    finally:
        fa.close()

    raw_field_set = set(raw_field_names)
    normalized_aliases: dict[str, list[str]] = {}
    for alias in aliases:
        norm = normalize_var_token(alias)
        if not norm:
            continue
        normalized_aliases.setdefault(norm, []).append(alias)

    def choose_best_alias(
        candidates: Sequence[str],
        requested_name: str,
    ) -> Optional[str]:
        unique = sorted(
            set(candidates),
            key=lambda cand: (
                0 if cand in raw_field_set else 1,
                abs(len(cand) - len(requested_name)),
                cand,
            ),
        )
        if not unique:
            return None
        if len(unique) == 1:
            return unique[0]

        first = unique[0]
        second = unique[1]
        first_score = (
            0 if first in raw_field_set else 1,
            abs(len(first) - len(requested_name)),
        )
        second_score = (
            0 if second in raw_field_set else 1,
            abs(len(second) - len(requested_name)),
        )
        if first_score == second_score:
            return None
        return first

    def resolve_alias(name: str) -> Optional[str]:
        if name in aliases:
            return name

        alt_candidates: list[str] = []
        if "." in name:
            alt_candidates.append(name.replace(".", "_"))
        elif "_" in name and not name.startswith("P_"):
            alt_candidates.append(name.replace("_", "."))
        alt_candidates.extend(REQUESTED_VAR_FALLBACK_ALIASES.get(name, ()))

        direct = [cand for cand in alt_candidates if cand in aliases]
        chosen = choose_best_alias(direct, name)
        if chosen is not None:
            return chosen

        norm_name = normalize_var_token(name)
        if not norm_name:
            return None

        normalized_hits = normalized_aliases.get(norm_name, [])
        chosen = choose_best_alias(normalized_hits, name)
        if chosen is not None:
            return chosen

        fuzzy_hits: list[str] = []
        for norm_alias, alias_names in normalized_aliases.items():
            if min(len(norm_alias), len(norm_name)) < 6:
                continue
            if abs(len(norm_alias) - len(norm_name)) > 2:
                continue
            if norm_alias.startswith(norm_name) or norm_name.startswith(norm_alias):
                fuzzy_hits.extend(alias_names)

        return choose_best_alias(fuzzy_hits, name)

    output_vars: list[str] = []
    missing: list[str] = []
    for name in requested_vars:
        if name == derived_model_rh_var:
            output_vars.append(name)
            continue

        resolved = resolve_alias(name)
        if resolved is not None:
            output_vars.append(resolved)
            continue

        missing.append(name)

    output_vars = list(dict.fromkeys(output_vars))
    read_vars = [v for v in output_vars if v != derived_model_rh_var]
    derive_model_rh = derived_model_rh_var in output_vars
    rh_q_var: str | None = None
    rh_t_var: str | None = None
    rh_p_var: str | None = None

    if derive_model_rh:
        rh_q_var = resolve_alias(model_rh_q_var)
        rh_t_var = resolve_alias(model_rh_t_var)

        for candidate in model_rh_pressure_candidates:
            resolved = resolve_alias(candidate)
            if resolved is not None:
                rh_p_var = resolved
                break

        if rh_q_var is None:
            missing.append(f"{derived_model_rh_var} requires {model_rh_q_var}")
        if rh_t_var is None:
            missing.append(f"{derived_model_rh_var} requires {model_rh_t_var}")
        if rh_p_var is None:
            missing.append(
                f"{derived_model_rh_var} requires one of: "
                + ", ".join(model_rh_pressure_candidates)
            )

        for dep in (rh_q_var, rh_t_var, rh_p_var):
            if dep and dep not in read_vars:
                read_vars.append(dep)

    return VariablePlan(
        output_vars=tuple(output_vars),
        read_vars=tuple(read_vars),
        missing_requested=tuple(missing),
        derive_model_rh=derive_model_rh,
        rh_q_var=rh_q_var,
        rh_t_var=rh_t_var,
        rh_p_var=rh_p_var,
    )


__all__ = [
    "REQUESTED_VAR_FALLBACK_ALIASES",
    "build_available_aliases",
    "normalize_var_token",
    "resolve_requested_vars",
    "var_to_ds_name",
]
