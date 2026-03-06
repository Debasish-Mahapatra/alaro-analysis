from __future__ import annotations

from pathlib import Path

from .constants import SEASONS
from .models import PeriodSpec


def resolve_seasons(season_args: list[str]) -> list[str]:
    normalized = [item.strip().lower() for item in season_args if item.strip()]
    if not normalized or "all" in normalized:
        return list(SEASONS.keys())
    unknown = [season for season in normalized if season not in SEASONS]
    if unknown:
        raise ValueError(f"Unknown season(s): {unknown}. Valid: {list(SEASONS.keys())} or all.")

    result: list[str] = []
    seen: set[str] = set()
    for season in normalized:
        if season in seen:
            continue
        seen.add(season)
        result.append(season)
    return result


def build_period_specs(mode_set: set[str], selected_seasons: list[str]) -> list[PeriodSpec]:
    specs: list[PeriodSpec] = []
    if "full" in mode_set:
        specs.append(
            PeriodSpec(
                key="full_2yr",
                label="Full 2-year (all months)",
                output_subdir=Path("2years"),
            )
        )

    if "seasonal" in mode_set:
        for season_key in selected_seasons:
            specs.append(
                PeriodSpec(
                    key=season_key,
                    label=SEASONS[season_key]["label"],
                    output_subdir=Path("seasonal") / season_key,
                    allowed_months=tuple(SEASONS[season_key]["months"]),
                )
            )
    return specs
