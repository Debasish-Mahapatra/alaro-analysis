"""Disdrometer exports. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def save_timeseries_csv(
    path: Path, rows: list[dict[str, Any]], experiments: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "valid_time_utc",
        "init_time_utc",
        "lead_hours",
        "obs_rain_number_m3",
        "obs_precip_rate_mm_h",
        "obs_median_volume_diameter_mm",
        "obs_sample_count",
    ]
    exp_fields: list[str] = []
    for exp in experiments:
        exp_fields.extend(
            [
                f"{exp}_rain_number_m3",
                f"{exp}_rain_number_kg",
                f"{exp}_rain_mixing_ratio_kgkg",
                f"{exp}_dmean_mm",
                f"{exp}_rho_air_kg_m3",
                f"{exp}_temperature_k",
                f"{exp}_pressure_pa",
                f"{exp}_method",
            ]
        )
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=base_fields + exp_fields)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def datetime64_key(value: np.datetime64) -> str:
    return np.datetime_as_string(value.astype("datetime64[s]"), unit="s")
