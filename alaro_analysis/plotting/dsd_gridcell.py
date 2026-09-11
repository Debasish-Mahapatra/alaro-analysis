"""Dsd gridcell. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from alaro_analysis.analysis.cfad import frequency_percent
from alaro_analysis.common.dsd_models import ExperimentCfad
from alaro_analysis.dsd.profile_cfad import DSD_FIELDS


def percentile_vmax(
    cfads: dict[str, ExperimentCfad],
    experiments: Sequence[str],
    field: str,
    percentile: float,
) -> float:
    values: list[np.ndarray] = []
    for experiment in experiments:
        freq = frequency_percent(cfads[experiment].grids[field].hist)
        finite = freq[np.isfinite(freq) & (freq > 0.0)]
        if finite.size:
            values.append(finite)
    if not values:
        return 1.0
    merged = np.concatenate(values)
    if percentile >= 100.0:
        return max(1.0, float(np.nanmax(merged)))
    return max(1.0, float(np.nanpercentile(merged, percentile)))


def matched_vmax_by_field(
    full_cfads: dict[str, ExperimentCfad],
    strong_cfads: dict[str, ExperimentCfad],
    experiments: Sequence[str],
    percentile: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for field in DSD_FIELDS:
        out[field] = max(
            percentile_vmax(full_cfads, experiments, field, percentile),
            percentile_vmax(strong_cfads, experiments, field, percentile),
        )
    return out
