"""Cfad cache. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from alaro_analysis.common.dsd_models import DSD_FIELDS as PANEL_FIELDS
from alaro_analysis.common.dsd_models import CfadGrid, ExperimentCfad
from alaro_analysis.data.cache import save_npz_compressed


def save_experiment_cfad(path: Path, cfad: ExperimentCfad, sig: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "temperature_k": cfad.temperature_k,
        "freezing_level_km": np.asarray([cfad.freezing_level_km], dtype=np.float64),
        "__signature__": np.asarray([sig]),
        "__metadata__": np.asarray(
            json.dumps(
                {
                    "schema_version": 2,
                    "dimensions": ["height_bin", "diagnostic_bin"],
                    "normalization": "hist contains raw counts; plotting normalizes each height bin",
                    "height_units": "km",
                    "field_units": {"d0_mm": "mm", "log_nw": "log10(m-3 mm-1)"},
                },
                sort_keys=True,
            )
        ),
    }
    for field, grid in cfad.grids.items():
        payload[f"height__{field}"] = grid.height_km
        payload[f"x_edges__{field}"] = grid.x_edges
        payload[f"hist__{field}"] = grid.hist
        payload[f"counts__{field}"] = grid.counts
        payload[f"n_profiles__{field}"] = np.asarray([grid.n_profiles], dtype=np.int64)
        payload[f"source__{field}"] = np.asarray([grid.source])
    save_npz_compressed(path, **payload)


def load_experiment_cfad(path: Path) -> ExperimentCfad:
    with np.load(path, allow_pickle=False) as data:
        grids: dict[str, CfadGrid] = {}
        for field in PANEL_FIELDS:
            grids[field] = CfadGrid(
                height_km=np.asarray(data[f"height__{field}"], dtype=np.float64),
                x_edges=np.asarray(data[f"x_edges__{field}"], dtype=np.float64),
                hist=np.asarray(data[f"hist__{field}"], dtype=np.int64),
                counts=np.asarray(data[f"counts__{field}"], dtype=np.int64),
                n_profiles=int(np.ravel(data[f"n_profiles__{field}"])[0]),
                source=str(data[f"source__{field}"][0]),
            )
        return ExperimentCfad(
            grids=grids,
            temperature_k=np.asarray(data["temperature_k"], dtype=np.float64),
            freezing_level_km=float(np.ravel(data["freezing_level_km"])[0]),
        )
