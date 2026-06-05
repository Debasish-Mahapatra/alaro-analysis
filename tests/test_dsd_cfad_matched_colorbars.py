from __future__ import annotations

import numpy as np

from alaro_analysis.workflows.dsd_cfad_profiles import CfadGrid, ExperimentCfad
from alaro_analysis.workflows import dsd_cfad_matched_colorbars as matched


def _cfads_with_hist(hist: np.ndarray) -> dict[str, ExperimentCfad]:
    grids = {}
    for field in matched.DSD_FIELDS:
        grids[field] = CfadGrid(
            height_km=np.array([0.5, 1.5]),
            x_edges=np.array([0.0, 1.0, 2.0]),
            hist=hist.copy(),
            counts=np.sum(hist, axis=1),
            n_profiles=2,
            source="test",
        )
    return {
        "control": ExperimentCfad(
            grids=grids,
            temperature_k=np.array([290.0, 280.0]),
            freezing_level_km=1.0,
        )
    }


def test_matched_vmax_by_field_uses_larger_exact_scale():
    full = _cfads_with_hist(np.array([[1, 1], [3, 1]], dtype=np.int64))
    strong = _cfads_with_hist(np.array([[9, 1], [1, 0]], dtype=np.int64))

    vmax = matched.matched_vmax_by_field(
        full,
        strong,
        ["control"],
        frequency_scale="exact-max",
        frequency_percentile=99.0,
    )

    assert vmax == {"d0_mm": 100.0, "log_nw": 100.0}


def test_matched_name_keeps_extension():
    assert matched.matched_name("plot.png") == "plot_matched_colorbar.png"
