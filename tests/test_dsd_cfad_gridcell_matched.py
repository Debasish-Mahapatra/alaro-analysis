from __future__ import annotations

import numpy as np

from alaro_analysis.workflows import dsd_cfad_gridcell_matched as gridcell


def test_add_grid_values_to_hist_bins_each_grid_cell_by_level():
    hist = np.zeros((2, 3), dtype=np.int64)
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array(
        [
            [[0.2, 0.8], [1.4, 4.0]],
            [[2.2, np.nan], [1.2, -1.0]],
        ]
    )
    sample_mask = np.array(
        [
            [[True, True], [True, True]],
            [[True, True], [False, True]],
        ]
    )

    gridcell.add_grid_values_to_hist(hist, values, sample_mask, edges)

    np.testing.assert_array_equal(hist, [[2, 1, 0], [0, 0, 1]])


def test_group_records_by_day_keeps_sorted_day_groups(tmp_path):
    def rec(day: str, hour: int):
        rain = tmp_path / "RAIN" / day / f"pfABOFABOF+{hour:04d}.nc"
        return (np.datetime64("2014-01-01"), np.datetime64("2014-01-01"), hour, {"RAIN": rain})

    groups = gridcell.group_records_by_day(
        [
            rec("pf20140102", 0),
            rec("pf20140101", 1),
            rec("pf20140101", 0),
        ]
    )

    assert [[row[2] for row in group] for group in groups] == [[1, 0], [0]]


def test_matched_vmax_uses_larger_percentile_from_pair():
    full = {"control": _experiment_with_hist(np.array([[1, 3], [0, 0]], dtype=np.int64))}
    strong = {"control": _experiment_with_hist(np.array([[1, 9], [1, 0]], dtype=np.int64))}

    vmax = gridcell.matched_vmax_by_field(full, strong, ["control"], 100.0)

    assert vmax == {"d0_mm": 100.0, "log_nw": 100.0}


def _experiment_with_hist(hist: np.ndarray):
    from alaro_analysis.workflows.dsd_cfad_profiles import CfadGrid, ExperimentCfad

    grids = {}
    for field in gridcell.DSD_FIELDS:
        grids[field] = CfadGrid(
            height_km=np.array([0.5, 1.5]),
            x_edges=np.array([0.0, 1.0, 2.0]),
            hist=hist.copy(),
            counts=np.sum(hist, axis=1),
            n_profiles=1,
            source="test",
        )
    return ExperimentCfad(
        grids=grids,
        temperature_k=np.array([290.0, 280.0]),
        freezing_level_km=1.0,
    )
