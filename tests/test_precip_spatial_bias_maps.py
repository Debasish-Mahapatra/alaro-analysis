from __future__ import annotations

from pathlib import Path

import numpy as np

from alaro_analysis.workflows.precip_spatial_bias_maps import (
    DATASET_BY_KEY,
    SpatialField,
    build_bias_maps,
    calc_relative_bias_map,
    write_bias_txt,
    write_spatial_mean_txt,
)


def _field(key: str, values: np.ndarray) -> SpatialField:
    cfg = DATASET_BY_KEY[key]
    return SpatialField(
        config=cfg,
        source_path=Path(f"/tmp/{cfg.filename}"),
        lat=np.array([-1.0, 0.0]),
        lon=np.array([10.0, 11.0]),
        values=np.asarray(values, dtype=np.float64),
        n_time=3,
    )


def test_calc_relative_bias_masks_small_reference_values():
    reference = np.array([[1.0, 0.005], [2.0, np.nan]])
    candidate = np.array([[1.5, 2.0], [1.0, 3.0]])

    bias = calc_relative_bias_map(reference, candidate, min_reference=0.01)

    assert np.isclose(bias[0, 0], 50.0)
    assert np.isnan(bias[0, 1])
    assert np.isclose(bias[1, 0], -50.0)
    assert np.isnan(bias[1, 1])


def test_build_bias_maps_uses_requested_reference():
    fields = {
        "radar": _field("radar", [[1.0, 2.0], [3.0, 4.0]]),
        "control": _field("control", [[2.0, 1.0], [3.0, 8.0]]),
    }

    maps = build_bias_maps(
        fields,
        reference_key="radar",
        comparison_keys=("control",),
        min_reference=0.01,
    )

    np.testing.assert_allclose(maps["control"], [[100.0, -50.0], [0.0, 100.0]])


def test_write_bias_txt_includes_summary_and_gridpoint_data(tmp_path: Path):
    fields = {
        "radar": _field("radar", [[1.0, 2.0], [3.0, 4.0]]),
        "control": _field("control", [[2.0, 1.0], [3.0, 8.0]]),
    }
    bias_maps = build_bias_maps(
        fields,
        reference_key="radar",
        comparison_keys=("control",),
        min_reference=0.01,
    )
    txt_path = tmp_path / "data_txt" / "spatial_relative_bias_vs_radar.txt"

    write_bias_txt(
        txt_path,
        figure_path=tmp_path / "spatial_relative_bias_vs_radar.png",
        data_dir=tmp_path,
        fields=fields,
        bias_maps=bias_maps,
        reference_key="radar",
        min_reference=0.01,
        vmin=-30.0,
        vmax=30.0,
    )

    text = txt_path.read_text(encoding="utf-8")
    assert "Spatial Relative Bias Map Data" in text
    assert "Panel summary" in text
    assert "Gridpoint data" in text
    assert "C1M,Radar" in text


def test_write_spatial_mean_txt_includes_gridpoint_data(tmp_path: Path):
    fields = {
        "radar": _field("radar", [[1.0, 2.0], [3.0, 4.0]]),
        "imerg": _field("imerg", [[2.0, 3.0], [4.0, 5.0]]),
    }
    txt_path = tmp_path / "data_txt" / "spatial_mean_rainfall_maps.txt"

    write_spatial_mean_txt(
        txt_path,
        figure_path=tmp_path / "spatial_mean_rainfall_maps.png",
        data_dir=tmp_path,
        fields=fields,
        vmin=0.0,
        vmax=5.0,
    )

    text = txt_path.read_text(encoding="utf-8")
    assert "Spatial Mean Rainfall Map Data" in text
    assert "Panel summary" in text
    assert "Gridpoint data" in text
    assert "Radar" in text
