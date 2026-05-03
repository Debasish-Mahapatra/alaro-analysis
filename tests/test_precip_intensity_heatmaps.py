from __future__ import annotations

from pathlib import Path

import numpy as np

from alaro_analysis.workflows.precip_intensity_heatmaps import (
    DATASETS,
    MODEL_DATASETS,
    compute_intensity_hour_histogram,
    local_hours_from_utc,
    write_convective_stratiform_txt,
    write_intensity_evolution_txt,
)


def test_local_hours_from_utc_applies_offset():
    times = np.array(
        ["2014-01-01T00:00:00", "2014-01-01T03:00:00", "2014-01-01T23:00:00"],
        dtype="datetime64[ns]",
    )

    hours = local_hours_from_utc(times, -4)

    np.testing.assert_array_equal(hours, np.array([20, 23, 19], dtype=np.int16))


def test_compute_intensity_hour_histogram_counts_wet_pixels():
    values = np.array(
        [
            [[0.05, 0.2], [2.0, np.nan]],
            [[0.4, 4.0], [40.0, 400.0]],
        ]
    )
    times = np.array(["2014-01-01T04:00:00", "2014-01-01T05:00:00"], dtype="datetime64[ns]")
    bins = np.array([0.1, 1.0, 10.0, 100.0])

    histogram = compute_intensity_hour_histogram(
        values,
        times,
        bins,
        wet_threshold=0.1,
        utc_offset_hours=-4,
    )

    assert histogram.shape == (3, 24)
    assert histogram[0, 0] == 1
    assert histogram[1, 0] == 1
    assert histogram[0, 1] == 1
    assert histogram[1, 1] == 1
    assert histogram[2, 1] == 1
    assert histogram.sum() == 5


def test_write_intensity_evolution_txt_includes_heatmap_data(tmp_path: Path):
    bins = np.array([0.1, 1.0, 10.0])
    radar = np.zeros((2, 24))
    radar[0, 0] = 3
    heatmaps = {cfg.key: np.zeros((2, 24)) for cfg in DATASETS}
    heatmaps["radar"] = radar
    heatmaps["control"][1, 1] = 5
    paths = {cfg.key: tmp_path / cfg.filename for cfg in DATASETS}
    txt_path = tmp_path / "data_txt" / "intensity_evolution.txt"

    write_intensity_evolution_txt(
        txt_path,
        figure_path=tmp_path / "intensity_evolution_pixel_level_both_contours.png",
        data_dir=tmp_path,
        paths=paths,
        heatmaps=heatmaps,
        intensity_bins=bins,
        wet_threshold=0.1,
        utc_offset_hours=-4,
    )

    text = txt_path.read_text(encoding="utf-8")
    assert "Intensity Evolution Pixel-Level Heatmap Data" in text
    assert "Heatmap data" in text
    assert "difference_vs_radar" in text


def test_write_convective_stratiform_txt_includes_components(tmp_path: Path):
    bins = np.array([0.1, 1.0, 10.0])
    heatmaps = {
        cfg.key: {
            "convective": np.ones((2, 24)),
            "stratiform": np.ones((2, 24)) * 2,
        }
        for cfg in MODEL_DATASETS
    }
    paths = {cfg.key: tmp_path / cfg.filename for cfg in MODEL_DATASETS}
    txt_path = tmp_path / "data_txt" / "convective_stratiform.txt"

    write_convective_stratiform_txt(
        txt_path,
        figure_path=tmp_path / "convective_stratiform_intensity_hour_heatmap.png",
        data_dir=tmp_path,
        paths=paths,
        heatmaps=heatmaps,
        intensity_bins=bins,
        wet_threshold=0.1,
        utc_offset_hours=-4,
        vmax=2000.0,
    )

    text = txt_path.read_text(encoding="utf-8")
    assert "Convective/Stratiform Intensity-Hour Heatmap Data" in text
    assert "convective" in text
    assert "stratiform" in text
