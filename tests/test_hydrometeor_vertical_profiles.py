from __future__ import annotations

from pathlib import Path

import numpy as np

from alaro_analysis.workflows import hydrometeor_vertical_profiles as hvp


def test_profile_from_diurnal_uses_counts_as_weights():
    mean = np.array(
        [
            [1.0, 3.0, np.nan],
            [10.0, 20.0, 30.0],
        ]
    )
    counts = np.array(
        [
            [1, 3, 0],
            [0, 2, 1],
        ]
    )

    profile, level_counts = hvp.profile_from_diurnal(mean, counts)

    assert np.allclose(profile, [2.5, 70.0 / 3.0])
    assert np.array_equal(level_counts, [4, 3])


def test_load_cached_profile_supports_legacy_diurnal_cache(tmp_path: Path):
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        mean=np.array([[1.0, 5.0], [2.0, np.nan]]),
        counts=np.array([[1, 3], [2, 0]]),
        n_files=np.array([4]),
        sample_file=np.array(["sample.nc"]),
    )

    profile = hvp.load_cached_profile(path)

    assert np.allclose(profile.profile, [4.0, 2.0])
    assert np.array_equal(profile.counts, [4, 2])
    assert profile.n_files == 4
    assert profile.source == str(path)


def test_load_height_cache_converts_meters_to_km(tmp_path: Path):
    path = tmp_path / "height.npz"
    np.savez(path, height_m=np.array([0.0, 1000.0, 2500.0]))

    height = hvp.load_height_cache(path)

    assert np.allclose(height, [0.0, 1.0, 2.5])


def test_write_data_txt_includes_plot_columns(tmp_path: Path):
    profiles = {}
    heights = {}
    height_sources = {}
    for experiment in hvp.EXPERIMENTS:
        profiles[experiment] = {}
        heights[experiment] = np.array([0.0, 1.0])
        height_sources[experiment] = f"{experiment}_height.npz"
        for idx, variable in enumerate(hvp.SPECIES, start=1):
            profiles[experiment][variable] = hvp.ProfileData(
                profile=np.array([idx * 1.0e-6, idx * 2.0e-6]),
                counts=np.array([10, 20]),
                n_files=20,
                source=f"{experiment}_{variable}.npz",
            )

    out = tmp_path / "data.txt"
    hvp.write_data_txt(out, profiles, heights, height_sources)

    text = out.read_text(encoding="utf-8")
    assert "Hydrometeor vertical profiles: C1M, G1M, G2M" in text
    assert "experiment,experiment_label,level_index,height_km" in text
    assert "LIQUID_WATER,SOLID_WATER,GRAUPEL,SNOW,RAIN" in text
    assert "control,C1M,0,0.0000000000e+00" in text
