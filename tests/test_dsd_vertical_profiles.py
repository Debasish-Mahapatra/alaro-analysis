from __future__ import annotations

from pathlib import Path

import numpy as np

from alaro_analysis.workflows import dsd_vertical_profiles as dvp


def test_ddh_lead_defaults_to_single_dsd_lead():
    assert dvp.ddh_lead_text((12,), None) == "0024"
    assert dvp.ddh_lead_text(None, None) == "0024"
    assert dvp.ddh_lead_text((1, 2), "7") == "0007"


def test_add_partial_and_profile_average():
    first = dvp.empty_partial(2)
    first["sums"]["d0_mm"][:] = [1.0, 4.0]
    first["counts"]["d0_mm"][:] = [1, 1]
    first["sums"]["log_nw"][:] = [3.0, 8.0]
    first["counts"]["log_nw"][:] = [1, 1]
    first["height_sum"][:] = [0.1, 1.0]
    first["height_count"][:] = [1, 1]
    first["temperature_sum"][:] = [300.0, 270.0]
    first["temperature_count"][:] = [1, 1]
    first["n_files"] = 1

    second = dvp.empty_partial(2)
    second["sums"]["d0_mm"][:] = [3.0, 0.0]
    second["counts"]["d0_mm"][:] = [1, 0]
    second["sums"]["log_nw"][:] = [5.0, 0.0]
    second["counts"]["log_nw"][:] = [1, 0]
    second["height_sum"][:] = [0.3, 1.2]
    second["height_count"][:] = [1, 1]
    second["temperature_sum"][:] = [302.0, 272.0]
    second["temperature_count"][:] = [1, 1]
    second["n_files"] = 1

    total: dict[str, object] = {}
    dvp.add_partial(total, first)
    dvp.add_partial(total, second)
    profile = dvp.profile_from_accumulator(total, source="synthetic")

    np.testing.assert_allclose(profile.height_km, [0.2, 1.1])
    np.testing.assert_allclose(profile.temperature_k, [301.0, 271.0])
    np.testing.assert_allclose(profile.values["d0_mm"], [2.0, 4.0])
    np.testing.assert_allclose(profile.values["log_nw"], [4.0, 8.0])
    assert profile.n_files == 2


def test_load_ddh_qv_profile_sorts_by_height(tmp_path: Path):
    lead_dir = tmp_path / "lead0024_VZ"
    lead_dir.mkdir()
    path = lead_dir / "control_QV.npz"
    np.savez(
        path,
        altitude_km=np.array([2.0, 0.5, 1.0]),
        days=np.array(["DDH20140101", "DDH20140102"]),
        block__VQVM=np.array([20.0, 5.0, 10.0]),
    )

    profile = dvp.load_ddh_qv_profile(
        tmp_path,
        "control",
        lead_text_value="0024",
        block="VQVM",
    )

    np.testing.assert_allclose(profile.height_km, [0.5, 1.0, 2.0])
    np.testing.assert_allclose(profile.values, [5.0, 10.0, 20.0])
    assert profile.n_days == 2
    assert profile.source == str(path)


def test_dsd_profile_cache_roundtrip(tmp_path: Path):
    profile = dvp.DsdVerticalProfile(
        height_km=np.array([0.1, 1.0]),
        temperature_k=np.array([298.0, 270.0]),
        temperature_count=np.array([3, 3]),
        values={
            "d0_mm": np.array([0.5, 0.7]),
            "log_nw": np.array([3.0, 4.0]),
        },
        counts={
            "d0_mm": np.array([2, 3]),
            "log_nw": np.array([2, 3]),
        },
        n_files=3,
        source="synthetic",
    )
    path = tmp_path / "profile.npz"

    dvp.save_dsd_profile(path, profile)
    loaded = dvp.load_dsd_profile(path)

    np.testing.assert_allclose(loaded.height_km, profile.height_km)
    np.testing.assert_allclose(loaded.temperature_k, profile.temperature_k)
    np.testing.assert_allclose(loaded.values["d0_mm"], profile.values["d0_mm"])
    np.testing.assert_array_equal(loaded.temperature_count, profile.temperature_count)
    np.testing.assert_array_equal(loaded.counts["log_nw"], profile.counts["log_nw"])
    assert loaded.n_files == 3
    assert loaded.source == "synthetic"


def test_freezing_level_interpolates_lowest_crossing():
    height = np.array([0.0, 1.0, 2.0, 3.0])
    temperature = np.array([290.0, 280.0, 270.0, 260.0])

    assert np.isclose(dvp.freezing_level_km(height, temperature), 1.685)
