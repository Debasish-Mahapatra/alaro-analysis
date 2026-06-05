from __future__ import annotations

import numpy as np

from alaro_analysis.workflows import dsd_cfad_profiles as cfad


def test_add_values_to_hist_counts_by_level():
    hist = np.zeros((3, 4), dtype=np.int64)
    edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    cfad.add_values_to_hist(hist, np.array([0.2, 2.5, np.nan]), edges)
    cfad.add_values_to_hist(hist, np.array([1.2, 4.5, -1.0]), edges)

    np.testing.assert_array_equal(
        hist,
        np.array(
            [
                [1, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ]
        ),
    )


def test_frequency_percent_normalizes_each_level():
    hist = np.array([[1, 3], [0, 0], [2, 2]], dtype=np.int64)

    freq = cfad.frequency_percent(hist)

    np.testing.assert_allclose(freq[0], [25.0, 75.0])
    assert np.all(np.isnan(freq[1]))
    np.testing.assert_allclose(freq[2], [50.0, 50.0])


def test_add_dsd_profile_to_accumulator_initializes_shapes():
    x_edges = {
        "d0_mm": np.array([0.0, 1.0, 2.0]),
        "log_nw": np.array([0.0, 5.0, 10.0]),
    }
    profile = {
        "height_km": np.array([0.5, 1.0]),
        "temperature_k": np.array([290.0, 280.0]),
        "d0_mm": np.array([0.8, 1.2]),
        "log_nw": np.array([4.0, 6.0]),
    }

    acc: dict[str, object] = {}
    cfad.add_dsd_profile_to_accumulator(acc, profile, x_edges)

    np.testing.assert_array_equal(acc["hist"]["d0_mm"], [[1, 0], [0, 1]])
    np.testing.assert_array_equal(acc["hist"]["log_nw"], [[1, 0], [0, 1]])
    assert acc["n_profiles"] == 1
