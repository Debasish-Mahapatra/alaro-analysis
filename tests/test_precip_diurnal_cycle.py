from __future__ import annotations

import numpy as np

from alaro_analysis.workflows.precip_diurnal_cycle import (
    compute_hourly_stats,
    local_hours_from_utc,
    shading_bounds,
)


def test_local_hours_apply_utc_minus_4_offset():
    times = np.array(
        ["2014-01-01T00:30:00", "2014-01-01T04:00:00", "2014-01-01T23:00:00"],
        dtype="datetime64[s]",
    )

    hours = local_hours_from_utc(times, -4)

    np.testing.assert_array_equal(hours, [20, 0, 19])


def test_hourly_stats_group_by_local_hour():
    values = np.array([1.0, 3.0, 10.0, np.nan, 5.0])
    hours = np.array([0, 0, 1, 1, 23])

    stats = compute_hourly_stats(values, hours)

    assert stats.count[0] == 2
    assert stats.mean[0] == 2.0
    assert np.isclose(stats.std[0], np.sqrt(2.0))
    assert stats.count[1] == 1
    assert stats.mean[1] == 10.0
    assert np.isnan(stats.std[1])
    assert stats.count[23] == 1
    assert stats.mean[23] == 5.0


def test_percent_shading_bounds_are_clipped_nonnegative():
    stats = compute_hourly_stats(np.array([1.0, 3.0]), np.array([0, 0]))

    lower, upper = shading_bounds(
        stats,
        mode="percent",
        std_multiplier=1.0,
        percent_uncertainty=0.10,
    )

    assert np.isclose(lower[0], 1.8)
    assert np.isclose(upper[0], 2.2)
