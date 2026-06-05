from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from alaro_analysis.workflows.precip_rebuild_hourly_imerg_common_valid import (
    apply_spatial_mask,
    build_radar_mask,
    clean_precipitation,
    hourly_mean_from_halfhours,
    hourly_times,
    intersect_time_values,
    needed_halfhour_times,
    parse_imerg_start_time,
    valid_domain_mean_times,
)


def test_parse_imerg_start_time_from_filename():
    path = Path("3B-HHR.MS.MRG.3IMERG.20140101-S003000-E005959.0030.V07B.HDF5.nc4")

    assert parse_imerg_start_time(path) == np.datetime64("2014-01-01T00:30:00", "ns")


def test_hourly_and_needed_halfhour_times():
    times = hourly_times("2014-01-01T00:00:00", "2014-01-01T02:00:00")

    np.testing.assert_array_equal(
        times,
        np.asarray(
            [
                "2014-01-01T00:00:00",
                "2014-01-01T01:00:00",
                "2014-01-01T02:00:00",
            ],
            dtype="datetime64[ns]",
        ),
    )
    np.testing.assert_array_equal(
        needed_halfhour_times(times),
        np.asarray(
            [
                "2014-01-01T00:00:00",
                "2014-01-01T00:30:00",
                "2014-01-01T01:00:00",
                "2014-01-01T01:30:00",
                "2014-01-01T02:00:00",
                "2014-01-01T02:30:00",
            ],
            dtype="datetime64[ns]",
        ),
    )


def test_clean_precipitation_and_hourly_mean():
    first = np.asarray([[1.0, -9999.9], [np.nan, 4.0]], dtype=np.float32)
    second = np.asarray([[3.0, 5.0], [7.0, -9999.9]], dtype=np.float32)

    cleaned = clean_precipitation(first)
    assert np.isnan(cleaned[0, 1])
    np.testing.assert_allclose(
        hourly_mean_from_halfhours(first, second),
        np.asarray([[2.0, 5.0], [7.0, 4.0]], dtype=np.float32),
    )


def test_spatial_mask_and_valid_domain_times():
    times = np.asarray(
        ["2014-01-01T00:00", "2014-01-01T01:00", "2014-01-01T02:00"],
        dtype="datetime64[ns]",
    )
    rain = xr.DataArray(
        np.asarray(
            [
                [[1.0, np.nan], [np.nan, np.nan]],
                [[2.0, np.nan], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ]
        ),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [0.0, 1.0], "lon": [10.0, 11.0]},
    )
    ds = xr.Dataset(
        {
            "rainfall_rate": rain,
            "valid_time_mask": ("time", np.asarray([1, 0, 1])),
        }
    )

    mask = build_radar_mask(ds)
    np.testing.assert_array_equal(mask.values, np.asarray([[True, False], [False, False]]))

    masked = apply_spatial_mask(ds, mask)
    assert "radar_mask" in masked
    valid = valid_domain_mean_times(masked, "rainfall_rate", use_radar_flag=True)
    np.testing.assert_array_equal(valid, times[:1])


def test_intersect_time_values_sorts_common_times():
    first = np.asarray(["2014-01-01T02", "2014-01-01T00"], dtype="datetime64[ns]")
    second = np.asarray(["2014-01-01T00", "2014-01-01T03"], dtype="datetime64[ns]")

    np.testing.assert_array_equal(
        intersect_time_values([first, second]),
        np.asarray(["2014-01-01T00"], dtype="datetime64[ns]"),
    )
