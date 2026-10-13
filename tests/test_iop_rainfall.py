"""Combined-IOP rainfall verification: the hourly window and JPEG export."""

import numpy as np

from alaro_analysis.recipes.iop.build_iop_rainfall_common_valid import hourly_axis, iop_days


def test_combined_iop_hours_cover_exactly_both_campaign_windows():
    days = iop_days()
    times = hourly_axis(days)
    assert len(days) == 59 + 62
    assert times.size == 24 * len(days)
    assert times[0] == np.datetime64("2014-02-01T00")
    assert times[-1] == np.datetime64("2014-10-15T23")
    gap = (times > np.datetime64("2014-03-31T23")) & (times < np.datetime64("2014-08-15T00"))
    assert not gap.any()
    assert np.all(np.diff(times[: 24 * 59]) == np.timedelta64(1, "h"))


def test_split_by_iop_keeps_exactly_each_campaign_window(tmp_path):
    import xarray as xr

    from alaro_analysis.recipes.iop.build_iop_rainfall_common_valid import (
        COMMON_SUBDIR,
        split_by_iop,
    )

    # One hour just outside each end of both windows, plus hours inside them.
    stamps = [
        "2014-01-31T23",
        "2014-02-01T00",
        "2014-03-31T23",
        "2014-04-01T00",
        "2014-08-14T23",
        "2014-08-15T00",
        "2014-09-20T12",
        "2014-10-15T23",
        "2014-10-16T00",
    ]
    times = np.array(stamps, dtype="datetime64[ns]")
    source = tmp_path / COMMON_SUBDIR
    source.mkdir()
    rain = xr.Dataset({"rainfall": (("time", "lat", "lon"), np.ones((times.size, 2, 3)))})
    rain.assign_coords(time=times).to_netcdf(source / "Radar_common_valid.nc")

    counts = split_by_iop(tmp_path)

    assert counts == {"iop1_split_hours": 2, "iop2_split_hours": 3}
    with xr.open_dataset(tmp_path / f"{COMMON_SUBDIR}-iop2" / "Radar_common_valid.nc") as ds:
        kept = np.asarray(ds["time"].values, dtype="datetime64[h]")
    assert kept.tolist() == np.array(stamps[5:8], dtype="datetime64[h]").tolist()


def test_colour_limit_takes_p98_of_one_figure_and_ignores_missing_cells():
    from alaro_analysis.recipes.iop.make_iop_rainfall_verification import colour_limit

    assert colour_limit([np.linspace(-20.0, 20.0, 101)]) == 20.0
    assert colour_limit([np.r_[np.full(50, 250.0), np.nan], np.full(50, 5.0)]) == 250.0
    assert colour_limit([np.full(4, np.nan)]) == 30.0


def test_convert_to_jpg_keeps_dpi_and_drops_alpha(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from alaro_analysis.common.figio import convert_to_jpg

    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([0, 1])
    png = tmp_path / "figure.png"
    fig.savefig(png, dpi=100, transparent=True)
    plt.close(fig)
    jpg = convert_to_jpg(png, tmp_path / "figure.jpg", dpi=450)
    with Image.open(jpg) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert round(image.info["dpi"][0]) == 450


def test_roi_common_valid_drops_incomplete_hours_and_splits_by_iop(tmp_path):
    import xarray as xr

    from alaro_analysis.recipes.iop.build_iop_rainfall_roi import (
        COMMON_SUBDIR,
        DATASETS,
        IMERG_FILE,
        build_common_valid,
    )

    times = np.array(
        ["2014-02-01T00", "2014-02-01T01", "2014-08-15T00", "2014-08-15T01"], dtype="datetime64[ns]"
    )
    lat, lon = np.array([-1.05, -0.95]), np.array([-60.05, -59.95, -59.85])
    xr.Dataset(coords={"lat": lat, "lon": lon}).to_netcdf(tmp_path / "imerg_grid_sample.nc")
    for name in DATASETS:
        values = np.ones((times.size, lat.size, lon.size), dtype=np.float32)
        if name == "G2M":
            values[1, 0, 2] = np.nan  # one missing cell drops the whole hour
        path, var = (IMERG_FILE, "precipitation") if name == "IMERG" else (f"{name}_to_imerg.nc", "total_rain")
        ds = xr.Dataset({var: (("time", "lat", "lon"), values)}, coords={"time": times, "lat": lat, "lon": lon})
        ds.to_netcdf(tmp_path / path)

    counts = build_common_valid(tmp_path, times)

    assert counts["common_valid"] == 3
    assert counts["complete_G2M"] == 3 and counts["complete_IMERG"] == 4
    assert (counts["iop1_common_valid"], counts["iop2_common_valid"]) == (1, 2)
    for subdir, n in ((COMMON_SUBDIR, 3), (f"{COMMON_SUBDIR}-iop1", 1), (f"{COMMON_SUBDIR}-iop2", 2)):
        for name in DATASETS:
            with xr.open_dataset(tmp_path / subdir / f"{name}_common_valid.nc") as ds:
                assert ds["rainfall"].shape == (n, 2, 3)
                assert np.isfinite(ds["rainfall"].values).all()


def test_roi_pdf_uses_imerg_as_the_only_reference():
    from alaro_analysis.recipes.iop.make_iop_rain_pdf_roi import DATASETS

    assert [cfg.label for cfg in DATASETS] == ["IMERG", "C1M", "G1M", "G2M", "G2M-XCU"]
    assert DATASETS[0].color == "black" and DATASETS[0].linestyle == "-"
