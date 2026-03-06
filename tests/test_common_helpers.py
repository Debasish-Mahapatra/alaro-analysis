from pathlib import Path

import numpy as np

from alaro_analysis.common.naming import safe_name
from alaro_analysis.common.spatial import build_spatial_window, parse_slice_arg, spatial_window_tag
from alaro_analysis.common.timeparse import (
    has_pf_subdirs,
    parse_month_from_day_name,
    parse_utc_hour_from_name,
)
from alaro_analysis.common.vertical import centers_to_edges


def test_safe_name():
    assert safe_name("KT273HUMI.SPECIF") == "kt273humi_specif"


def test_parse_utc_hour_from_name():
    assert parse_utc_hour_from_name("pfABOFABOF+0001.nc") == 1
    assert parse_utc_hour_from_name("pfABOFABOF+0024.nc") is None


def test_parse_month_from_day_name():
    assert parse_month_from_day_name("pf20141231") == 12
    assert parse_month_from_day_name("foo") is None


def test_has_pf_subdirs(tmp_path: Path):
    assert has_pf_subdirs(tmp_path) is False
    (tmp_path / "pf20140101").mkdir()
    assert has_pf_subdirs(tmp_path) is True


def test_parse_slice_arg_and_window():
    assert parse_slice_arg("10:20", "x") == (10, 20)
    assert parse_slice_arg(":20", "x") == (None, 20)
    assert parse_slice_arg("10:", "x") == (10, None)

    window = build_spatial_window("10:20", "30:40")
    assert (window.y_start, window.y_end, window.x_start, window.x_end) == (10, 20, 30, 40)
    assert spatial_window_tag(window) == "y10-20_x30-40"


def test_centers_to_edges():
    edges = centers_to_edges(np.array([1.0, 3.0, 5.0]))
    np.testing.assert_allclose(edges, np.array([0.0, 2.0, 4.0, 6.0]))
