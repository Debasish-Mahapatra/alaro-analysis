from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from alaro_analysis.analysis.diagnostic_profiles import align_tlyx_shapes
from alaro_analysis.data.dataset_io import (
    nanmean_with_count,
    resolve_data_var_name,
    to_time_level_yx,
)


def test_wrong_single_variable_requires_explicit_legacy_choice():
    dataset = xr.Dataset({"RAIN": ("x", [1.0])})
    with pytest.raises(KeyError, match="TEMPERATURE"):
        resolve_data_var_name(dataset, "TEMPERATURE")
    assert resolve_data_var_name(dataset, "TEMPERATURE", allow_single_variable=True) == "RAIN"


def test_named_dimensions_are_transposed_before_reduction():
    original = np.arange(120).reshape(2, 3, 4, 5)
    reordered = original.transpose(3, 1, 0, 2)
    result = to_time_level_yx(reordered, ("x", "level", "time", "y"), Path("sample.nc"), "RAIN")
    np.testing.assert_array_equal(result, original)


def test_unknown_axes_and_mismatched_fields_are_rejected():
    with pytest.raises(ValueError, match="Ambiguous"):
        to_time_level_yx(np.zeros((2, 3, 4)), ("a", "b", "c"), Path("x.nc"), "RAIN")
    with pytest.raises(ValueError, match="Incompatible"):
        align_tlyx_shapes([np.zeros((1, 2, 3, 4)), np.zeros((1, 3, 3, 4))])


def test_mean_and_count_use_the_same_finite_population():
    mean, counts = nanmean_with_count(np.array([[1.0, np.inf, np.nan, 3.0]]), (1,))
    np.testing.assert_array_equal(counts, [2])
    np.testing.assert_array_equal(mean, [2.0])


def test_vertical_profile_uses_named_axes_and_coordinate(tmp_path):
    from alaro_analysis.data.dataset_io import read_vertical_profile

    path = tmp_path / "profile.nc"
    values = np.arange(12.0).reshape(2, 3, 2, 1)  # x, level, y, time
    xr.Dataset(
        {"T": (("x", "level", "y", "time"), values)}, coords={"level": [10, 20, 30]}
    ).to_netcdf(path)
    mean, coord = read_vertical_profile(path, "T")
    np.testing.assert_allclose(mean, values.mean(axis=(0, 2, 3)))
    np.testing.assert_array_equal(coord, [10, 20, 30])


def test_masked_field_reader_requires_one_time_step(tmp_path):
    from alaro_analysis.data.masked_fields import read_field

    path = tmp_path / "RAIN" / "pf20140101" / "pfABOFABOF+0000.nc"
    path.parent.mkdir(parents=True)
    xr.Dataset({"RAIN": (("x", "level", "y", "time"), np.ones((2, 3, 2, 1)))}).to_netcdf(path)
    assert read_field(path).shape == (3, 2, 2)
    xr.Dataset({"RAIN": (("time", "level", "y", "x"), np.ones((2, 3, 2, 2)))}).to_netcdf(path)
    with pytest.raises(ValueError, match="one time step"):
        read_field(path)
