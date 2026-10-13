"""Masked fractions skill score: agreement with nwpeval and the footprint mask."""

import numpy as np
import pytest

from alaro_analysis.analysis.fractions_skill import (
    fss_components,
    fss_from_components,
    neighbourhood_fractions,
    useful_skill,
)


def _fss(obs, model, valid, size, score=None):
    return fss_from_components(*fss_components(obs, model, valid, size, score))


def test_perfect_forecast_scores_one():
    rng = np.random.default_rng(1)
    obs = (rng.random((4, 9, 8)) > 0.8).astype(float)
    valid = np.ones((9, 8), bool)
    assert _fss(obs, obs, valid, 3) == pytest.approx(1.0)


def test_matches_nwpeval_on_a_footprint():
    xr = pytest.importorskip("xarray")
    nwpeval = pytest.importorskip("nwpeval")
    rng = np.random.default_rng(0)
    obs = rng.gamma(0.4, 2.0, size=(6, 12, 10))
    model = rng.gamma(0.4, 2.0, size=(6, 12, 10))
    footprint = np.ones((12, 10), bool)
    footprint[:3, :4] = False
    footprint[:, -2:] = False
    dims = ("time", "lat", "lon")
    fp = xr.DataArray(footprint, dims=dims[1:])
    for size in (1, 3, 5, 15):
        ours = _fss((obs >= 1.0).astype(float), (model >= 1.0).astype(float), footprint, size)
        theirs = float(
            nwpeval.fss(
                xr.DataArray(obs, dims=dims).where(fp),
                xr.DataArray(model, dims=dims).where(fp),
                1.0,
                size,
            )
        )
        assert ours == pytest.approx(theirs, rel=1e-12)


def test_rain_outside_the_footprint_does_not_count():
    obs = np.zeros((3, 9, 9))
    obs[:, 4, 4] = 1.0
    model = obs.copy()
    model[:, :, 7:] = 1.0  # rain only in the last two columns
    footprint = np.ones((9, 9), bool)
    footprint[:, 7:] = False

    assert _fss(obs, model, footprint, 3) == pytest.approx(1.0)
    # Scored over the whole box instead, the stray rain costs skill.
    assert _fss(obs, model, np.ones((9, 9), bool), 3) < 1.0


def test_fractions_use_valid_neighbours_only():
    binary = np.array([[1.0, 0.0, 0.0]])
    valid = np.array([[True, True, False]])
    fractions = neighbourhood_fractions(binary, valid, 3)
    # Centre window holds two valid cells, one of them wet.
    assert fractions[0, 1] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        neighbourhood_fractions(binary, valid, 2)


def test_useful_skill_level():
    assert useful_skill(0.1) == pytest.approx(0.55)
