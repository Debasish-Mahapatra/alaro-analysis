"""IOP hydrometeor histograms: binning of log10 mass fraction, with no normalisation."""

import numpy as np

from alaro_analysis.data.hydrometeor_histograms import (
    DBIN,
    LOG_EDGES,
    NBINS,
    histogram_levels,
    species_for,
)


def test_values_land_in_their_log10_bin_and_counts_stay_raw():
    # One level, four points: two at 1e-6, one at 1e-3, one zero.
    values = np.array([[1e-6, 1e-6, 1e-3, 0.0]])

    counts = histogram_levels(values)

    six = int(np.floor((-6.0 - LOG_EDGES[0]) / DBIN))
    three = int(np.floor((-3.0 - LOG_EDGES[0]) / DBIN))
    assert counts[0, six] == 2
    assert counts[0, three] == 1
    # No normalisation: the row is a count, not a fraction of the level.
    assert counts[0].sum() == 3


def test_masked_zero_and_out_of_range_values_are_dropped():
    values = np.array([[np.nan, 0.0, -1e-6, 1e-20, 1.0, 1e-5]])

    counts = histogram_levels(values)

    # Only 1e-5 is positive and inside the 1e-11..1e-1 range.
    assert counts.sum() == 1
    assert counts[0, int(np.floor((-5.0 - LOG_EDGES[0]) / DBIN))] == 1
    assert counts.shape[1] == NBINS


def test_control_has_no_prognostic_graupel():
    assert "GRAUPEL" not in species_for("control")
    for experiment in ("graupel", "2mom", "2mom-xcu"):
        assert "GRAUPEL" in species_for(experiment)
