from __future__ import annotations

from datetime import datetime

import numpy as np

from alaro_analysis.common.constants import G
from alaro_analysis.workflows.updraft_mass_flux_timeseries import (
    UpdraftFluxRecord,
    aggregate_daily_records,
    rank_daily_top_days,
    rank_top_days,
    select_top_percentile_days,
    summarize_flux,
    updraft_mass_flux,
    valid_time_from_file,
)


def test_valid_time_from_file_ignores_duplicate_24_hour_lead():
    assert valid_time_from_file("pf20140102", "pfABOFABOF+0003.nc") == datetime(2014, 1, 2, 3)
    assert valid_time_from_file("pf20140102", "pfABOFABOF+0024.nc") is None


def test_updraft_mass_flux_is_positive_upward_and_zero_when_inactive():
    omega = np.array([[[-G, G], [-2.0 * G, -G]]])
    mesh = np.array([[[0.5, 0.5], [0.0, np.nan]]])

    flux = updraft_mass_flux(omega, mesh)

    np.testing.assert_allclose(flux, np.array([[[0.5, -0.5], [0.0, 0.0]]]))


def test_summarize_flux_domain_mean_and_top_percentile_contribution():
    flux = np.array([1.0, 2.0, 3.0, 4.0, 0.0, np.nan])
    finite = np.array([True, True, True, True, True, False])

    (
        domain_mean,
        active_p50,
        top_domain_mean,
        top_active_mean,
        active_mean,
        active_fraction,
        finite_count,
        active_count,
        top_count,
    ) = summarize_flux(flux, finite, percentile=50.0)

    assert domain_mean == 2.0
    assert active_p50 == 2.5
    assert top_domain_mean == (3.0 + 4.0) / 5.0
    assert top_active_mean == 3.5
    assert active_mean == 2.5
    assert active_fraction == 4 / 5
    assert finite_count == 5
    assert active_count == 4
    assert top_count == 2


def test_rank_top_days_keeps_one_strongest_hour_per_day():
    rows = [
        UpdraftFluxRecord("graupel", datetime(2014, 1, 1, 1), "pf20140101", "a.nc", 1, 1, 1, 1, 1, 0.1, 10, 1, 1),
        UpdraftFluxRecord("graupel", datetime(2014, 1, 1, 2), "pf20140101", "b.nc", 1, 1, 5, 1, 1, 0.1, 10, 1, 1),
        UpdraftFluxRecord("graupel", datetime(2014, 1, 2, 1), "pf20140102", "c.nc", 1, 1, 3, 1, 1, 0.1, 10, 1, 1),
    ]

    ranked = rank_top_days(rows, metric="top10_domain_mean_kg_m2_s", top_days=2)

    assert [row.day for row in ranked] == ["pf20140101", "pf20140102"]
    assert [row.filename for row in ranked] == ["b.nc", "c.nc"]


def test_select_top_percentile_days_uses_daily_means_not_hourly_peaks():
    rows = [
        UpdraftFluxRecord("graupel", datetime(2014, 1, 1, 0), "pf20140101", "a.nc", 0, 0, 1, 0, 0, 0.1, 10, 1, 1),
        UpdraftFluxRecord("graupel", datetime(2014, 1, 1, 1), "pf20140101", "b.nc", 0, 0, 3, 0, 0, 0.1, 10, 1, 1),
        UpdraftFluxRecord("graupel", datetime(2014, 1, 2, 0), "pf20140102", "c.nc", 0, 0, 10, 0, 0, 0.1, 10, 1, 1),
        UpdraftFluxRecord("graupel", datetime(2014, 1, 2, 1), "pf20140102", "d.nc", 0, 0, 0, 0, 0, 0.1, 10, 1, 1),
        UpdraftFluxRecord("graupel", datetime(2014, 1, 3, 0), "pf20140103", "e.nc", 0, 0, 4, 0, 0, 0.1, 10, 1, 1),
        UpdraftFluxRecord("graupel", datetime(2014, 1, 3, 1), "pf20140103", "f.nc", 0, 0, 4, 0, 0, 0.1, 10, 1, 1),
    ]

    daily = aggregate_daily_records(rows)
    selected, threshold = select_top_percentile_days(
        daily,
        metric="daily_mean_top10_domain_mean_kg_m2_s",
        day_percentile=90.0,
    )

    assert threshold > 4.0
    assert [row.day for row in selected if row.selected] == ["pf20140102"]
    assert daily[1].daily_mean_top10_domain_mean_kg_m2_s == 5.0


def test_rank_daily_top_days_uses_daily_mean_metric():
    rows = [
        UpdraftFluxRecord("graupel", datetime(2014, 1, 1, 0), "pf20140101", "a.nc", 0, 0, 100, 0, 0, 0.1, 10, 1, 1),
        UpdraftFluxRecord("graupel", datetime(2014, 1, 1, 1), "pf20140101", "b.nc", 0, 0, 0, 0, 0, 0.1, 10, 1, 1),
        UpdraftFluxRecord("graupel", datetime(2014, 1, 2, 0), "pf20140102", "c.nc", 0, 0, 60, 0, 0, 0.1, 10, 1, 1),
        UpdraftFluxRecord("graupel", datetime(2014, 1, 2, 1), "pf20140102", "d.nc", 0, 0, 60, 0, 0, 0.1, 10, 1, 1),
    ]

    daily = aggregate_daily_records(rows)
    ranked = rank_daily_top_days(
        daily,
        metric="daily_mean_top10_domain_mean_kg_m2_s",
        top_days=1,
    )

    assert [row.day for row in ranked] == ["pf20140102"]
