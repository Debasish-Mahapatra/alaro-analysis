from __future__ import annotations

import math

import numpy as np

from alaro_analysis.workflows.disdrometer_comparison import (
    ObservationSeries,
    average_observation_window,
    datetime64_from_seconds_since,
    discover_model_records,
    integrate_drop_number,
    lead_label,
    parse_lead_selection,
    rain_mean_volume_diameter_mm,
    single_moment_rain_number_per_kg,
    summarize_experiment,
)


def test_integrate_drop_number_with_static_widths():
    density = np.array(
        [
            [10.0, 20.0, 30.0],
            [1.0, np.nan, 3.0],
        ]
    )
    widths = np.array([0.1, 0.2, 0.3])

    np.testing.assert_allclose(integrate_drop_number(density, widths), [14.0, 1.0])


def test_integrate_drop_number_with_time_varying_transposed_widths():
    density = np.array([[10.0, 20.0], [5.0, 6.0]])
    widths = np.array([[0.1, 0.3], [0.2, 0.4]])

    np.testing.assert_allclose(integrate_drop_number(density, widths.T), [5.0, 3.9])


def test_single_moment_rain_number_uses_model_equilibrium_diameter():
    qr = np.array([-1.0e-6, 0.0, 2.0e-6])
    result = single_moment_rain_number_per_kg(qr)

    expected = np.array([0.0, 0.0, (0.0019 / 9.0e-4**3) * 2.0e-6])
    np.testing.assert_allclose(result, expected)


def test_rain_mean_volume_diameter_round_trips_single_moment_formula():
    qr = np.array([1.0e-6])
    nr = single_moment_rain_number_per_kg(qr)

    dmean_mm = rain_mean_volume_diameter_mm(qr, nr)

    assert math.isclose(float(dmean_mm[0]), 0.901553, rel_tol=1.0e-5)


def test_datetime64_from_seconds_since():
    result = datetime64_from_seconds_since(
        np.array([0.0, 60.0, 3600.0]),
        "seconds since 2014-09-25",
    )

    assert result.tolist() == [
        np.datetime64("2014-09-25T00:00:00"),
        np.datetime64("2014-09-25T00:01:00"),
        np.datetime64("2014-09-25T01:00:00"),
    ]


def test_average_observation_window_counts_finite_rain_number_samples():
    obs = ObservationSeries(
        times=np.array(
            [
                "2014-09-25T00:00:00",
                "2014-09-25T00:30:00",
                "2014-09-25T01:00:00",
            ],
            dtype="datetime64[s]",
        ),
        rain_number_m3=np.array([100.0, np.nan, 300.0]),
        precip_rate_mm_h=np.array([1.0, np.nan, 3.0]),
        median_volume_diameter_mm=np.array([0.8, np.nan, 1.2]),
    )

    window = average_observation_window(
        obs,
        np.datetime64("2014-09-25T00:30:00"),
        half_window_minutes=30,
    )

    assert window.sample_count == 2
    assert window.rain_number_m3 == 200.0
    assert window.precip_rate_mm_h == 2.0
    assert window.median_volume_diameter_mm == 1.0


def test_summarize_experiment_reports_bias_and_log_correlation():
    obs = np.array([10.0, 100.0, np.nan, 1000.0])
    model = np.array([20.0, 50.0, 30.0, 2000.0])
    qr = np.array([1.0e-8, 0.0, 1.0e-8, 2.0e-8])

    summary = summarize_experiment("2mom", obs, model, qr)

    assert summary["label"] == "G2M"
    assert summary["n_matched"] == 3
    assert summary["n_positive_pair"] == 3
    assert summary["bias_median_m3"] == 10.0
    assert summary["median_model_obs_ratio"] == 2.0
    assert np.isfinite(summary["log10_corr"])
    assert summary["model_rain_qr_positive_count"] == 3


def test_parse_lead_selection_accepts_all_single_and_range():
    assert parse_lead_selection("all") is None
    assert parse_lead_selection("0024") == (24,)
    assert parse_lead_selection("0000-0002,0024") == (0, 1, 2, 24)
    assert lead_label(None) == "all_leads"
    assert lead_label((24,)) == "lead0024"


def test_discover_model_records_uses_init_valid_and_lead_keys(tmp_path):
    base = tmp_path / "control" / "untar-output" / "pf20140101"
    base.mkdir(parents=True)
    for lead in ["0000", "0001", "0024"]:
        (base / f"pfABOFABOF+{lead}").touch()

    records = discover_model_records("control", None, tmp_path)

    assert set(records) == {
        ("2014-01-01T00:00:00", "2014-01-01T00:00:00", 0),
        ("2014-01-01T01:00:00", "2014-01-01T00:00:00", 1),
        ("2014-01-02T00:00:00", "2014-01-01T00:00:00", 24),
    }
    assert discover_model_records("control", (24,), tmp_path) == {
        ("2014-01-02T00:00:00", "2014-01-01T00:00:00", 24): base
        / "pfABOFABOF+0024"
    }
