from pathlib import Path

import numpy as np

from alaro_analysis.ddh.plot_condensation_partition import (
    common_rate_xlim,
    compute_partition,
    load_partition,
    save_npz,
    summarize_partition,
    write_by_level_csv,
    write_summary_csv,
)


def test_compute_partition_uses_positive_qv_sink_terms():
    part = compute_partition(
        experiment="control",
        lead="24",
        altitude_km=np.array([0.0, 1.0, 2.0]),
        condcv=np.array([-2.0, 1.0, -3.0]),
        condrs=np.array([-1.0, -4.0, 0.0]),
        n_days=730,
    )

    np.testing.assert_allclose(part.convective_gkgday, [2.0, 0.0, 3.0])
    np.testing.assert_allclose(part.resolved_gkgday, [1.0, 4.0, 0.0])
    np.testing.assert_allclose(part.total_gkgday, [3.0, 4.0, 3.0])
    np.testing.assert_allclose(part.convective_fraction, [2 / 3, 0.0, 1.0])
    np.testing.assert_allclose(part.resolved_fraction, [1 / 3, 1.0, 0.0])
    assert part.lead == "0024"
    assert part.n_days == 730


def test_compute_partition_zero_total_fractions_are_nan():
    part = compute_partition(
        experiment="control",
        lead="0024",
        altitude_km=np.array([0.0, 1.0]),
        condcv=np.array([0.0, 2.0]),
        condrs=np.array([0.0, 1.0]),
    )

    np.testing.assert_allclose(part.total_gkgday, [0.0, 0.0])
    assert np.isnan(part.convective_fraction).all()
    assert np.isnan(part.resolved_fraction).all()


def test_summarize_partition_integrates_by_altitude_and_finds_peak():
    part = compute_partition(
        experiment="graupel",
        lead="0024",
        altitude_km=np.array([2.0, 1.0, 0.0]),
        condcv=np.array([-2.0, -2.0, -2.0]),
        condrs=np.array([-1.0, -3.0, -5.0]),
        n_days=730,
    )

    summary = summarize_partition(part)

    assert summary["experiment"] == "graupel"
    assert summary["lead"] == "0024"
    assert summary["n_days"] == 730
    assert summary["column_convective_gkgday_km"] == 4.0
    assert summary["column_resolved_gkgday_km"] == 6.0
    assert summary["column_total_gkgday_km"] == 10.0
    assert summary["column_convective_fraction"] == 0.4
    assert summary["column_resolved_fraction"] == 0.6
    assert summary["peak_total_gkgday"] == 7.0
    assert summary["peak_total_altitude_km"] == 0.0


def test_common_rate_xlim_uses_global_maximum_for_comparable_panels():
    first = compute_partition(
        experiment="control",
        lead="0024",
        altitude_km=np.array([0.0, 1.0]),
        condcv=np.array([-1.0, -2.0]),
        condrs=np.array([-1.0, -1.0]),
    )
    second = compute_partition(
        experiment="graupel",
        lead="0024",
        altitude_km=np.array([0.0, 1.0]),
        condcv=np.array([-4.0, -5.0]),
        condrs=np.array([-2.0, -3.0]),
    )

    assert common_rate_xlim([first, second], pad_fraction=0.1) == (0.0, 8.8)


def test_load_partition_reads_aggregated_qv_npz(tmp_path: Path):
    lead_dir = tmp_path / "lead0024_VZ"
    lead_dir.mkdir()
    np.savez(
        lead_dir / "control_QV.npz",
        altitude_km=np.array([0.0, 1.0]),
        days=np.array(["DDH20140101", "DDH20140102"]),
        block__condcv=np.array([-1.0, -2.0]),
        block__condrs=np.array([-3.0, -4.0]),
    )

    part = load_partition("control", lead="0024", agg_dir=tmp_path)

    np.testing.assert_allclose(part.convective_gkgday, [1.0, 2.0])
    np.testing.assert_allclose(part.resolved_gkgday, [3.0, 4.0])
    assert part.n_days == 2


def test_save_npz_and_csv_outputs(tmp_path: Path):
    part = compute_partition(
        experiment="2mom",
        lead="0024",
        altitude_km=np.array([0.0, 1.0]),
        condcv=np.array([-1.0, -2.0]),
        condrs=np.array([-3.0, -4.0]),
    )

    npz_path = tmp_path / "condensation.npz"
    by_level_path = tmp_path / "analytics" / "by_level.csv"
    summary_path = tmp_path / "analytics" / "summary.csv"

    save_npz([part], npz_path)
    write_by_level_csv([part], by_level_path)
    write_summary_csv([part], summary_path)

    with np.load(npz_path) as data:
        np.testing.assert_allclose(data["2mom_total_gkgday"], [4.0, 6.0])
        assert data["experiments"].tolist() == ["2mom"]

    assert "convective_fraction" in by_level_path.read_text()
    assert "column_total_gkgday_km" in summary_path.read_text()
