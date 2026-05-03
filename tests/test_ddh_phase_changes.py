from pathlib import Path

import numpy as np

from alaro_analysis.ddh.plot_phase_changes import (
    combine_total_evaporation_profiles,
    combine_total_sublimation_profiles,
    common_positive_xlim,
    compute_phase_profile,
    load_all_phase_profiles,
    plot_species_panels,
    plot_total_evaporation_experiment_comparison,
    plot_total_sublimation_experiment_comparison,
    save_npz,
    summarize_profile,
    write_by_level_csv,
    write_summary_csv,
)


def test_compute_condensation_profile_uses_positive_hydrometeor_source():
    profile = compute_phase_profile(
        experiment="control",
        lead="24",
        variable="QL",
        species="cloud_liquid",
        process="condensation_deposition",
        title="Cloud liquid condensation",
        altitude_km=np.array([0.0, 1.0, 2.0]),
        convection_block=np.array([1.0, -2.0, 3.0]),
        resolved_block=np.array([4.0, 5.0, -6.0]),
        sign="source",
        n_days=730,
    )

    np.testing.assert_allclose(profile.convection_gkgday, [1.0, 0.0, 3.0])
    np.testing.assert_allclose(profile.resolved_gkgday, [4.0, 5.0, 0.0])
    np.testing.assert_allclose(profile.total_gkgday, [5.0, 5.0, 3.0])
    assert profile.lead == "0024"
    assert profile.n_days == 730


def test_compute_evaporation_profile_uses_positive_hydrometeor_loss():
    profile = compute_phase_profile(
        experiment="2mom",
        lead="0024",
        variable="QS",
        species="snow",
        process="evaporation_sublimation",
        title="Snow sublimation",
        altitude_km=np.array([0.0, 1.0]),
        convection_block=np.array([-1.0, 2.0]),
        resolved_block=np.array([-3.0, -4.0]),
        sign="loss",
    )

    np.testing.assert_allclose(profile.convection_gkgday, [1.0, 0.0])
    np.testing.assert_allclose(profile.resolved_gkgday, [3.0, 4.0])
    np.testing.assert_allclose(profile.total_gkgday, [4.0, 4.0])


def test_load_all_phase_profiles_from_synthetic_aggregates(tmp_path: Path):
    lead_dir = tmp_path / "lead0024_VZ"
    lead_dir.mkdir()
    np.savez(
        lead_dir / "control_QL.npz",
        **{
            "altitude_km": np.array([0.0, 1.0]),
            "days": np.array(["DDH20140101"]),
            "block__cond-cv": np.array([1.0, 2.0]),
            "block__cond-rs": np.array([3.0, 4.0]),
        },
    )
    np.savez(
        lead_dir / "control_QR.npz",
        **{
            "altitude_km": np.array([0.0, 1.0]),
            "days": np.array(["DDH20140101"]),
            "block__evap-cv": np.array([-1.0, -2.0]),
            "block__evap-rs": np.array([-3.0, -4.0]),
        },
    )

    profiles = load_all_phase_profiles(agg_dir=tmp_path)

    assert [(p.variable, p.process) for p in profiles] == [
        ("QL", "condensation_deposition"),
        ("QR", "evaporation_sublimation"),
    ]
    np.testing.assert_allclose(profiles[1].total_gkgday, [4.0, 6.0])


def test_summary_xlim_and_output_writers(tmp_path: Path):
    profile = compute_phase_profile(
        experiment="graupel",
        lead="0024",
        variable="QG",
        species="graupel",
        process="evaporation_sublimation",
        title="Graupel sublimation/evaporation",
        altitude_km=np.array([0.0, 1.0, 2.0]),
        convection_block=np.array([-1.0, -2.0, -3.0]),
        resolved_block=np.array([-2.0, -4.0, -6.0]),
        sign="loss",
    )

    rows = summarize_profile(profile)
    total = next(row for row in rows if row["pathway"] == "total")
    assert total["column_gkgday_km"] == 12.0
    assert total["peak_gkgday"] == 9.0
    assert total["peak_altitude_km"] == 2.0
    assert common_positive_xlim([profile], pad_fraction=0.1) == (0.0, 9.9)

    npz_path = tmp_path / "phase.npz"
    by_level_path = tmp_path / "analytics" / "by_level.csv"
    summary_path = tmp_path / "analytics" / "summary.csv"
    save_npz([profile], npz_path)
    write_by_level_csv([profile], by_level_path)
    write_summary_csv([profile], summary_path)

    with np.load(npz_path) as data:
        np.testing.assert_allclose(
            data["graupel_QG_evaporation_sublimation_total_gkgday"],
            [3.0, 6.0, 9.0],
        )
    assert "rate_gkgday" in by_level_path.read_text()
    assert "column_gkgday_km" in summary_path.read_text()


def test_plot_species_panels_writes_one_species_figure(tmp_path: Path):
    profile = compute_phase_profile(
        experiment="graupel",
        lead="0024",
        variable="QG",
        species="graupel",
        process="evaporation_sublimation",
        title="Graupel sublimation/evaporation",
        altitude_km=np.array([0.0, 1.0, 2.0]),
        convection_block=np.array([-1.0, -2.0, -3.0]),
        resolved_block=np.array([-2.0, -4.0, -6.0]),
        sign="loss",
    )

    out = tmp_path / "qg_evaporation_sublimation.png"
    returned = plot_species_panels(
        [profile],
        "evaporation_sublimation",
        "graupel",
        out,
    )

    assert returned == out
    assert out.stat().st_size > 0


def test_combine_total_sublimation_profiles_sums_snow_and_graupel():
    snow = compute_phase_profile(
        experiment="2mom",
        lead="0024",
        variable="QS",
        species="snow",
        process="evaporation_sublimation",
        title="Snow sublimation",
        altitude_km=np.array([0.0, 1.0]),
        convection_block=np.array([-1.0, -2.0]),
        resolved_block=np.array([-3.0, -4.0]),
        sign="loss",
    )
    graupel = compute_phase_profile(
        experiment="2mom",
        lead="0024",
        variable="QG",
        species="graupel",
        process="evaporation_sublimation",
        title="Graupel sublimation/evaporation",
        altitude_km=np.array([0.0, 1.0]),
        convection_block=np.array([-5.0, -6.0]),
        resolved_block=np.array([-7.0, -8.0]),
        sign="loss",
    )

    total = combine_total_sublimation_profiles([snow, graupel])

    assert len(total) == 1
    assert total[0].species == "total_sublimation"
    assert total[0].process == "sublimation"
    np.testing.assert_allclose(total[0].convection_gkgday, [6.0, 8.0])
    np.testing.assert_allclose(total[0].resolved_gkgday, [10.0, 12.0])
    np.testing.assert_allclose(total[0].total_gkgday, [16.0, 20.0])


def test_combine_total_evaporation_profiles_sums_all_evaporation_terms():
    rain = compute_phase_profile(
        experiment="control",
        lead="0024",
        variable="QR",
        species="rain",
        process="evaporation_sublimation",
        title="Rain evaporation",
        altitude_km=np.array([0.0, 1.0]),
        convection_block=np.array([-1.0, -2.0]),
        resolved_block=np.array([-3.0, -4.0]),
        sign="loss",
    )
    snow = compute_phase_profile(
        experiment="control",
        lead="0024",
        variable="QS",
        species="snow",
        process="evaporation_sublimation",
        title="Snow sublimation",
        altitude_km=np.array([0.0, 1.0]),
        convection_block=np.array([-5.0, -6.0]),
        resolved_block=np.array([-7.0, -8.0]),
        sign="loss",
    )

    total = combine_total_evaporation_profiles([rain, snow])

    assert len(total) == 1
    assert total[0].species == "total_evaporation"
    assert total[0].process == "evaporation"
    np.testing.assert_allclose(total[0].convection_gkgday, [6.0, 8.0])
    np.testing.assert_allclose(total[0].resolved_gkgday, [10.0, 12.0])
    np.testing.assert_allclose(total[0].total_gkgday, [16.0, 20.0])


def test_plot_total_sublimation_experiment_comparison_writes_figure(tmp_path: Path):
    snow = compute_phase_profile(
        experiment="control",
        lead="0024",
        variable="QS",
        species="snow",
        process="evaporation_sublimation",
        title="Snow sublimation",
        altitude_km=np.array([0.0, 1.0]),
        convection_block=np.array([-1.0, -2.0]),
        resolved_block=np.array([-3.0, -4.0]),
        sign="loss",
    )
    total = combine_total_sublimation_profiles([snow])

    out = tmp_path / "total_sublimation_by_experiment.png"
    returned = plot_total_sublimation_experiment_comparison(total, out)

    assert returned == out
    assert out.stat().st_size > 0


def test_plot_total_evaporation_experiment_comparison_writes_figure(tmp_path: Path):
    rain = compute_phase_profile(
        experiment="control",
        lead="0024",
        variable="QR",
        species="rain",
        process="evaporation_sublimation",
        title="Rain evaporation",
        altitude_km=np.array([0.0, 1.0]),
        convection_block=np.array([-1.0, -2.0]),
        resolved_block=np.array([-3.0, -4.0]),
        sign="loss",
    )
    total = combine_total_evaporation_profiles([rain])

    out = tmp_path / "total_evaporation_by_experiment.png"
    returned = plot_total_evaporation_experiment_comparison(total, out)

    assert returned == out
    assert out.stat().st_size > 0
