from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_expected_console_scripts_are_declared():
    with (ROOT / "pyproject.toml").open("rb") as fh:
        project = tomllib.load(fh)["project"]
    scripts = project["scripts"]

    assert scripts["alaro-convert"] == "alaro_analysis.converter:main"
    assert scripts["alaro-surface"] == "alaro_analysis.workflows.surface:main"
    assert scripts["alaro-temperature"] == "alaro_analysis.workflows.temperature:main"
    assert scripts["alaro-hydrometeor"] == "alaro_analysis.workflows.hydrometeor:main"
    assert scripts["alaro-diagnostics"] == "alaro_analysis.workflows.diagnostics:main"
    assert scripts["alaro-radiation-compare"] == (
        "alaro_analysis.workflows.radiation_compare:main"
    )
    assert scripts["alaro-pair-analysis"] == "alaro_analysis.workflows.pair_analysis:main"
    assert scripts["alaro-panel-anomaly"] == "alaro_analysis.workflows.panel_anomaly:main"
    assert scripts["alaro-disdrometer-masked-domain"] == (
        "alaro_analysis.workflows.disdrometer_masked_domain:main"
    )
    assert scripts["alaro-disdrometer-dsd"] == (
        "alaro_analysis.workflows.disdrometer_dsd:main"
    )
    assert scripts["alaro-radiosonde-profiles"] == (
        "alaro_analysis.workflows.radiosonde_profiles:main"
    )
    assert scripts["alaro-plot-radiosonde-profiles"] == (
        "alaro_analysis.workflows.plot_radiosonde_profiles:main"
    )
    assert scripts["alaro-precip-distribution"] == (
        "alaro_analysis.workflows.precip_distribution:main"
    )
    assert scripts["alaro-precip-diurnal-cycle"] == (
        "alaro_analysis.workflows.precip_diurnal_cycle:main"
    )
    assert scripts["alaro-precip-rebuild-hourly-imerg-common-valid"] == (
        "alaro_analysis.workflows.precip_rebuild_hourly_imerg_common_valid:main"
    )
    assert scripts["alaro-precip-spatial-bias-maps"] == (
        "alaro_analysis.workflows.precip_spatial_bias_maps:main"
    )
    assert scripts["alaro-precip-intensity-heatmaps"] == (
        "alaro_analysis.workflows.precip_intensity_heatmaps:main"
    )
    assert scripts["alaro-hydrometeor-vertical-profiles"] == (
        "alaro_analysis.workflows.hydrometeor_vertical_profiles:main"
    )
    assert scripts["alaro-dsd-vertical-profiles"] == (
        "alaro_analysis.workflows.dsd_vertical_profiles:main"
    )
    assert scripts["alaro-dsd-cfad-profiles"] == (
        "alaro_analysis.workflows.dsd_cfad_profiles:main"
    )
    assert scripts["alaro-dsd-cfad-strong-convection"] == (
        "alaro_analysis.workflows.dsd_cfad_strong_convection:main"
    )
    assert scripts["alaro-dsd-cfad-matched-colorbars"] == (
        "alaro_analysis.workflows.dsd_cfad_matched_colorbars:main"
    )
    assert scripts["alaro-dsd-cfad-gridcell-matched"] == (
        "alaro_analysis.workflows.dsd_cfad_gridcell_matched:main"
    )
    assert scripts["alaro-updraft-spatial-snapshot"] == (
        "alaro_analysis.workflows.updraft_spatial_snapshot:main"
    )
    assert scripts["alaro-updraft-mass-flux-timeseries"] == (
        "alaro_analysis.workflows.updraft_mass_flux_timeseries:main"
    )
    assert scripts["alaro-ddh-closure-budget-compression"] == (
        "alaro_analysis.ddh.plot_closure_budget_compression:main"
    )
    assert scripts["alaro-tobac-rainfall-tracking"] == (
        "alaro_analysis.tobac.rainfall_tracking:main"
    )
    assert scripts["alaro-convective-feedback"] == (
        "alaro_analysis.workflows.convective_feedback:main"
    )
    assert scripts["alaro-deaccumulate-precipitation"] == (
        "alaro_analysis.workflows.deaccumulate_precipitation:main"
    )
    assert scripts["alaro-kt273-diurnal"] == (
        "alaro_analysis.workflows.kt273_diurnal:main"
    )
    assert scripts["alaro-microphysics-budget"] == (
        "alaro_analysis.workflows.microphysics_budget:main"
    )
    assert scripts["alaro-updraft-hydrometeor"] == (
        "alaro_analysis.workflows.updraft_hydrometeor:main"
    )


def test_fa_stack_is_optional():
    with (ROOT / "pyproject.toml").open("rb") as fh:
        project = tomllib.load(fh)["project"]

    assert "faxarray" not in project["dependencies"]
    assert "faxarray" in project["optional-dependencies"]["fa"]
    assert "faxarray" in project["optional-dependencies"]["full"]


@pytest.mark.parametrize(
    "module",
    [
        "alaro_analysis.converter.cli",
        "alaro_analysis.workflows.surface",
        "alaro_analysis.workflows.temperature",
        "alaro_analysis.workflows.hydrometeor",
        "alaro_analysis.workflows.diagnostics",
        "alaro_analysis.workflows.radiation_compare",
        "alaro_analysis.workflows.pair_analysis",
        "alaro_analysis.workflows.panel_anomaly",
        "alaro_analysis.ddh.plot_condensation_partition",
        "alaro_analysis.ddh.plot_phase_changes",
        "alaro_analysis.workflows.disdrometer_comparison",
        "alaro_analysis.workflows.disdrometer_masked_domain",
        "alaro_analysis.workflows.disdrometer_dsd",
        "alaro_analysis.workflows.radiosonde_profiles",
        "alaro_analysis.workflows.plot_radiosonde_profiles",
        "alaro_analysis.workflows.precip_distribution",
        "alaro_analysis.workflows.precip_diurnal_cycle",
        "alaro_analysis.workflows.precip_rebuild_hourly_imerg_common_valid",
        "alaro_analysis.workflows.precip_spatial_bias_maps",
        "alaro_analysis.workflows.precip_intensity_heatmaps",
        "alaro_analysis.workflows.hydrometeor_vertical_profiles",
        "alaro_analysis.workflows.dsd_vertical_profiles",
        "alaro_analysis.workflows.dsd_cfad_profiles",
        "alaro_analysis.workflows.dsd_cfad_strong_convection",
        "alaro_analysis.workflows.dsd_cfad_matched_colorbars",
        "alaro_analysis.workflows.dsd_cfad_gridcell_matched",
        "alaro_analysis.workflows.updraft_spatial_snapshot",
        "alaro_analysis.workflows.updraft_mass_flux_timeseries",
        "alaro_analysis.ddh.plot_closure_budget_compression",
        "alaro_analysis.tobac.rainfall_tracking",
        "alaro_analysis.workflows.convective_feedback",
        "alaro_analysis.workflows.deaccumulate_precipitation",
        "alaro_analysis.workflows.kt273_diurnal",
        "alaro_analysis.workflows.microphysics_budget",
        "alaro_analysis.workflows.updraft_hydrometeor",
    ],
)
def test_command_module_help_smoke(module: str):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 and "No module named" in result.stderr:
        pytest.skip(f"Optional command dependency missing: {result.stderr.strip()}")

    assert result.returncode == 0, result.stderr
    assert "--help" in result.stdout
