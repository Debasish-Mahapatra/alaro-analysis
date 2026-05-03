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
