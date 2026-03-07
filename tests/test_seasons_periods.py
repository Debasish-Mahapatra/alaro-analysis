from pathlib import Path

from alaro_analysis.common.models import PeriodSpec
from alaro_analysis.common.seasons import build_period_specs, resolve_seasons


def test_resolve_seasons_all():
    seasons = resolve_seasons(["all"])
    assert "wet" in seasons
    assert "dry" in seasons


def test_resolve_seasons_subset_dedup():
    seasons = resolve_seasons(["wet", "WET", "dry"])
    assert seasons == ["wet", "dry"]


def test_build_period_specs_full_and_seasonal():
    specs = build_period_specs({"full", "seasonal"}, ["wet"])
    assert len(specs) == 2

    full = specs[0]
    wet = specs[1]
    assert full.key == "full_2yr"
    assert full.output_subdir == Path("2years")
    assert full.allowed_months is None

    assert wet.key == "wet"
    assert wet.output_subdir == Path("seasonal") / "wet"
    assert wet.allowed_months == (12, 1, 2, 3, 4)


def test_periodspec_supports_legacy_positional_order():
    spec = PeriodSpec("key", "label", (12, 1, 2), Path("seasonal") / "wet")
    assert spec.key == "key"
    assert spec.label == "label"
    assert spec.allowed_months == (12, 1, 2)
    assert spec.output_subdir == Path("seasonal") / "wet"
