"""Microphysics exports."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from alaro_analysis.analysis.microphysics_budget import (
    _mean_over_hours,
    _mean_scalar_over_hours,
)
from alaro_analysis.common.microphysics_budget_settings import (
    EXPERIMENTS,
    FLUX_FLOOR,
    HOUR_BAND_DEFAULT,
    HOUR_BAND_LABEL,
    SPECIES,
)
from alaro_analysis.plotting.microphysics_budget import _b1_global_mass_xlim


def _txt_path_for_plot(output_path: Path) -> Path:
    return output_path.parent / "data_txt" / f"{output_path.stem}.txt"


def _fmt(value: object) -> str:
    if isinstance(value, str):
        return value
    numeric = float(value)
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.10g}"


def _write_headline(fh, title: str, output_path: Path, notes: Sequence[str] = ()):
    fh.write(f"{title}\n")
    fh.write(f"{'=' * len(title)}\n")
    fh.write(f"Source plot: {output_path}\n")
    fh.write(f"Hour averaging: {HOUR_BAND_LABEL} ({HOUR_BAND_DEFAULT.start or 0}-23 UTC)\n")
    for note in notes:
        fh.write(f"{note}\n")
    fh.write("\n")


def _write_csv_section(fh, title: str, columns: Sequence[str], rows: Sequence[Sequence[object]]):
    fh.write(f"{title}\n")
    fh.write(f"{'-' * len(title)}\n")
    fh.write(",".join(columns) + "\n")
    for row in rows:
        fh.write(",".join(_fmt(value) for value in row) + "\n")
    fh.write("\n")


def _write_plot_txt(output_path: Path, writer):
    txt_path = _txt_path_for_plot(output_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as fh:
        writer(fh)
    print(f"  txt: {txt_path}")


def _write_freezing_level_section(fh, finals: dict[str, dict[str, np.ndarray]]):
    rows = []
    for exp, label in EXPERIMENTS.items():
        if exp not in finals:
            continue
        z0 = _mean_scalar_over_hours(finals[exp]["z_freeze"], HOUR_BAND_DEFAULT)
        rows.append((exp, label, z0 / 1000.0 if np.isfinite(z0) else np.nan))
    _write_csv_section(fh, "Freezing-level data", ("experiment", "label", "z_freeze_km"), rows)


def write_b1_species_fraction_txt(finals: dict[str, dict[str, np.ndarray]], output_path: Path):
    def _writer(fh):
        _write_headline(
            fh,
            "B1 Species Fraction Plot Data",
            output_path,
            (
                "Values are fractional hydrometeor mass per model layer.",
                "Each species is divided by total hydrometeor mass at the same height.",
            ),
        )
        _write_freezing_level_section(fh, finals)
        for exp, label in EXPERIMENTS.items():
            if exp not in finals:
                continue
            f = finals[exp]
            z_km = _mean_over_hours(f["height_m"], HOUR_BAND_DEFAULT) / 1000.0
            total = _mean_over_hours(f["mass_per_level_total"], HOUR_BAND_DEFAULT)
            total_safe = np.where(total > 0, total, np.nan)
            species_profiles = {
                sp: _mean_over_hours(f[f"mass_per_level_{sp}"], HOUR_BAND_DEFAULT) / total_safe
                for sp in SPECIES
            }
            rows = [
                (z_km[i], *(species_profiles[sp][i] for sp in SPECIES)) for i in range(z_km.size)
            ]
            _write_csv_section(
                fh,
                f"{label} fractional profiles",
                ("height_km", *(f"{sp}_fraction" for sp in SPECIES)),
                rows,
            )

    _write_plot_txt(output_path, _writer)


def write_b1_species_amount_txt(finals: dict[str, dict[str, np.ndarray]], output_path: Path):
    xlim = _b1_global_mass_xlim(finals)

    def _writer(fh):
        _write_headline(
            fh,
            "B1 Species Amount Plot Data",
            output_path,
            (
                "Values are actual hydrometeor mass per model layer.",
                "Units: kg m-2.",
                f"Shared plot x-axis limits: {_fmt(xlim[0])}, {_fmt(xlim[1])} kg m-2.",
            ),
        )
        _write_freezing_level_section(fh, finals)
        for exp, label in EXPERIMENTS.items():
            if exp not in finals:
                continue
            f = finals[exp]
            z_km = _mean_over_hours(f["height_m"], HOUR_BAND_DEFAULT) / 1000.0
            species_profiles = {
                sp: _mean_over_hours(f[f"mass_per_level_{sp}"], HOUR_BAND_DEFAULT) for sp in SPECIES
            }
            total = sum(np.nan_to_num(species_profiles[sp], nan=0.0) for sp in SPECIES)
            rows = [
                (z_km[i], *(species_profiles[sp][i] for sp in SPECIES), total[i])
                for i in range(z_km.size)
            ]
            _write_csv_section(
                fh,
                f"{label} mass profiles",
                ("height_km", *(f"{sp}_kg_m-2" for sp in SPECIES), "total_kg_m-2"),
                rows,
            )

    _write_plot_txt(output_path, _writer)


def write_b1b_mass_above_species_txt(finals: dict[str, dict[str, np.ndarray]], output_path: Path):
    def _writer(fh):
        _write_headline(
            fh,
            "B1b Mass Above Species Plot Data",
            output_path,
            (
                "Values are integrated condensate mass above height.",
                "Units in this file match the plot axis: g m-2.",
            ),
        )
        _write_freezing_level_section(fh, finals)
        for exp, label in EXPERIMENTS.items():
            if exp not in finals:
                continue
            f = finals[exp]
            z_km = _mean_over_hours(f["height_m"], HOUR_BAND_DEFAULT) / 1000.0
            species_profiles = {
                sp: _mean_over_hours(f[f"mass_above_{sp}"], HOUR_BAND_DEFAULT) * 1000.0
                for sp in SPECIES
            }
            rows = [
                (z_km[i], *(species_profiles[sp][i] for sp in SPECIES)) for i in range(z_km.size)
            ]
            _write_csv_section(
                fh,
                f"{label} mass-above profiles",
                ("height_km", *(f"{sp}_g_m-2" for sp in SPECIES)),
                rows,
            )

    _write_plot_txt(output_path, _writer)


def write_b2_surface_reach_txt(finals: dict[str, dict[str, np.ndarray]], output_path: Path):
    def _writer(fh):
        _write_headline(
            fh,
            "B2 Surface-Reach Fraction Plot Data",
            output_path,
            (
                "Surface-reach fraction = surface_flux / max_column_flux.",
                "Flux units: kg m-2 s-1.",
            ),
        )
        for exp, label in EXPERIMENTS.items():
            if exp not in finals:
                continue
            f = finals[exp]
            surf = f["surface_flux"]
            col = f["max_column_flux"]
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(col > 0, surf / col, np.nan)
            rows = [(hour, surf[hour], col[hour], frac[hour]) for hour in range(24)]
            _write_csv_section(
                fh,
                f"{label} hourly surface-reach data",
                (
                    "hour_utc",
                    "surface_flux_kg_m-2_s-1",
                    "max_column_flux_kg_m-2_s-1",
                    "surface_reach_fraction",
                ),
                rows,
            )

    _write_plot_txt(output_path, _writer)


def write_b3_residence_time_txt(finals: dict[str, dict[str, np.ndarray]], output_path: Path):
    def _writer(fh):
        _write_headline(
            fh,
            "B3 Residence Time Plot Data",
            output_path,
            (
                r"Residence time tau(z) = mass_above(z) / flux(z).",
                f"Flux floor used before tau is plotted: {FLUX_FLOOR:g} kg m-2 s-1.",
                "Residence-time units: seconds.",
            ),
        )
        _write_freezing_level_section(fh, finals)
        for exp, label in EXPERIMENTS.items():
            if exp not in finals:
                continue
            f = finals[exp]
            z_km = _mean_over_hours(f["height_m"], HOUR_BAND_DEFAULT) / 1000.0
            mass_tot = _mean_over_hours(f["mass_above_total"], HOUR_BAND_DEFAULT)
            flux = _mean_over_hours(f["flux_total"], HOUR_BAND_DEFAULT)
            mass_snow = _mean_over_hours(
                f.get("mass_above_SNOW", np.zeros_like(flux)), HOUR_BAND_DEFAULT
            )
            mass_graupel = _mean_over_hours(
                f.get("mass_above_GRAUPEL", np.zeros_like(flux)), HOUR_BAND_DEFAULT
            )
            flux_ok = flux > FLUX_FLOOR
            with np.errstate(divide="ignore", invalid="ignore"):
                tau_total = np.where(flux_ok, mass_tot / flux, np.nan)
                tau_snow = np.where(flux_ok, mass_snow / flux, np.nan)
                tau_graupel = np.where(flux_ok, mass_graupel / flux, np.nan)
            rows = [
                (
                    z_km[i],
                    flux[i],
                    mass_tot[i],
                    mass_snow[i],
                    mass_graupel[i],
                    tau_total[i],
                    tau_snow[i],
                    tau_graupel[i],
                )
                for i in range(z_km.size)
            ]
            _write_csv_section(
                fh,
                f"{label} residence-time profiles",
                (
                    "height_km",
                    "flux_total_kg_m-2_s-1",
                    "mass_above_total_kg_m-2",
                    "mass_above_SNOW_kg_m-2",
                    "mass_above_GRAUPEL_kg_m-2",
                    "tau_total_s",
                    "tau_snow_s",
                    "tau_graupel_s",
                ),
                rows,
            )

    _write_plot_txt(output_path, _writer)


def write_b4_flux_divergence_txt(finals: dict[str, dict[str, np.ndarray]], output_path: Path):
    def _writer(fh):
        _write_headline(
            fh,
            "B4 Flux Divergence Plot Data",
            output_path,
            (
                "Positive values indicate a net source; negative values indicate a net sink.",
                "Raw plot values are -dF/dz in 10^-6 kg m-3 s-1.",
                "Normalized plot values are (-dF/dz / surface_flux) in 10^-3 m-1.",
            ),
        )
        _write_freezing_level_section(fh, finals)
        for exp, label in EXPERIMENTS.items():
            if exp not in finals:
                continue
            f = finals[exp]
            z_km = _mean_over_hours(f["height_m"], HOUR_BAND_DEFAULT) / 1000.0
            fdiv = _mean_over_hours(f["flux_div"], HOUR_BAND_DEFAULT)
            surf = _mean_scalar_over_hours(f["surface_flux"], HOUR_BAND_DEFAULT)
            if np.isfinite(surf) and surf > 0:
                normalized = fdiv / surf * 1000.0
            else:
                normalized = np.full_like(fdiv, np.nan)
            rows = [
                (z_km[i], fdiv[i], fdiv[i] * 1e6, surf, normalized[i]) for i in range(z_km.size)
            ]
            _write_csv_section(
                fh,
                f"{label} flux-divergence profiles",
                (
                    "height_km",
                    "flux_divergence_kg_m-3_s-1",
                    "flux_divergence_1e-6_kg_m-3_s-1",
                    "surface_flux_kg_m-2_s-1",
                    "normalized_flux_divergence_1e-3_m-1",
                ),
                rows,
            )

    _write_plot_txt(output_path, _writer)
