"""Microphysics budget settings."""

from __future__ import annotations

from alaro_analysis.common.constants import G as G
from alaro_analysis.common.paths import runs_path

DATA_ROOT = runs_path("ALARO")


OUTPUT_DIR = runs_path("figures/microphysics_budget")


DATA_TXT_DIR = OUTPUT_DIR / "data_txt"


CACHE_DIR = runs_path("processed-data/microphysics_budget")


EXPERIMENTS = {
    "control": "C1M",
    "graupel": "G1M",
    "2mom": "G2M",
}


SPECIES = ("RAIN", "SNOW", "GRAUPEL", "LIQUID_WATER", "SOLID_WATER")


SPECIES_LABEL = {
    "RAIN": "rain",
    "SNOW": "snow",
    "GRAUPEL": "graupel",
    "LIQUID_WATER": "cloud liq.",
    "SOLID_WATER": "cloud ice",
}


SPECIES_COLOR = {
    "RAIN": "#1f77b4",
    "SNOW": "#17becf",
    "GRAUPEL": "#9467bd",
    "LIQUID_WATER": "#d62728",
    "SOLID_WATER": "#bcbd22",
}


MIN_LEAD_HOUR = 0


N_WORKERS = 32


T_FREEZE = 273.15  # K


PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")


PANEL_TITLE_FONTSIZE = 16


PANEL_LABEL_FONTSIZE = 14


B1_LEGEND_FONTSIZE = 12


HOUR_BAND_DIURNAL = slice(0, 24)  # full diurnal cycle (default)


HOUR_BAND_DAYTIME = slice(12, 19)  # 12–18 LT convection window (unused)


HOUR_BAND_DEFAULT = HOUR_BAND_DIURNAL


HOUR_BAND_LABEL = "24-hour mean"


FLUX_FLOOR = 1e-8  # kg m-2 s-1; below this, residence time is meaningless
