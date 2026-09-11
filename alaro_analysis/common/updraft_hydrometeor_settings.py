"""Updraft hydrometeor settings."""

from __future__ import annotations

import numpy as np

from alaro_analysis.common.constants import G as G
from alaro_analysis.common.paths import runs_path

DATA_ROOT = runs_path("ALARO")


EXPERIMENTS = {
    "control": "C1M",
    "graupel": "G1M",
    "2mom": "G2M",
}


HYDROMETEORS = ["RAIN", "SNOW", "GRAUPEL", "LIQUID_WATER", "SOLID_WATER"]


HYDRO_LABELS = {
    "RAIN": "Rain (kg/kg)",
    "SNOW": "Snow (kg/kg)",
    "GRAUPEL": "Graupel (kg/kg)",
    "LIQUID_WATER": "Cloud liquid water (kg/kg)",
    "SOLID_WATER": "Cloud ice (kg/kg)",
}


H_BINS = np.linspace(0.0, 20.0, 101)  # 0.2 km bins (in km)


STRATIFY_CONFIGS = {
    "flux": {
        "bins": np.linspace(-0.1, 3.0, 156),  # ~0.02 kg/m²/s
        "label": "Updraft flux",
        "unit": r"kg m$^{-2}$ s$^{-1}$",
        "coarse_edges": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        "regression_range": (0.1, 2.5),
    },
    "extent": {
        "bins": np.linspace(0.0, 1.0, 101),  # 0.01 fraction
        "label": "Updraft extent",
        "unit": "fraction",
        "coarse_edges": np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]),
        "regression_range": (0.02, 0.8),
    },
    "intensity": {
        "bins": np.linspace(0.0, 150.0, 151),  # 1.0 Pa/s
        "label": "Updraft intensity",
        "unit": "Pa/s",
        "coarse_edges": np.array([0.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0]),
        "regression_range": (5.0, 120.0),
    },
}


FLUX_BINS = STRATIFY_CONFIGS["flux"]["bins"]


MAX_DAYS: int | None = None


MIN_LEAD_HOUR = 3


OUTPUT_DIR = runs_path("figures")


CACHE_DIR = runs_path("processed-data/updraft_hydrometeor")


N_WORKERS = 32


HYDRO_SHORT = {
    "RAIN": "Rain",
    "SNOW": "Snow",
    "GRAUPEL": "Graupel",
    "LIQUID_WATER": "Liquid water",
    "SOLID_WATER": "Ice",
}


EXP_LINE_COLORS = {
    "control": "#d62728",  # red
    "graupel": "#1f77b4",  # blue
    "2mom": "#2ca02c",  # green
}


FS = 22  # base font (axis labels)


FS_TICK = 20  # tick labels


FS_TITLE = 24  # panel titles


FS_CBAR = 20  # colorbar label + ticks


FS_MARG = 16  # marginal axis labels/ticks


FS_LEG = 16  # legend
