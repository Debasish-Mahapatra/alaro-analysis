"""Convective feedback settings."""

from __future__ import annotations

import numpy as np

from alaro_analysis.common.constants import G as G
from alaro_analysis.common.paths import runs_path

DATA_ROOT = runs_path("ALARO")


OUTPUT_DIR = runs_path("figures")


CACHE_DIR = runs_path("processed-data/convective_feedback")


EXPERIMENTS = {
    "control": "C1M",
    "graupel": "G1M",
    "2mom": "G2M",
}


HYDROMETEORS = ["RAIN", "SNOW", "GRAUPEL", "LIQUID_WATER", "SOLID_WATER"]


MIN_LEAD_HOUR = 3


N_WORKERS = 32


LOADING_BINS = np.linspace(0.0, 0.01, 101)  # 0–10 g/kg in 0.1 g/kg steps


H_BINS = np.linspace(0.0, 20.0, 101)


DD_H_BINS = np.linspace(0.0, 20.0, 101)


THERMO_VARS = {
    "HUMI.SPECIFI": {"label": "Specific humidity", "unit": "g/kg", "scale": 1000.0},
    "HUMI.RELATIVE": {"label": "Relative humidity", "unit": "%", "scale": 100.0},
    "TEMPERATURE": {"label": "Temperature", "unit": "K", "scale": 1.0},
}


PRECONV_LEADS = (6, 7, 8, 9)  # lead hours corresponding to ~06–09 LT


H4_H_BINS = np.linspace(0.0, 20.0, 101)
