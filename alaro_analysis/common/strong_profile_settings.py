"""Strong profile settings."""

from __future__ import annotations

import re

import numpy as np

from alaro_analysis.common.paths import data_path, runs_path

DATA_ROOT = runs_path("ALARO")


PAPER_ROOT = data_path("paper1")


H_BINS_KM = np.linspace(0.0, 20.0, 101)


TEMP_VAR = "TEMPERATURE"


STEP_RE = re.compile(r"\+(\d{4})(?:\.[^.]+)*\.nc$")


ISOTHERMS_C = (0.0, -10.0, -20.0)


ISOTHERM_STYLE = {
    0.0: {"color": "#111111", "linestyle": "--", "linewidth": 1.1},
    -10.0: {"color": "#555555", "linestyle": "-.", "linewidth": 1.0},
    -20.0: {"color": "#888888", "linestyle": ":", "linewidth": 1.2},
}
