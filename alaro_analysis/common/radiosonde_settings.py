"""Radiosonde settings. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import re

from alaro_analysis.common.paths import data_path, runs_path

DEFAULT_SONDE_ROOT = data_path("ARM/radiosond_data/maosondewnpnM1.b1")


DEFAULT_ALARO_ROOT = runs_path("ALARO")


DEFAULT_OUTPUT_SUBDIR = "radiosonde-matched-profiles"


MODEL_FILE_TEMPLATE = "pfABOFABOF+{hour:04d}"


SONDE_RE = re.compile(r"^maosondewnpnM1\.b1\.(\d{8})\.(\d{6})\.cdf$")


EXPERIMENT_ALIASES = {
    "c1m": "control",
    "control": "control",
    "g1m": "graupel",
    "graupel": "graupel",
    "g2m": "2mom",
    "2mom": "2mom",
}
