"""Observation settings. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from pathlib import Path

from alaro_analysis.common.constants import RUNS_ROOT

MODEL_ROOT = RUNS_ROOT / "ALARO"


OBS_ZIP = RUNS_ROOT / "Obs" / "MANAUS_DISDROMETER_DATA.zip"


PROCESSED_DIR = RUNS_ROOT / "processed-data" / "disdrometer_comparison"


ANALYTICS_DIR = PROCESSED_DIR / "analytics"


FIGURE_DIR = RUNS_ROOT / "figures" / "disdrometer_comparison"


CACHE_DIR = Path("cache") / "disdrometer"


BOTTOM_LEVEL = 87


DEFAULT_LAT = -3.21297


DEFAULT_LON = -60.5981


DEFAULT_ALT_M = 50.0
