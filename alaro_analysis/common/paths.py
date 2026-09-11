"""Machine locations, independent of scientific settings.

Set ALARO_DATA_ROOT, ALARO_RUNS_ROOT or ALARO_WORKSPACE_ROOT before invoking
a command. Existing HPC locations remain the defaults. CLI path options can
also be supplied in alaro.toml; no analysis module needs a personal path.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("ALARO_DATA_ROOT", "/mnt/HDS_CLIMATE/CLIMATE/deba")).expanduser()
RUNS_ROOT = Path(os.environ.get("ALARO_RUNS_ROOT", DATA_ROOT / "ALARO-RUNS")).expanduser()
WORKSPACE_ROOT = Path(
    os.environ.get("ALARO_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])
).expanduser()


def data_path(relative: str = "") -> Path:
    """Resolve an observation, external-tool or study path under DATA_ROOT."""
    return DATA_ROOT / relative


def runs_path(relative: str = "") -> Path:
    """Resolve a model input or analysis product under RUNS_ROOT."""
    return RUNS_ROOT / relative


def workspace_path(relative: str = "") -> Path:
    """Resolve a repository resource without depending on the current directory."""
    return WORKSPACE_ROOT / relative
