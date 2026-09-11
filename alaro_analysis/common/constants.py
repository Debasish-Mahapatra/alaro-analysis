from __future__ import annotations

import re

from .experiments import EXPERIMENT_CATALOG
from .paths import RUNS_ROOT as RUNS_ROOT

# Root of all ALARO model runs, regridded data, and figure outputs.
# (/gpfs/... and /mnt/HDS_CLIMATE/... are two mount paths to the same storage.)

EXPERIMENTS = ("control", "graupel", "2mom")
EXPERIMENT_LABELS = {key: exp.label for key, exp in EXPERIMENT_CATALOG.items()}

# Line/marker colour per base experiment.  Kept here so every workflow
# (main plots, convective_feedback, microphysics_budget, kt273_diurnal,
# updraft_hydrometeor, ddh) renders the three models in the same palette.
EXPERIMENT_COLORS = {key: exp.color for key, exp in EXPERIMENT_CATALOG.items()}

SEASONS = {
    "wet": {"label": "Wet (Dec-Apr)", "months": (12, 1, 2, 3, 4)},
    "transition_wet_to_dry": {"label": "Transition Wet->Dry (May-Jun)", "months": (5, 6)},
    "dry": {"label": "Dry (Jul-Sep)", "months": (7, 8, 9)},
    "transition_dry_to_wet": {"label": "Transition Dry->Wet (Oct-Nov)", "months": (10, 11)},
}

DAY_RE = re.compile(r"^(?:pf|sfx)(\d{8})$")
FILE_HOUR_RE = re.compile(r"\+(\d{4})(?:\.[^.]+)*\.nc$")
SANITIZE_RE = re.compile(r"[^A-Za-z0-9]+")

FREEZING_K = 273.15
CP_D = 1004.0
LV = 2.5e6
G = 9.80665
RD = 287.05
EPS = 0.622
P0 = 100000.0
EARTH_RADIUS_M = 6371000.0
