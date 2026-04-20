from __future__ import annotations

import re

EXPERIMENTS = ("control", "graupel", "2mom")
EXPERIMENT_LABELS = {"control": "C1M", "graupel": "G1M", "2mom": "G2M", "2mom-xcu": "G2M-XCU"}

# Line/marker colour per base experiment.  Kept here so every workflow
# (main plots, convective_feedback, microphysics_budget, kt273_diurnal,
# updraft_hydrometeor, ddh) renders the three models in the same palette.
EXPERIMENT_COLORS = {
    "control": "#d62728",
    "graupel": "#1f77b4",
    "2mom":    "#2ca02c",
}

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
