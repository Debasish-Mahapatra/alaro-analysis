"""Iop settings."""

from __future__ import annotations

import re
from datetime import date

from alaro_analysis.common.paths import runs_path

ROOT = runs_path("")
CONT = ROOT / "ALARO"
IOP = ROOT / "iop-runs"
OUT = IOP / "figures"
CACHE = IOP / "analysis" / "cache"
LOG = IOP / "analysis" / "logs"
EXPS = ("NO3M", "graupel", "2mom")
LABEL = {"NO3M": "G2M-XCU", "graupel": "G1M", "2mom": "G2M"}
VARS = (
    "SURFPREC.EAU.GEC",
    "SURFPREC.EAU.CON",
    "SURFPREC.NEI.GEC",
    "SURFPREC.NEI.CON",
    "SURFPREC.GRA.GEC",
)
PERIODS = {1: (date(2014, 1, 15), date(2014, 2, 28)), 2: (date(2014, 7, 29), date(2014, 8, 31))}
G = 9.80665
OFFSET = -4
NLEV = 87
WORKERS = 32
FH = re.compile("\\+(\\d{4})$")
