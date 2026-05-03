"""Shared constants, IO helpers, and plotting primitives for the ddh/ package.

Anything used by more than one of aggregate_budgets, extract_temperature,
plot_budgets, plot_case_study lives here so each script stays focused on its
own orchestration logic.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.ticker as mticker
import numpy as np

from ..common.constants import (
    CP_D as CP_DRY,
    EXPERIMENT_COLORS as EXP_COLORS,
    EXPERIMENT_LABELS,
    FREEZING_K as T_FREEZE_K,
)
from .plot_style import process_color

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
UNTAR_ROOT     = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-untar")
PROCESSED_BASE = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed")
AGG_DIR        = PROCESSED_BASE / "_aggregated"
FIG_DIR        = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/DDH-figures")
LOG_ROOT       = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/alaro-analysis/cache/logs")

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
# The ddh pipeline runs on the three base experiments.  Label map reuses
# :data:`alaro_analysis.common.constants.EXPERIMENT_LABELS` so adding a new
# variant only requires editing the shared table.
EXPERIMENTS: dict[str, str] = {
    k: EXPERIMENT_LABELS[k] for k in ("control", "graupel", "2mom")
}
# EXP_COLORS imported from common.constants.EXPERIMENT_COLORS (see import above).
SPECIES_ALL: tuple[str, ...] = ("QV", "QL", "QI", "QR", "QS", "QG")

# ---------------------------------------------------------------------------
# Physical constants & plot limits
# ---------------------------------------------------------------------------
# T_FREEZE_K and CP_DRY come from common.constants (see import above).
Z_MAX_KM   = 20.0

# ---------------------------------------------------------------------------
# Budget-component colouring and human-readable labels
# ---------------------------------------------------------------------------
BLOCK_COLORS: dict[str, str] = {
    "cond-cv":     process_color("Condensation"),
    "cond-rs":     process_color("Condensation"),
    "evap-cv":     process_color("Evaporation"),
    "evap-rs":     process_color("Evaporation"),
    "auto-cv":     process_color("Autoconversion"),
    "auto-rs":     process_color("Autoconversion"),
    "prec-cv":     process_color("Precipitation"),
    "prec-rs":     process_color("Precipitation"),
    "turdiff":     process_color("Turbulence (diff)"),
    "turconv":     process_color("Turbulence (conv)"),
    "dynam":       process_color("Dynamics"),
    "neg":         process_color("Negativity correction"),
    # Residuals / component sums across all species share a neutral colour.
    "TQVRESIDUAL": "#bcbd22", "TQLRESIDUAL": "#bcbd22",
    "TQIRESIDUAL": "#bcbd22", "TQNRESIDUAL": "#bcbd22",
    "TQRRESIDUAL": "#bcbd22", "TQSRESIDUAL": "#bcbd22",
    "TQGRESIDUAL": "#bcbd22",
    "TQLCOMPSUM":  "#888888", "TQICOMPSUM":  "#888888",
    "TQNCOMPSUM":  "#888888", "TQRCOMPSUM":  "#888888",
    "TQSCOMPSUM":  "#888888", "TQGCOMPSUM":  "#888888",
}
BLOCK_LABELS: dict[str, str] = {
    "cond-cv":   "condensation (convective)",
    "cond-rs":   "condensation (resolved)",
    "evap-cv":   "evaporation (convective)",
    "evap-rs":   "evaporation (resolved)",
    "auto-cv":   "autoconversion (convective)",
    "auto-rs":   "autoconversion (resolved)",
    "prec-cv":   "precipitation flux (convective)",
    "prec-rs":   "precipitation flux (resolved)",
    "turdiff":   "turbulent diffusion",
    "turconv":   "turbulent convection",
    "dynam":     "dynamics",
    "neg":       "negative correction",
}


def pretty_block_label(block: str) -> str:
    """Map a raw fbl-block name to a human-readable label.

    Falls back to a descriptive name for residual / component-sum blocks and
    finally to the raw name if nothing else matches.
    """
    if block in BLOCK_LABELS:
        return BLOCK_LABELS[block]
    if block.endswith("RESIDUAL"):
        return "residual"
    if block.endswith("COMPSUM"):
        return "sum of components"
    return block


# ---------------------------------------------------------------------------
# File IO
# ---------------------------------------------------------------------------

def read_dta(path: Path, ycoor: str = "VZ") -> tuple[np.ndarray, np.ndarray]:
    """Parse a two-column ddhb .dta file.

    The first column is the vertical coordinate and the second is the value.
    With ``ycoor='VZ'`` the coordinate is altitude in km (top-first, positive).
    With ``ycoor='VP'`` it is pressure in hPa with a display minus sign that
    we flip so the returned value is positive and increases downward.
    """
    arr = np.loadtxt(path)
    if ycoor == "VZ":
        coord = arr[:, 0]
    else:
        coord = -arr[:, 0]
    return coord.astype(np.float64), arr[:, 1].astype(np.float64)


def load_budget(exp: str, var: str, lead: str = "0024") -> dict | None:
    """Load the aggregated npz for one experiment and species.

    Returns ``{"altitude_km": (n_lev,), "blocks": dict, "n_days": int}`` or
    ``None`` if the file does not exist.
    """
    path = AGG_DIR / f"lead{lead}_VZ" / f"{exp}_{var}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    alt = d["altitude_km"] if "altitude_km" in d.files else None
    blocks = {k[len("block__"):]: d[k]
              for k in d.files if k.startswith("block__")}
    return {
        "altitude_km": alt,
        "blocks":      blocks,
        "n_days":      int(d["days"].shape[0]),
    }


def load_temperature(exp: str) -> dict | None:
    """Load the annual-mean temperature profile produced by extract_temperature."""
    path = AGG_DIR / f"temperature_{exp}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {
        "altitude_km":   d["altitude_km"],
        "temperature_k": d["temperature_k"],
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def tick_formatter() -> mticker.FuncFormatter:
    """`%g` tick formatter (strips trailing zeros).  Apply to every axis."""
    return mticker.FuncFormatter(lambda v, _: f"{v:g}")


def freezing_level_km(temp: dict | None) -> float:
    """Altitude (km) where T_mean crosses 273.15 K, via linear interpolation.

    Returns NaN if the profile never crosses.  Picks the lowest crossing by
    altitude (the conventional freezing level, not a stratospheric crossing).
    """
    if temp is None:
        return float("nan")
    z = np.asarray(temp["altitude_km"],   dtype=np.float64)
    t = np.asarray(temp["temperature_k"], dtype=np.float64)
    m = np.isfinite(z) & np.isfinite(t)
    z, t = z[m], t[m]
    if z.size < 2:
        return float("nan")
    order = np.argsort(z)
    z, t = z[order], t[order]
    diff = t - T_FREEZE_K
    crossings = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if crossings.size == 0:
        return float("nan")
    i = int(crossings[0])
    w = diff[i] / (diff[i] - diff[i + 1])
    return float(z[i] + w * (z[i + 1] - z[i]))


def set_altitude_axis(ax, z_max: float = Z_MAX_KM) -> None:
    """Standardised altitude y-axis from 0 up to ``z_max`` km."""
    ax.set_ylim(0, z_max)
    ax.set_ylabel("Altitude (km)")


def draw_freeze_lines(ax, temps: dict[str, dict]) -> None:
    """Draw a per-experiment dashed line at each mean 0 C altitude.

    Colouring follows :data:`EXP_COLORS` so the lines match the experiment
    colour in the same panel.  A neutral dashed stub is added to the legend.
    """
    for exp, tdata in temps.items():
        z0 = freezing_level_km(tdata)
        if np.isfinite(z0):
            ax.axhline(z0, color=EXP_COLORS.get(exp, "k"), lw=1.0, ls="--",
                       alpha=0.7, zorder=1)
    ax.plot([], [], color="k", lw=1.0, ls="--", alpha=0.7,
            label=r"0 $^{\circ}$C isotherm")
