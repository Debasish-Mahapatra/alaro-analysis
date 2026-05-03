"""Shared DDH plotting styles.

This module is the single source for the process-color convention used by DDH
budget plots: one color per physical process, with linestyle separating
resolved and convective/parametrized pathways.
"""
from __future__ import annotations

from typing import Any


PROCESS_COLORS: dict[str, str] = {
    "Dynamics": "#1B6EC2",
    "Micro": "#D93425",
    "Condensation": "#228B22",
    "Evaporation": "#9B59B6",
    "Autoconversion": "#E8891D",
    "Precipitation": "#FF6B6B",
    "Turbulence (diff)": "#00A6A6",
    "Turbulence (conv)": "#E84393",
    "Radiation (solar)": "#FFD700",
    "Radiation (thermal)": "#17B890",
    "Shear": "#6C3483",
    "Buoyancy": "#F39C12",
    "Dissipation": "#C70039",
    "Advection": "#0066CC",
    "Diffusion": "#FF8C00",
    "GWD drag": "#20B2AA",
    "Negativity correction": "#8B4513",
}

# Backwards-compatible spelling used by older scripts.
PROCESS_COLOURS = PROCESS_COLORS

TOTAL_COLOR = "#111111"
RESIDUAL_COLOR = "#666666"
SUM_COLOR = "#333333"
DEFAULT_COLOR = "#777777"
FREEZING_COLOR = "#555555"
CONVECTION_COLOR = "#D62728"
RESOLVED_COLOR = "#1F77B4"

SOLID_LINESTYLE = "-"
RESOLVED_LINESTYLE = "--"
CONVECTION_LINESTYLE = "-."
FREEZING_LINESTYLE = ":"
RESIDUAL_LINESTYLE = (0, (5, 3))
SUM_LINESTYLE = (0, (2, 2))

TOTAL_LINEWIDTH = 2.4
PROCESS_LINEWIDTH = 1.5
PARTITION_LINEWIDTH = 1.7
EXPERIMENT_COMPARISON_LINEWIDTH = 1.6

RESOLVED_ALPHA = 0.95
CONVECTION_ALPHA = 0.85
DEFAULT_ALPHA = 0.92
NEUTRAL_ALPHA = 0.85

EXPERIMENT_PANEL_FIGSIZE = (14, 6)
FIGURE_TITLE_FONTSIZE = 14
PANEL_GRID_ALPHA = 0.3
PANEL_LEGEND_FONTSIZE = 9
FREEZING_LINEWIDTH = 1.5
FREEZING_ALPHA = 0.95


def get_process_name(label: str) -> str:
    """Extract the base process name from a display label."""
    if label in ("Dynamics", "Advection", "GWD drag", "Negativity correction"):
        return label
    if label in ("Turbulence (diff)", "Turbulence (conv)"):
        return label
    if label in ("Radiation (solar)", "Radiation (thermal)"):
        return label
    if label in ("Shear", "Buoyancy", "Dissipation", "Diffusion"):
        return label
    if "(" in label and ")" in label:
        return label.split("(")[0].strip()
    return label


def process_color(process: str, default: str = DEFAULT_COLOR) -> str:
    """Return the configured color for a physical process."""
    return PROCESS_COLORS.get(process, default)


def get_line_style(label: str) -> tuple[str, float, Any, float, int]:
    """Return ``(color, linewidth, linestyle, alpha, zorder)`` for a term."""
    lower = label.lower()
    if label == "Tendency":
        return TOTAL_COLOR, TOTAL_LINEWIDTH, SOLID_LINESTYLE, 1.0, 10
    if "residual" in lower:
        return RESIDUAL_COLOR, 1.8, RESIDUAL_LINESTYLE, NEUTRAL_ALPHA, 9
    if "sum of" in lower:
        return SUM_COLOR, 1.8, SUM_LINESTYLE, NEUTRAL_ALPHA, 8

    color = process_color(get_process_name(label))
    if "resolved" in lower:
        return color, PROCESS_LINEWIDTH, RESOLVED_LINESTYLE, RESOLVED_ALPHA, 3
    if "conv" in lower or "convective" in lower:
        return color, PROCESS_LINEWIDTH, CONVECTION_LINESTYLE, CONVECTION_ALPHA, 3
    return color, PROCESS_LINEWIDTH, SOLID_LINESTYLE, DEFAULT_ALPHA, 3


def partition_line_style(
    process: str,
    pathway: str,
) -> tuple[str, float, Any, float, int]:
    """Style total/resolved/convection partitions for a single process."""
    lower = pathway.lower()
    if lower == "total":
        return TOTAL_COLOR, TOTAL_LINEWIDTH, SOLID_LINESTYLE, 1.0, 2

    if lower in {"resolved", "rs"}:
        return RESOLVED_COLOR, PARTITION_LINEWIDTH, RESOLVED_LINESTYLE, RESOLVED_ALPHA, 5
    if lower in {"convection", "convective", "cv", "parametrized"}:
        return CONVECTION_COLOR, PARTITION_LINEWIDTH, CONVECTION_LINESTYLE, CONVECTION_ALPHA, 6
    return process_color(process), PARTITION_LINEWIDTH, SOLID_LINESTYLE, DEFAULT_ALPHA, 3


def pathway_from_block(block: str) -> str:
    """Infer resolved/convection pathway from a raw DDH block name."""
    lower = block.lower()
    if lower.endswith("-cv") or lower.endswith("cv"):
        return "convection"
    if lower.endswith("-rs") or lower.endswith("rs"):
        return "resolved"
    return "default"


def pathway_line_attributes(pathway: str) -> tuple[Any, float]:
    """Return ``(linestyle, alpha)`` for a pathway name."""
    lower = pathway.lower()
    if lower in {"convection", "convective", "cv", "parametrized"}:
        return CONVECTION_LINESTYLE, CONVECTION_ALPHA
    if lower in {"resolved", "rs"}:
        return RESOLVED_LINESTYLE, RESOLVED_ALPHA
    return SOLID_LINESTYLE, DEFAULT_ALPHA
