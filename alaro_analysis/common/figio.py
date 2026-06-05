"""Shared boilerplate for the common-valid rainfall figure workflows.

Every plotting workflow used to repeat the same three things: a dataset
descriptor dataclass, the ``--data-dir``/``--output-dir``/``--dpi`` CLI
arguments, and the ``fig.savefig(..., dpi=..., bbox_inches="tight")`` call.
They live here once so a new figure script only has to describe its datasets
and its own analysis.

``--dpi`` defaults to 450, the project-standard publication resolution.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DPI = 450


@dataclass(frozen=True)
class DatasetConfig:
    """One common-valid dataset (observation or model experiment).

    ``color``/``linestyle``/``linewidth`` are only meaningful for line plots;
    map/heatmap workflows leave them at their defaults.
    """

    key: str
    label: str
    filename: str
    variable: str
    color: str = "black"
    linestyle: str = "-"
    linewidth: float = 3.0


def add_io_args(
    parser: argparse.ArgumentParser,
    *,
    default_data_dir: Path,
    default_output_dir: Path,
    default_dpi: int = DEFAULT_DPI,
) -> argparse.ArgumentParser:
    """Add the standard ``--data-dir``/``--output-dir``/``--dpi`` arguments."""
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--dpi", type=int, default=default_dpi)
    return parser


def strip_cbar_zeros(cbar, axis: str = "y") -> None:
    """Apply the project ``%g`` tick formatter to a colorbar.

    ``%g`` renders "0.0040" as "0.004" and "1.00e-05" as "1e-05". Pass
    ``axis="x"`` for horizontal colorbars, ``"y"`` (default) for vertical.
    """
    import matplotlib.ticker as mticker

    target = cbar.ax.xaxis if axis == "x" else cbar.ax.yaxis
    target.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))


def save_figure(
    fig,
    path: Path,
    *,
    dpi: int = DEFAULT_DPI,
    facecolor: str | None = None,
    tight: bool = True,
) -> Path:
    """Save ``fig`` to ``path``, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {"dpi": dpi}
    if tight:
        kwargs["bbox_inches"] = "tight"
    if facecolor is not None:
        kwargs["facecolor"] = facecolor
    fig.savefig(path, **kwargs)
    return path
