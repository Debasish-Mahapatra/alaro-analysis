"""Plot ARM radiosonde and matched ALARO thermodynamic profiles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.radiosonde_settings import (
    DEFAULT_ALARO_ROOT,
    DEFAULT_OUTPUT_SUBDIR,
    DEFAULT_SONDE_ROOT,
)


def build_parser() -> argparse.ArgumentParser:
    from alaro_analysis.plotting.radiosondes import (
        DEFAULT_OUTPUT_DIR,
        PANEL_VARS,
    )

    parser = argparse.ArgumentParser(
        description="Plot matched ARM sonde and ALARO profiles for tdry, dewpoint, and RH."
    )
    parser.add_argument("--alaro-root", default=str(DEFAULT_ALARO_ROOT))
    parser.add_argument("--input-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument(
        "--model-source",
        choices=["point", "masked-area"],
        default="point",
        help="Use nearest-neighbor extracted profiles or radar-masked domain area averages.",
    )
    parser.add_argument(
        "--sonde-root",
        default=str(DEFAULT_SONDE_ROOT),
        help="ARM sonde directory used by --model-source masked-area.",
    )
    parser.add_argument(
        "--masked-netcdf-subdir",
        default="masked-netcdf",
        help="Experiment subdirectory containing radar-masked NetCDF files.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["control", "graupel", "2mom"],
        help="Experiments or aliases to plot: c1m/control g1m/graupel g2m/2mom.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=list(PANEL_VARS),
        default=None,
        help="Subset of variables to plot: tdry dewpoint rh.",
    )
    parser.add_argument("--stat", choices=["mean", "median"], default="mean")
    parser.add_argument("--max-height-m", type=float, default=20000.0)
    parser.add_argument("--height-step-m", type=float, default=250.0)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel readers for --model-source masked-area.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Progress print interval for --model-source masked-area.",
    )
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument(
        "--layout",
        choices=["separate", "combined", "both"],
        default="separate",
        help="separate: one figure per variable; combined: one multi-panel figure; both.",
    )
    parser.add_argument(
        "--combined-filename",
        default="radiosonde_profiles_panels",
        help="Base filename (no extension) for the combined multi-panel figure.",
    )
    parser.add_argument(
        "--diff-vars",
        nargs="*",
        choices=list(PANEL_VARS),
        default=None,
        help=(
            "Combined-layout panels to render as model-minus-ARM-sonde differences "
            "(so small variation is visible). Example: --diff-vars tdry dewpoint."
        ),
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Do not draw a title on the plot.",
    )
    parser.add_argument(
        "--write-txt",
        action="store_true",
        help="Write the plotted profile data to output-dir/data_txt with matching stems.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the separate and/or combined figures.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from alaro_analysis.plotting.radiosondes import (
        plot_combined_profiles,
        plot_profiles,
    )

    parser = build_parser()
    args = parse_configured_args(parser, "plot_radiosonde_profiles", argv=argv)
    layout = getattr(args, "layout", "separate")
    outputs: list[Path] = []
    if layout in ("separate", "both"):
        outputs.extend(plot_profiles(args))
    if layout in ("combined", "both"):
        outputs.extend(plot_combined_profiles(args))
    for output in outputs:
        print(f"Wrote {output}")
    return 0


def __getattr__(name):
    """Compatibility for pre-refactor imports; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    raise SystemExit(main())
