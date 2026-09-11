"""CFAD-style DSD distributions sampled only in strong updraft regions."""

from __future__ import annotations

import argparse
from pathlib import Path

from alaro_analysis.common.cli_config import add_config_argument, parse_configured_args
from alaro_analysis.common.constants import EXPERIMENTS
from alaro_analysis.common.dsd import MP_FIXED_N0_PER_M3_MM
from alaro_analysis.data.strong_cfad_exports import write_data_txt
from alaro_analysis.dsd.strong_cfad import (
    CACHE_DIR,
    DATA_TXT_DIR,
    FIGURE_NAME,
    OUTPUT_DIR,
    TEXT_NAME,
    build_cfads,
)
from alaro_analysis.dsd.surface import MASK_FILE, NETCDF_ROOT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build D0/log10(Nw) CFADs using only strong updraft regions "
            "identified from UD_OMEGA and UD_MESH_FRAC."
        )
    )
    add_config_argument(parser)
    parser.add_argument("--netcdf-root", type=Path, default=NETCDF_ROOT)
    parser.add_argument("--mask-file", type=Path, default=MASK_FILE)
    parser.add_argument("--mask-var", default=None)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data-txt-dir", type=Path, default=DATA_TXT_DIR)
    parser.add_argument(
        "--lead",
        default="all",
        help="Forecast leads used for D0/log10(Nw). Use 'all' for all model timesteps.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS),
        choices=list(EXPERIMENTS),
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--tasks-per-child", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--min-qr", type=float, default=1.0e-7)
    parser.add_argument(
        "--onemom-closure",
        choices=("abel_boutle", "fixed_n0"),
        default="abel_boutle",
    )
    parser.add_argument(
        "--twomom-closure",
        choices=("gamma_mu1", "marshall_palmer"),
        default="gamma_mu1",
    )
    parser.add_argument("--n0-fixed", type=float, default=MP_FIXED_N0_PER_M3_MM)
    parser.add_argument(
        "--convection-metric",
        choices=("omega", "flux"),
        default="omega",
        help=(
            "Strong-convection mask. 'omega' uses -UD_OMEGA in Pa/s; "
            "'flux' uses (-UD_OMEGA * UD_MESH_FRAC) / g in kg m-2 s-1."
        ),
    )
    parser.add_argument(
        "--min-updraft-pa-s",
        type=float,
        default=10.0,
        help="Minimum upward pressure velocity for --convection-metric=omega.",
    )
    parser.add_argument(
        "--min-updraft-flux",
        type=float,
        default=0.01,
        help="Minimum updraft mass flux for --convection-metric=flux.",
    )
    parser.add_argument(
        "--min-updraft-mesh-frac",
        type=float,
        default=0.0,
        help="Minimum active updraft mesh fraction required for either metric.",
    )
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--d0-range", type=float, nargs=2, default=(0.0, 2.0))
    parser.add_argument("--lognw-range", type=float, nargs=2, default=(0.0, 9.0))
    parser.add_argument("--max-height-km", type=float, default=9.0)
    parser.add_argument("--dpi", type=int, default=350)
    parser.add_argument("--recompute", action="store_true")
    return parse_configured_args(parser, "dsd_cfad_strong_convection", argv=argv)


def main(argv: list[str] | None = None) -> None:
    from alaro_analysis.plotting.dsd_cfad import plot_cfads

    args = parse_args(argv)
    cfads, lead_tag, settings = build_cfads(args)
    output_path = args.output_dir / FIGURE_NAME
    text_path = args.data_txt_dir / TEXT_NAME
    plot_cfads(
        output_path,
        cfads,
        args.experiments,
        max_height_km=args.max_height_km,
        dpi=args.dpi,
    )
    write_data_txt(
        text_path,
        cfads,
        args.experiments,
        lead_tag=lead_tag,
        args=args,
        settings=settings,
    )
    print(f"Saved figure: {output_path}", flush=True)
    print(f"Saved data:   {text_path}", flush=True)


def __getattr__(name):
    """Compatibility for pre-refactor imports; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    main()
