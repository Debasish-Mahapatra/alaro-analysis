"""Vertical profiles of DSD diagnostics and DDH QV for C1M, G1M, and G2M."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from alaro_analysis.common.cli_config import add_config_argument, parse_configured_args
from alaro_analysis.common.constants import EXPERIMENTS
from alaro_analysis.common.dsd import MP_FIXED_N0_PER_M3_MM
from alaro_analysis.data.dsd_profile_exports import write_data_txt
from alaro_analysis.ddh.io import AGG_DIR as DDH_AGG_DIR
from alaro_analysis.dsd.surface import MASK_FILE, NETCDF_ROOT
from alaro_analysis.dsd.vertical_profiles import (
    CACHE_DIR,
    DATA_TXT_DIR,
    FIGURE_NAME,
    OUTPUT_DIR,
    TEXT_NAME,
    build_profiles,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from alaro_analysis.plotting.dsd_profiles import PANEL_SPECS

    parser = argparse.ArgumentParser(
        description=("Plot vertical profiles of D0, log10(Nw), and DDH QV for C1M, G1M, and G2M.")
    )
    add_config_argument(parser)
    parser.add_argument("--netcdf-root", type=Path, default=NETCDF_ROOT)
    parser.add_argument("--ddh-agg-dir", type=Path, default=DDH_AGG_DIR)
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
        "--ddh-lead",
        default=None,
        help="DDH lead used for the QV panel. Defaults to 0024.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS),
        choices=list(EXPERIMENTS),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tasks-per-child", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=100)
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
        help="DSD closure for G2M from prognostic rain mass and number.",
    )
    parser.add_argument("--n0-fixed", type=float, default=MP_FIXED_N0_PER_M3_MM)
    parser.add_argument("--qv-block", default="VQVM")
    parser.add_argument(
        "--qv-label",
        default=PANEL_SPECS[-1][2],
        help="X-axis label for the DDH QV panel.",
    )
    parser.add_argument("--max-height-km", type=float, default=9.0)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--recompute", action="store_true")
    return parse_configured_args(parser, "dsd_vertical_profiles", argv=argv)


def main(argv: list[str] | None = None) -> None:
    from alaro_analysis.plotting.dsd_profiles import plot_profiles

    args = parse_args(argv)
    dsd_profiles, qv_profiles, lead_tag, ddh_lead = build_profiles(args)

    output_path = args.output_dir / FIGURE_NAME
    text_path = args.data_txt_dir / TEXT_NAME
    plot_profiles(
        output_path,
        dsd_profiles,
        qv_profiles,
        args.experiments,
        qv_label=args.qv_label,
        max_height_km=args.max_height_km,
        dpi=args.dpi,
    )
    write_data_txt(
        text_path,
        dsd_profiles,
        qv_profiles,
        args.experiments,
        lead_tag=lead_tag,
        ddh_lead=ddh_lead,
        onemom_closure=args.onemom_closure,
        twomom_closure=args.twomom_closure,
        qv_block=args.qv_block,
    )
    print(f"Saved figure: {output_path}", flush=True)
    print(f"Saved data:   {text_path}", flush=True)


def __getattr__(name):
    """Compatibility for pre-refactor imports; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    main()
