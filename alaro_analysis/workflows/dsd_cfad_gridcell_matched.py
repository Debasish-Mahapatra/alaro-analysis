"""Grid-cell DSD CFADs for full-domain and strong-updraft samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from alaro_analysis.common.cli_config import add_config_argument, parse_configured_args
from alaro_analysis.common.constants import EXPERIMENTS
from alaro_analysis.common.dsd import MP_FIXED_N0_PER_M3_MM
from alaro_analysis.data.gridcell_cfad_exports import write_data_txt
from alaro_analysis.dsd.gridcell_cfad import (
    CACHE_DIR,
    DATA_TXT_DIR,
    FULL_FIGURE_NAME,
    FULL_TEXT_NAME,
    OUTPUT_DIR,
    STRONG_FIGURE_NAME,
    STRONG_TEXT_NAME,
    build_pairs,
)
from alaro_analysis.dsd.surface import MASK_FILE, NETCDF_ROOT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build matched full-domain and strong-updraft CFADs from grid-cell "
            "D0/log10(Nw), not from domain-mean DSD profiles."
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
    parser.add_argument("--tasks-per-child", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=50)
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
    )
    parser.add_argument("--min-updraft-pa-s", type=float, default=10.0)
    parser.add_argument("--min-updraft-flux", type=float, default=0.01)
    parser.add_argument("--min-updraft-mesh-frac", type=float, default=0.0)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--d0-range", type=float, nargs=2, default=(0.0, 2.0))
    parser.add_argument("--lognw-range", type=float, nargs=2, default=(0.0, 9.0))
    parser.add_argument("--max-height-km", type=float, default=9.0)
    parser.add_argument("--dpi", type=int, default=350)
    parser.add_argument(
        "--frequency-percentile",
        type=float,
        default=99.0,
        help=(
            "Percentile of positive frequency values used for the matched "
            "row colorbar scale. Use 100 for the literal maximum."
        ),
    )
    parser.add_argument("--recompute", action="store_true")
    return parse_configured_args(parser, "dsd_cfad_gridcell_matched", argv=argv)


def main(argv: list[str] | None = None) -> None:
    from alaro_analysis.plotting.dsd_cfad import plot_cfads
    from alaro_analysis.plotting.dsd_gridcell import matched_vmax_by_field

    args = parse_args(argv)
    full_cfads, strong_cfads, lead_tag, settings = build_pairs(args)
    row_vmax_values = matched_vmax_by_field(
        full_cfads,
        strong_cfads,
        args.experiments,
        float(args.frequency_percentile),
    )
    print(
        "Matched frequency vmax (%): "
        + ", ".join(f"{field}={value:g}" for field, value in row_vmax_values.items()),
        flush=True,
    )

    full_output = args.output_dir / FULL_FIGURE_NAME
    strong_output = args.output_dir / STRONG_FIGURE_NAME
    full_text = args.data_txt_dir / FULL_TEXT_NAME
    strong_text = args.data_txt_dir / STRONG_TEXT_NAME
    plot_cfads(
        full_output,
        full_cfads,
        args.experiments,
        max_height_km=args.max_height_km,
        dpi=args.dpi,
        row_vmax_values=row_vmax_values,
    )
    plot_cfads(
        strong_output,
        strong_cfads,
        args.experiments,
        max_height_km=args.max_height_km,
        dpi=args.dpi,
        row_vmax_values=row_vmax_values,
    )
    write_data_txt(
        full_text,
        full_cfads,
        args.experiments,
        title="Grid-cell full-domain CFAD-style DSD diagnostic distributions",
        lead_tag=lead_tag,
        args=args,
        settings=settings,
        row_vmax_values=row_vmax_values,
    )
    write_data_txt(
        strong_text,
        strong_cfads,
        args.experiments,
        title="Grid-cell strong-convection CFAD-style DSD diagnostic distributions",
        lead_tag=lead_tag,
        args=args,
        settings=settings,
        row_vmax_values=row_vmax_values,
    )
    print(f"Saved full-domain figure:        {full_output}", flush=True)
    print(f"Saved strong-convection figure: {strong_output}", flush=True)
    print(f"Saved full-domain data:          {full_text}", flush=True)
    print(f"Saved strong-convection data:   {strong_text}", flush=True)


def __getattr__(name):
    """Compatibility for pre-refactor imports; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    main()
