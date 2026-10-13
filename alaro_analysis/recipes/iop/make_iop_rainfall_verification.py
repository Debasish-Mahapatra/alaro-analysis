#!/usr/bin/env python3
"""Rainfall verification against the SIPAM radar and GPM IMERG for the GoAmazon IOPs.

Reads the common-valid sets written by ``build_iop_rainfall_common_valid`` (radar
valid-time mask and radar footprint already applied to all six datasets) and
writes figures 1-5 of the IOP series (``iop_settings.figure_files``): one folder per
figure, ``<NN>_<name>/<stem>_<period>_450dpi.jpg`` with the plotted data in
``data/<stem>_<period>.txt``, for the periods combined, iop1 and iop2.

    01 spatial bias vs radar     02 spatial bias vs imerg     03 mean diurnal cycle
    04 rain pdf loglog           05 rain pdf log-linear

These are the only figures that use the 0.1 deg radar mask.

The figures are drawn by the same functions as the paper's rain figures.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.figio import DEFAULT_DPI, convert_to_jpg
from alaro_analysis.common.iop_settings import (
    FIGURES,
    GOAMAZON_IOPS,
    IOP_PERIODS,
    figure_files,
    write_figure_index,
)
from alaro_analysis.recipes.iop.build_iop_rainfall_common_valid import (
    COMMON_SUBDIR,
    DEFAULT_WORK_DIR,
)
from alaro_analysis.workflows import precip_distribution as pdf_wf
from alaro_analysis.workflows import precip_diurnal_cycle as diurnal_wf
from alaro_analysis.workflows import precip_spatial_bias_maps as bias_wf

# Period -> common-valid subdirectory
PERIODS = {
    "combined": COMMON_SUBDIR,
    "iop1": f"{COMMON_SUBDIR}-iop1",
    "iop2": f"{COMMON_SUBDIR}-iop2",
}
FIGURE_NUMBERS = {
    "spatial_bias_vs_radar": 1,
    "spatial_bias_vs_imerg": 2,
    "mean_diurnal_cycle": 3,
    "rain_pdf_loglog": 4,
    "rain_pdf_log_linear": 5,
}
BIAS_MAPS = (
    {
        "stem": "spatial_bias_vs_radar",
        "reference": "radar",
        "candidates": ("imerg", "control", "graupel", "2mom", "no3m"),
        "title_suffix": "Radar",
        "figsize": (16.5, 10.5),
        "layout": (2, 3),
    },
    {
        "stem": "spatial_bias_vs_imerg",
        "reference": "imerg",
        "candidates": ("control", "graupel", "2mom", "no3m"),
        "title_suffix": "IMERG",
        "figsize": (12.0, 10.0),
        "layout": (2, 2),
    },
)
UTC_OFFSET_HOURS = -4
MIN_REFERENCE_MM_H = 0.01
PDF_MIN_MM_H = 0.1
PDF_BINS = 99
DIURNAL_SHADING = {"shade_mode": "percent", "std_multiplier": 1.0, "percent_uncertainty": 0.10}


def figure_outputs(root: Path, stem: str, period: str) -> tuple[Path, Path, Path]:
    """Temporary PNG render, final JPG, and data text of one figure."""
    return figure_files(root, FIGURE_NUMBERS[stem], stem, period)


def period_header(data_dir: Path, iops: tuple[int, ...]) -> list[str]:
    with xr.open_dataset(data_dir / "Radar_common_valid.nc") as ds:
        times = np.asarray(ds["time"].values, dtype="datetime64[h]")
    names = " and ".join(f"IOP{iop}" for iop in iops)
    combined = " combined" if len(iops) > 1 else ""
    lines = [f"Period: GoAmazon2014/5 {names}{combined} (Martin et al. 2016, ACP 16, 4785)."]
    for iop in iops:
        start, end = GOAMAZON_IOPS[iop]
        lo = np.datetime64(start, "h")
        hi = np.datetime64(end, "h") + np.timedelta64(23, "h")
        n = int(np.count_nonzero((times >= lo) & (times <= hi)))
        total = 24 * ((end - start).days + 1)
        lines.append(f"IOP{iop}: {start} to {end} inclusive, {n} common-valid hours of {total}.")
    lines += [
        f"Common-valid hours: {times.size}. The radar valid-time mask and radar footprint "
        "are applied to every dataset.",
        "C1M: continuous control run on the IOP dates. G1M, G2M, G2M-XCU: IOP warm-restart runs.",
    ]
    return lines


def finish(png: Path, jpg: Path, txt: Path, header: list[str], dpi: int) -> None:
    convert_to_jpg(png, jpg, dpi=dpi)
    png.unlink()
    # Keep each writer's two-line title first, then the period description. The
    # bias writer titles the file after the JPG stem; drop its resolution suffix.
    lines = txt.read_text(encoding="utf-8").splitlines(keepends=True)
    title = lines[0].rstrip("\n").replace("_450dpi", "")
    body = f"{title}\n{'=' * len(title)}\n" + "\n".join(header) + "\n" + "".join(lines[2:])
    txt.write_text(body, encoding="utf-8")
    print(f"[saved] {jpg}\n[saved] {txt}", flush=True)


def colour_limit(maps: Iterable[np.ndarray]) -> float:
    """Symmetric bias limit of one figure: the p98 of |bias| over its maps, rounded up to 10 %.

    IOP biases span +/-30 % to several hundred percent (IOP2 against a dry IMERG),
    so a limit taken from the data keeps about 2 % of the cells saturated.
    """
    values = np.concatenate([np.abs(m[np.isfinite(m)]).ravel() for m in maps])
    if values.size == 0:
        return 30.0
    return float(10.0 * np.ceil(np.percentile(values, 98.0) / 10.0))


def make_bias_maps(
    data_dir: Path, root: Path, period: str, header: list[str], limit: float | None, dpi: int
) -> None:
    """Draw both bias figures; without ``limit`` each figure takes its own from the data."""
    fields = bias_wf.read_all_spatial_means(data_dir)
    for spec in BIAS_MAPS:
        png, jpg, txt = figure_outputs(root, spec["stem"], period)
        maps = bias_wf.build_bias_maps(
            fields,
            reference_key=spec["reference"],
            comparison_keys=spec["candidates"],
            min_reference=MIN_REFERENCE_MM_H,
        )
        vmax = limit if limit is not None else colour_limit(maps.values())
        bias_wf.plot_bias_maps(
            fields,
            maps,
            reference_key=spec["reference"],
            output_path=png,
            title_suffix=spec["title_suffix"],
            figsize=spec["figsize"],
            layout=spec["layout"],
            vmin=-vmax,
            vmax=vmax,
            dpi=dpi,
        )
        bias_wf.write_bias_txt(
            txt,
            figure_path=jpg,
            data_dir=data_dir,
            fields=fields,
            bias_maps=maps,
            reference_key=spec["reference"],
            min_reference=MIN_REFERENCE_MM_H,
            vmin=-vmax,
            vmax=vmax,
        )
        finish(png, jpg, txt, header, dpi)


def make_diurnal_cycle(data_dir: Path, root: Path, period: str, header: list[str], dpi: int) -> None:
    png, jpg, txt = figure_outputs(root, "mean_diurnal_cycle", period)
    stats = {
        cfg.key: diurnal_wf.read_domain_mean_stats(
            data_dir, cfg, utc_offset_hours=UTC_OFFSET_HOURS
        )
        for cfg in diurnal_wf.DATASETS
    }
    diurnal_wf.plot_diurnal_cycle(stats, output_path=png, dpi=dpi, **DIURNAL_SHADING)
    diurnal_wf.write_diurnal_txt(
        txt,
        data_dir=data_dir,
        stats_by_key=stats,
        utc_offset_hours=UTC_OFFSET_HOURS,
        output_path=jpg,
        **DIURNAL_SHADING,
    )
    finish(png, jpg, txt, header, dpi)


def make_pdfs(data_dir: Path, root: Path, period: str, header: list[str], dpi: int) -> None:
    samples = pdf_wf.read_samples(data_dir)
    edges = pdf_wf.common_log_bins(samples, lower=PDF_MIN_MM_H, n_bins=PDF_BINS)
    counts, densities = {}, {}
    for sample in samples:
        key = sample.config.key
        counts[key], densities[key] = pdf_wf.compute_unconditional_pdf(sample.values, edges)
    for stem, xscale_log, x_axis in (
        ("rain_pdf_loglog", True, "log"),
        ("rain_pdf_log_linear", False, "linear"),
    ):
        png, jpg, txt = figure_outputs(root, stem, period)
        pdf_wf.plot_pdf(
            samples,
            edges=edges,
            densities=densities,
            output_path=png,
            xscale_log=xscale_log,
            yscale_log=True,
            show_tail_marker=False,
            dpi=dpi,
        )
        pdf_wf.write_pdf_txt(
            txt,
            figure_path=jpg,
            data_dir=data_dir,
            samples=samples,
            edges=edges,
            densities=densities,
            counts=counts,
            min_threshold=PDF_MIN_MM_H,
            x_axis=x_axis,
            y_axis="log",
        )
        finish(png, jpg, txt, header, dpi)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--output-dir", type=Path, default=FIGURES)
    p.add_argument("--periods", nargs="+", default=list(PERIODS), choices=list(PERIODS))
    p.add_argument(
        "--bias-limit",
        type=float,
        default=None,
        help="Fixed symmetric bias-map colour limit in percent (default: each figure's own).",
    )
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_iop_rainfall_verification", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for key in args.periods:
        data_dir = args.work_dir / PERIODS[key]
        root = args.output_dir
        header = period_header(data_dir, IOP_PERIODS[key])
        make_bias_maps(data_dir, root, key, header, args.bias_limit, args.dpi)
        make_diurnal_cycle(data_dir, root, key, header, args.dpi)
        make_pdfs(data_dir, root, key, header, args.dpi)
    write_figure_index(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
