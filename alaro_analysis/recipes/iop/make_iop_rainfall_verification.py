#!/usr/bin/env python3
"""Rainfall verification against the SIPAM radar and GPM IMERG for the GoAmazon IOPs.

Reads the common-valid sets written by ``build_iop_rainfall_common_valid`` (radar
valid-time mask and radar footprint already applied to all six datasets) and
writes the rain figures of the IOP series (``iop_settings.figure_files``): one folder per
figure, ``<NN>_<name>/<stem>_<period>_450dpi.jpg`` with the plotted data in
``data/<stem>_<period>.txt``, for the periods combined, iop1 and iop2.

    spatial bias vs radar     spatial bias vs imerg     mean diurnal cycle
    rain pdf loglog           rain pdf log-linear       rmse vs radar / imerg

plus two paper panels: the bias maps against the radar over the RMSE maps
(``spatial_bias_and_rmse_vs_radar_<period>``) and the diurnal cycles of all periods side
by side (``mean_diurnal_cycle_combined_iop1_iop2``).

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
from alaro_analysis.common.experiments import DIAG_CONTROL_SHOWN, shown_runs
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
# Panel shapes by number of panels: (layout, figsize); four panels go 2x2.
BIAS_SHAPES = {6: ((2, 3), (18.0, 10.5)), 5: ((2, 3), (16.5, 10.5)), 4: ((2, 2), (12.0, 10.0))}
RMSE_SHAPES = {6: ((2, 3), (19.5, 11.0)), 5: ((2, 3), (16.5, 10.5)), 4: ((2, 2), (13.0, 10.5))}


def _maps(stem: str, reference: str, candidates: tuple[str, ...], title_suffix: str, shapes: dict) -> dict:
    """One map figure: the runs shown in this set, laid out by their number."""
    candidates = shown_runs(candidates)
    layout, figsize = shapes[len(candidates)]
    return {"stem": stem, "reference": reference, "candidates": candidates,
            "title_suffix": title_suffix, "figsize": figsize, "layout": layout}


MODELS = ("nodiag", "control", "graupel", "2mom", "no3m")
BIAS_MAPS = (
    _maps("spatial_bias_vs_radar", "radar", ("imerg", *MODELS), "Radar", BIAS_SHAPES),
    _maps("spatial_bias_vs_imerg", "imerg", MODELS, "IMERG", BIAS_SHAPES),
)
# Hourly RMSE maps against each reference, same panels as the bias maps.
RMSE_MAPS = (
    _maps("rmse_vs_radar", "radar", ("imerg", *MODELS), "Radar", RMSE_SHAPES),
    _maps("rmse_vs_imerg", "imerg", MODELS, "IMERG", RMSE_SHAPES),
)
UTC_OFFSET_HOURS = -4
MIN_REFERENCE_MM_H = 0.01
PDF_MIN_MM_H = 0.1
PDF_BINS = 99
DIURNAL_SHADING = {"shade_mode": "percent", "std_multiplier": 1.0, "percent_uncertainty": 0.10}
RUN_NOTE = (
    "C1M_diaggrp: continuous control run on the IOP dates. C1M_nodiaggrp, G1M, G2M, G2M-XCU: IOP warm-restart runs."
    if DIAG_CONTROL_SHOWN else
    "C1M: control without the diagnostic graupel. C1M, G1M, G2M, G2M-XCU: IOP warm-restart runs."
)


def figure_outputs(root: Path, stem: str, period: str) -> tuple[Path, Path, Path]:
    """Temporary PNG render, final JPG, and data text of one figure."""
    return figure_files(root, stem, stem, period)


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
        RUN_NOTE,
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


# Written under the colour bar of the bias maps.
PERIOD_LABELS = {"combined": "Combined IOPs", "iop1": "IOP1", "iop2": "IOP2"}


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
            period_label=PERIOD_LABELS[period],
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


def rmse_limit(maps: Iterable[np.ndarray]) -> float:
    """Upper colour limit of one RMSE figure: the p98 of the cell RMSE, rounded up to 0.5 mm/h."""
    values = np.concatenate([m[np.isfinite(m)].ravel() for m in maps])
    return float(0.5 * np.ceil(np.percentile(values, 98.0) / 0.5)) if values.size else 1.0


def make_rmse_maps(data_dir: Path, root: Path, period: str, header: list[str], dpi: int) -> None:
    """Hourly RMSE maps against the radar and against IMERG (supplementary figures)."""
    fields = bias_wf.read_all_spatial_means(data_dir)
    for spec in RMSE_MAPS:
        png, jpg, txt = figure_outputs(root, spec["stem"], period)
        maps = bias_wf.build_rmse_maps(
            fields, reference_key=spec["reference"], comparison_keys=spec["candidates"]
        )
        vmax = rmse_limit(maps.values())
        bias_wf.plot_bias_maps(
            fields,
            maps,
            reference_key=spec["reference"],
            output_path=png,
            title_suffix=spec["title_suffix"],
            figsize=spec["figsize"],
            layout=spec["layout"],
            vmin=0.0,
            vmax=vmax,
            dpi=dpi,
            period_label=PERIOD_LABELS[period],
            cmap="viridis",
            colorbar_label=r"RMSE (mm h$^{-1}$)",
            title_join=" vs ",
            panel_note=lambda data: f"RMSE: {bias_wf.pooled_rmse(data):.2f} mm/h",
        )
        bias_wf.write_rmse_txt(
            txt, figure_path=jpg, fields=fields, rmse_maps=maps,
            reference_key=spec["reference"], vmax=vmax,
        )
        finish(png, jpg, txt, header, dpi)


BIAS_RMSE_STEM = "spatial_bias_and_rmse_vs_radar"


def make_bias_rmse_maps(
    data_dir: Path, root: Path, period: str, header: list[str], limit: float | None, dpi: int
) -> None:
    """Bias maps against the radar on top, hourly RMSE maps below, one column per dataset."""
    spec = BIAS_MAPS[0]
    fields = bias_wf.read_all_spatial_means(data_dir)
    bias = bias_wf.build_bias_maps(fields, reference_key=spec["reference"], comparison_keys=spec["candidates"],
                                   min_reference=MIN_REFERENCE_MM_H)
    rmse = bias_wf.build_rmse_maps(fields, reference_key=spec["reference"], comparison_keys=spec["candidates"])
    bias_max = limit if limit is not None else colour_limit(bias.values())
    rmse_max = rmse_limit(rmse.values())
    rows = (
        {"maps": bias, "vmin": -bias_max, "vmax": bias_max, "cmap": "RdBu_r",
         "colorbar_label": "Relative bias (%)", "panel_note": lambda data: f"Mean: {float(np.nanmean(data)):+.1f}%"},
        {"maps": rmse, "vmin": 0.0, "vmax": rmse_max, "cmap": "viridis",
         "colorbar_label": r"RMSE (mm h$^{-1}$)",
         "panel_note": lambda data: f"RMSE: {bias_wf.pooled_rmse(data):.2f} mm/h"},
    )
    png, jpg, txt = figure_files(root, spec["stem"], BIAS_RMSE_STEM, period)
    bias_wf.plot_map_rows(fields, rows, reference_key=spec["reference"], output_path=png,
                          figsize=(3.6 * len(bias) + 1.2, 9.6), dpi=dpi, period_label=PERIOD_LABELS[period])
    bias_wf.write_bias_txt(txt, figure_path=jpg, data_dir=data_dir, fields=fields, bias_maps=bias,
                           reference_key=spec["reference"], min_reference=MIN_REFERENCE_MM_H,
                           vmin=-bias_max, vmax=bias_max)
    rmse_txt = txt.with_name(f"{txt.stem}_rmse.txt")
    bias_wf.write_rmse_txt(rmse_txt, figure_path=jpg, fields=fields, rmse_maps=rmse,
                           reference_key=spec["reference"], vmax=rmse_max)
    with txt.open("a", encoding="utf-8") as fh:
        fh.write("\n\n" + rmse_txt.read_text(encoding="utf-8"))
    rmse_txt.unlink()
    finish(png, jpg, txt, header, dpi)


def make_diurnal_cycle(data_dir: Path, root: Path, period: str, header: list[str], dpi: int) -> None:
    png, jpg, txt = figure_outputs(root, "mean_diurnal_cycle", period)
    stats = {
        cfg.key: diurnal_wf.read_domain_mean_stats(
            data_dir, cfg, utc_offset_hours=UTC_OFFSET_HOURS
        )
        for cfg in diurnal_wf.DATASETS
    }
    diurnal_wf.plot_diurnal_cycle(stats, output_path=png, dpi=dpi, period_label=PERIOD_LABELS[period],
                                  **DIURNAL_SHADING)
    diurnal_wf.write_diurnal_txt(
        txt,
        data_dir=data_dir,
        stats_by_key=stats,
        utc_offset_hours=UTC_OFFSET_HOURS,
        output_path=jpg,
        **DIURNAL_SHADING,
    )
    finish(png, jpg, txt, header, dpi)


def make_diurnal_cycle_periods(work_dir: Path, root: Path, periods: list[str], dpi: int) -> None:
    """The mean diurnal cycles of all periods side by side, combined IOPs first."""
    stats, texts = {}, []
    png, jpg, txt = figure_outputs(root, "mean_diurnal_cycle", "_".join(periods))
    for key in periods:
        data_dir = work_dir / PERIODS[key]
        stats[PERIOD_LABELS[key]] = {
            cfg.key: diurnal_wf.read_domain_mean_stats(data_dir, cfg, utc_offset_hours=UTC_OFFSET_HOURS)
            for cfg in diurnal_wf.DATASETS
        }
        part = txt.with_name(f"{txt.stem}_{key}.txt")
        diurnal_wf.write_diurnal_txt(part, data_dir=data_dir, stats_by_key=stats[PERIOD_LABELS[key]],
                                     utc_offset_hours=UTC_OFFSET_HOURS, output_path=jpg, **DIURNAL_SHADING)
        lines = part.read_text(encoding="utf-8").splitlines()
        texts.append("\n".join(["", "#" * 60, PERIOD_LABELS[key].upper(),
                                *period_header(data_dir, IOP_PERIODS[key]), *lines[2:]]))
        part.unlink()
    diurnal_wf.plot_diurnal_cycle_periods(stats, output_path=png, dpi=dpi, **DIURNAL_SHADING)
    title = f"Mean diurnal cycle of rain: {jpg.stem.replace('_450dpi', '')}"
    txt.write_text(f"{title}\n{'=' * len(title)}\n" + "\n".join(texts) + "\n", encoding="utf-8")
    convert_to_jpg(png, jpg, dpi=dpi)
    png.unlink()
    print(f"[saved] {jpg}\n[saved] {txt}", flush=True)


def make_pdfs(data_dir: Path, root: Path, period: str, header: list[str], dpi: int) -> None:
    samples = pdf_wf.read_samples(data_dir)
    edges = pdf_wf.common_log_bins(samples, lower=PDF_MIN_MM_H, n_bins=PDF_BINS)
    counts, densities = {}, {}
    for sample in samples:
        key = sample.config.key
        counts[key], densities[key] = pdf_wf.compute_unconditional_pdf(sample.values, edges)
    png, jpg, txt = figure_outputs(root, "rain_pdf", period)
    pdf_wf.plot_pdf_panels(samples, edges=edges, densities=densities, output_path=png, dpi=dpi,
                           period_label=PERIOD_LABELS[period])
    pdf_wf.write_pdf_txt(
        txt,
        figure_path=jpg,
        data_dir=data_dir,
        samples=samples,
        edges=edges,
        densities=densities,
        counts=counts,
        min_threshold=PDF_MIN_MM_H,
        x_axis="log in (a), linear in (b)",
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
        make_rmse_maps(data_dir, root, key, header, args.dpi)
        make_bias_rmse_maps(data_dir, root, key, header, args.bias_limit, args.dpi)
        make_diurnal_cycle(data_dir, root, key, header, args.dpi)
        make_pdfs(data_dir, root, key, header, args.dpi)
    if len(args.periods) > 1:
        make_diurnal_cycle_periods(args.work_dir, args.output_dir, args.periods, args.dpi)
    write_figure_index(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
