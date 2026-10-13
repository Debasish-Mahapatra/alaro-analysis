#!/usr/bin/env python3
"""Rain-rate distribution over the whole ROI against IMERG for the GoAmazon IOPs.

The log-linear PDF of figure 5 redone without the radar: IMERG and the four models
on the 0.1 deg IMERG grid over the ROI (-67..-53 E, -10..4 N), every hour of the
IOPs where all five are complete. Reads the sets written by
``build_iop_rainfall_roi`` and writes ``rain_pdf_log_linear_roi_vs_imerg_<period>``
into the figure-5 folder, with the plotted data in ``data/``.

Same bins, density and axes as figure 5. The data text adds a boundary check:
the same statistics over the ROI interior (1 deg in from every side) and over
the 1 deg edge ring, so an edge effect in the models would show up there.

Also writes two supplementary figures per IOP (``iop_settings.supplementary_files``):
S01 total rain of IMERG and the four runs, S02 their relative bias against IMERG
(same bias definition, masking and colour limit rule as figure 2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.figio import DEFAULT_DPI, DatasetConfig, convert_to_jpg
from alaro_analysis.common.iop_settings import (
    FIGURES,
    GOAMAZON_IOPS,
    IOP_PERIODS,
    figure_files,
    supplementary_files,
    write_figure_index,
)
from alaro_analysis.recipes.iop.build_iop_rainfall_roi import (
    COMMON_SUBDIR,
    DEFAULT_WORK_DIR,
    ROI_BBOX,
)
from alaro_analysis.recipes.iop.make_iop_rainfall_verification import (
    MIN_REFERENCE_MM_H,
    PDF_BINS,
    PDF_MIN_MM_H,
    colour_limit,
    finish,
)
from alaro_analysis.workflows import precip_distribution as pdf_wf

FIGURE_NUMBER = 5
STEM = "rain_pdf_log_linear_roi_vs_imerg"
PERIODS = {
    "combined": COMMON_SUBDIR,
    "iop1": f"{COMMON_SUBDIR}-iop1",
    "iop2": f"{COMMON_SUBDIR}-iop2",
}
# IMERG is the reference here, so it takes the reference style (black, solid).
DATASETS = (
    DatasetConfig("imerg", "IMERG", "IMERG_common_valid.nc", "rainfall", "black"),
    *(cfg for cfg in pdf_wf.DATASETS if cfg.key not in ("radar", "imerg")),
)
EDGE_CELLS = 10  # 1 deg on the 0.1 deg grid
MANAUS = (-60.02, -3.10)
ACCUMULATION_STEM = "rain_accumulation_roi"
BIAS_STEM = "rain_bias_vs_imerg_roi"


def read_fields(data_dir: Path) -> dict[str, np.ndarray]:
    fields = {}
    for cfg in DATASETS:
        with xr.open_dataset(data_dir / cfg.filename) as ds:
            fields[cfg.key] = np.asarray(ds[cfg.variable].values, dtype=np.float64)
    return fields


def samples_of(fields: dict[str, np.ndarray], region: np.ndarray) -> list[pdf_wf.SampleSet]:
    out = []
    for cfg in DATASETS:
        values = fields[cfg.key][:, region].ravel()
        out.append(pdf_wf.SampleSet(cfg, values[np.isfinite(values)]))
    return out


def period_header(data_dir: Path, iops: tuple[int, ...], shape: tuple[int, int]) -> list[str]:
    with xr.open_dataset(data_dir / DATASETS[0].filename) as ds:
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
        lines.append(f"IOP{iop}: {start} to {end} inclusive, {n} complete hours of {total}.")
    lo, hi, la, ha = ROI_BBOX
    lines += [
        f"Domain: whole ROI, lon {lo:g} to {hi:g}, lat {la:g} to {ha:g}, {shape[0]} x {shape[1]} "
        "cells of the 0.1 deg IMERG grid. No radar footprint and no radar time mask.",
        "The model window lies 11-15 native cells inside the C zone (the ROI itself 19-24), "
        "so no relaxation-zone or extension-zone cells enter.",
        f"Hours: {times.size}, every IOP hour where IMERG and all four models are complete.",
        "Models: hourly FA rain (CON + GEC) remapped conservatively (cdo remapcon) to the "
        "IMERG grid. IMERG: V07B Final, mean of the two half-hours of each hour.",
        "C1M: continuous control run on the IOP dates. G1M, G2M, G2M-XCU: IOP warm-restart runs.",
    ]
    return lines


def boundary_lines(fields: dict[str, np.ndarray], shape: tuple[int, int]) -> list[str]:
    """Mean rain, rain frequencies and p99.9 over the ROI, its interior and its edge ring."""
    interior = np.zeros(shape, dtype=bool)
    interior[EDGE_CELLS:-EDGE_CELLS, EDGE_CELLS:-EDGE_CELLS] = True
    regions = {"roi": np.ones(shape, dtype=bool), "interior": interior, "edge_ring": ~interior}
    lines = [
        "",
        "Boundary check",
        "--------------",
        f"interior = ROI minus {EDGE_CELLS} cells (1 deg) on every side; edge_ring = those cells.",
        "Model-minus-IMERG differences that grow toward the edge ring would point at "
        "boundary contamination.",
        "dataset,region,mean_mm_day,freq_ge_0.1_mm_h,freq_ge_1_mm_h,freq_ge_10_mm_h,p99.9_mm_h",
    ]
    for cfg in DATASETS:
        for name, region in regions.items():
            v = fields[cfg.key][:, region].ravel()
            v = v[np.isfinite(v)]
            lines.append(
                f"{cfg.label},{name},{24.0 * v.mean():.4f},{np.mean(v >= 0.1):.6f},"
                f"{np.mean(v >= 1.0):.6f},{np.mean(v >= 10.0):.7f},{np.percentile(v, 99.9):.4f}"
            )
    return lines


def make_pdf(data_dir: Path, root: Path, period: str, dpi: int) -> None:
    fields = read_fields(data_dir)
    shape = next(iter(fields.values())).shape[1:]
    samples = samples_of(fields, np.ones(shape, dtype=bool))
    edges = pdf_wf.common_log_bins(samples, lower=PDF_MIN_MM_H, n_bins=PDF_BINS)
    counts, densities = {}, {}
    for sample in samples:
        key = sample.config.key
        counts[key], densities[key] = pdf_wf.compute_unconditional_pdf(sample.values, edges)
    png, jpg, txt = figure_files(root, FIGURE_NUMBER, STEM, period)
    pdf_wf.plot_pdf(
        samples,
        edges=edges,
        densities=densities,
        output_path=png,
        xscale_log=False,
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
        x_axis="linear",
        y_axis="log",
    )
    header = period_header(data_dir, IOP_PERIODS[period], shape)
    finish(png, jpg, txt, header, dpi)
    with txt.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(boundary_lines(fields, shape)) + "\n")


def _map_axes(ax, lon, lat, field, label, note, **pcolor):
    pm = ax.pcolormesh(lon, lat, field, shading="auto", **pcolor)
    ax.plot(*MANAUS, "o", ms=5, mfc="white", mec="black")
    ax.set_title(label, fontsize=14)
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.tick_params(labelsize=11)
    return pm


def _map_txt(txt: Path, title: str, header: list[str], rows: list[str]) -> None:
    lines = [title, "=" * len(title), *header, "", *rows]
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_supplementary_maps(data_dir: Path, root: Path, period: str, dpi: int) -> None:
    """S01 total rain (IMERG + four runs, 2x3) and S02 bias against IMERG (2x2) of one IOP."""
    (iop,) = IOP_PERIODS[period]
    totals = {}
    for cfg in DATASETS:
        with xr.open_dataset(data_dir / cfg.filename) as ds:
            lon, lat = np.asarray(ds["lon"].values), np.asarray(ds["lat"].values)
            totals[cfg.label] = np.asarray(ds[cfg.variable].sum("time").values, dtype=np.float64)
            hours = int(ds.sizes["time"])
    shape = totals["IMERG"].shape
    header = period_header(data_dir, (iop,), shape)
    title_text = f"IOP{iop}"

    # S01: total rain, one colour scale for the five panels
    vmax = float(50.0 * np.ceil(np.percentile(np.concatenate([v.ravel() for v in totals.values()]), 99) / 50.0))
    png, jpg, txt = supplementary_files(root, 1, ACCUMULATION_STEM, period)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), constrained_layout=True)
    for ax, (label, total) in zip(axes.ravel(), totals.items()):
        pm = _map_axes(ax, lon, lat, total, label, f"mean {total.mean():.0f} mm",
                       cmap="YlGnBu", vmin=0.0, vmax=vmax)
    axes[1, 2].axis("off")
    cbar = fig.colorbar(pm, cax=axes[1, 2].inset_axes([0.08, 0.55, 0.84, 0.07]), orientation="horizontal")
    cbar.set_label("Total rain over the IOP (mm)", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    fig.suptitle(title_text, fontsize=18, fontweight="bold")
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    convert_to_jpg(png, jpg, dpi=dpi)
    png.unlink()
    rows = [f"Figure: {jpg}", f"Total over {hours} hours; colour scale 0 to {vmax:g} mm (p99 of all panels).",
            "dataset,domain_mean_total_mm,domain_max_total_mm"]
    rows += [f"{k},{v.mean():.2f},{v.max():.2f}" for k, v in totals.items()]
    _map_txt(txt, f"{ACCUMULATION_STEM}_{period}", header, rows)
    print(f"[saved] {jpg}", flush=True)

    # S02: relative bias against IMERG, figure-2 definition
    ref = totals["IMERG"]
    biases = {k: np.where(ref / hours > MIN_REFERENCE_MM_H, 100.0 * (v - ref) / ref, np.nan)
              for k, v in totals.items() if k != "IMERG"}
    limit = colour_limit(biases.values())
    png, jpg, txt = supplementary_files(root, 2, BIAS_STEM, period)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5), constrained_layout=True)
    for ax, (label, bias) in zip(axes.ravel(), biases.items()):
        domain = 100.0 * (totals[label].mean() - ref.mean()) / ref.mean()
        pm = _map_axes(ax, lon, lat, bias, label, f"domain {domain:+.0f} %",
                       cmap="RdBu_r", vmin=-limit, vmax=limit)
    cbar = fig.colorbar(pm, ax=axes, shrink=0.8, extend="both")
    cbar.set_label("Relative bias vs IMERG (%)", fontsize=13)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    fig.suptitle(title_text, fontsize=18, fontweight="bold")
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    convert_to_jpg(png, jpg, dpi=dpi)
    png.unlink()
    rows = [f"Figure: {jpg}",
            "Bias: 100 * (model - IMERG) / IMERG of the IOP mean per cell; cells with IMERG mean "
            f"<= {MIN_REFERENCE_MM_H:g} mm h^-1 masked.",
            f"Colour limits: +/-{limit:g} % (p98 of |bias| over the four panels, rounded up to 10 %).",
            "dataset,domain_bias_percent,cell_bias_p02,cell_bias_median,cell_bias_p98"]
    for k, b in biases.items():
        v = b[np.isfinite(b)]
        domain = 100.0 * (totals[k].mean() - ref.mean()) / ref.mean()
        rows.append(f"{k},{domain:.2f},{np.percentile(v, 2):.2f},{np.median(v):.2f},{np.percentile(v, 98):.2f}")
    _map_txt(txt, f"{BIAS_STEM}_{period}", header, rows)
    print(f"[saved] {jpg}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--output-dir", type=Path, default=FIGURES)
    p.add_argument("--periods", nargs="+", default=list(PERIODS), choices=list(PERIODS))
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_iop_rain_pdf_roi", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for key in args.periods:
        make_pdf(args.work_dir / PERIODS[key], args.output_dir, key, args.dpi)
        if len(IOP_PERIODS[key]) == 1:
            make_supplementary_maps(args.work_dir / PERIODS[key], args.output_dir, key, args.dpi)
    write_figure_index(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
