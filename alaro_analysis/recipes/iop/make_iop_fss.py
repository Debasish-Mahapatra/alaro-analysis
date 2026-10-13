#!/usr/bin/env python3
"""Fractions skill score of hourly rain against the SIPAM radar for the GoAmazon IOPs.

Reads the common-valid hourly rain written by ``build_iop_rainfall_common_valid``
(0.1 deg grid, radar valid-time mask already applied) and scores each model
against the radar with the fractions skill score over a range of rain thresholds
and neighbourhood widths. Two heatmaps per experiment:

    (a) absolute thresholds, 0.1 to 20 mm/h
    (b) percentile thresholds, each field thresholded at its own percentile, so the
        rain-amount bias drops out and only the placement is scored

The score is taken over the radar footprint only (``analysis.fractions_skill``):
fractions use the valid cells of each neighbourhood. When nwpeval is installed, its
``fss`` is run on the same footprint as an independent check and written to the data
file next to ours.

On this grid the model rain carries more small-scale variance than the radar, so
some of the loss of skill at small neighbourhoods is resolution rather than
placement.
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
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle

from alaro_analysis.analysis.fractions_skill import (
    fss_components,
    fss_from_components,
    useful_skill,
)
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

# Period -> common-valid subdirectory
PERIODS = {
    "combined": COMMON_SUBDIR,
    "iop1": f"{COMMON_SUBDIR}-iop1",
    "iop2": f"{COMMON_SUBDIR}-iop2",
}
EXPERIMENTS = ("C1M", "G1M", "G2M", "G2M-XCU")
FIGURE_NUMBER = 12
THRESHOLDS_MM_H = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
PERCENTILES = (90.0, 95.0, 97.5, 99.0, 99.5)
# Odd widths only, so each neighbourhood is centred on its cell.
SIZES = (1, 3, 5, 7, 9, 11, 15, 19, 23)
# 0.1 deg at about 3 S.
CELL_KM = 11.1
TEXT_DARK, TEXT_LIGHT = "#111111", "#ffffff"


def load_period(work: Path, subdir: str, label: str):
    """Radar and model hourly rain on the common-valid hours, and the radar footprint."""
    with xr.open_dataset(work / subdir / "Radar_common_valid.nc") as ds:
        radar = np.asarray(ds["rainfall"].values, dtype=float)
        times = np.asarray(ds["time"].values)
    with xr.open_dataset(work / subdir / f"{label}_common_valid.nc") as ds:
        model = np.asarray(ds["rainfall"].values, dtype=float)
        if not np.array_equal(np.asarray(ds["time"].values), times):
            raise ValueError(f"{label} and radar hours differ in {subdir}")
    footprint = np.isfinite(radar).all(axis=0) & np.isfinite(model).all(axis=0)
    return radar, model, footprint, times


def score_table(radar, model, footprint, radar_thresholds, model_thresholds):
    """FSS per (threshold, width) and the base rates of both fields."""
    table = np.full((len(radar_thresholds), len(SIZES)), np.nan)
    base_radar = np.zeros(len(radar_thresholds))
    base_model = np.zeros(len(radar_thresholds))
    for i, (t_radar, t_model) in enumerate(zip(radar_thresholds, model_thresholds)):
        wet_radar = ((radar >= t_radar) & footprint).astype(float)
        wet_model = ((model >= t_model) & footprint).astype(float)
        base_radar[i] = wet_radar[:, footprint].mean()
        base_model[i] = wet_model[:, footprint].mean()
        for j, size in enumerate(SIZES):
            table[i, j] = fss_from_components(
                *fss_components(wet_radar, wet_model, footprint, size)
            )
    return table, base_radar, base_model


def nwpeval_table(radar, model, footprint) -> np.ndarray | None:
    """nwpeval's FSS on the same footprint, as an independent check."""
    try:
        from nwpeval.metrics.fss import fss as nwpeval_fss
    except ImportError:
        return None
    inside = xr.DataArray(footprint, dims=("lat", "lon"))
    obs = xr.DataArray(radar, dims=("time", "lat", "lon")).where(inside)
    mod = xr.DataArray(model, dims=("time", "lat", "lon")).where(inside)
    table = np.full((len(THRESHOLDS_MM_H), len(SIZES)), np.nan)
    for i, threshold in enumerate(THRESHOLDS_MM_H):
        for j, size in enumerate(SIZES):
            table[i, j] = float(nwpeval_fss(obs, mod, threshold, size))
    return table


def draw_panel(ax, table, useful, row_labels, title, letter, y_label):
    norm = Normalize(0.0, 1.0)
    image = ax.imshow(table, cmap="Blues", norm=norm, origin="lower", aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table[i, j]
            if not np.isfinite(value):
                continue
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color=TEXT_LIGHT if value > 0.62 else TEXT_DARK,
            )
            if value >= useful[i]:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=TEXT_DARK, linewidth=1.8
                    )
                )
    ax.set_xticks(
        range(len(SIZES)),
        labels=[f"{n}\n{n * CELL_KM:.0f} km" for n in SIZES],
    )
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.set_xlabel("Neighbourhood width (grid cells)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=10, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        -0.02, 1.02, letter, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=14, fontweight="bold",
    )
    return image


def draw(label: str, absolute: dict, percentile: dict, png: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.6), constrained_layout=True)
    image = draw_panel(
        axes[0],
        absolute["table"],
        absolute["useful"],
        [f"{t:g}" for t in THRESHOLDS_MM_H],
        f"{label} vs radar, fixed thresholds",
        "(a)",
        r"Rain threshold (mm h$^{-1}$)",
    )
    draw_panel(
        axes[1],
        percentile["table"],
        percentile["useful"],
        [f"{p:g}th" for p in PERCENTILES],
        f"{label} vs radar, percentile thresholds",
        "(b)",
        "Percentile of each field",
    )
    cbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("Fractions skill score", fontsize=12)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    fig.legend(
        handles=[Patch(facecolor="none", edgecolor=TEXT_DARK, linewidth=1.8)],
        labels=[r"Useful skill: FSS $\geq$ 0.5 + $f_0$/2"],
        loc="outside lower center",
        frameon=False,
        fontsize=11,
    )
    fig.savefig(png, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def write_txt(txt: Path, label, iops, n_hours, n_cells, absolute, percentile, check):
    names = " and ".join(f"IOP{iop}" for iop in iops)
    combined = " combined" if len(iops) > 1 else ""
    title = f"Fractions skill score, {label} against the SIPAM radar"
    lines = [
        title,
        "=" * len(title),
        f"Period: GoAmazon2014/5 {names}{combined} (Martin et al. 2016, ACP 16, 4785).",
    ]
    for iop in iops:
        start, end = GOAMAZON_IOPS[iop]
        lines.append(f"IOP{iop}: {start} to {end} inclusive.")
    lines += [
        f"Hourly rain on the 0.1 deg grid, {n_hours} common-valid hours (radar valid-time mask),",
        f"scored over the {n_cells} cells of the radar footprint.",
        "FSS = 1 - sum (O - M)^2 / sum (O^2 + M^2) over every footprint cell and hour (Roberts",
        "and Lean 2008). Neighbourhood fractions use the footprint cells of each window only.",
        "Useful skill: FSS >= 0.5 + f0/2, f0 the radar base rate (outlined in the figure).",
        "Percentile thresholds: each field at its own percentile over footprint cell-hours, so",
        "the rain-amount bias drops out and the score measures placement.",
        f"Neighbourhood width in grid cells; one cell is about {CELL_KM:g} km.",
        "Caveat: the model rain carries more small-scale variance than the radar on this grid,",
        "so part of the low skill at small widths is resolution rather than placement.",
        "",
    ]

    def block(heading, rows, table, extra):
        lines.extend([heading, "-" * len(heading)])
        lines.append(",".join(["threshold", *extra[0], *[f"n{n}" for n in SIZES]]))
        for i, row in enumerate(rows):
            values = [f"{v:.4f}" if np.isfinite(v) else "nan" for v in table[i]]
            lines.append(",".join([row, *extra[1][i], *values]))
        lines.append("")

    block(
        "(a) Fixed thresholds (mm/h)",
        [f"{t:g}" for t in THRESHOLDS_MM_H],
        absolute["table"],
        (
            ["radar_base_rate", "model_base_rate", "useful_skill"],
            [
                [f"{a:.5f}", f"{b:.5f}", f"{u:.4f}"]
                for a, b, u in zip(
                    absolute["base_radar"], absolute["base_model"], absolute["useful"]
                )
            ],
        ),
    )
    block(
        "(b) Percentile thresholds",
        [f"p{p:g}" for p in PERCENTILES],
        percentile["table"],
        (
            ["radar_mm_h", "model_mm_h", "useful_skill"],
            [
                [f"{a:.4f}", f"{b:.4f}", f"{u:.4f}"]
                for a, b, u in zip(
                    percentile["radar_thresholds"], percentile["model_thresholds"],
                    percentile["useful"],
                )
            ],
        ),
    )
    if check is not None:
        heading = "nwpeval.metrics.fss on the same footprint (independent check)"
        lines.extend([heading, "-" * len(heading)])
        lines.append(",".join(["threshold", *[f"n{n}" for n in SIZES]]))
        for i, threshold in enumerate(THRESHOLDS_MM_H):
            values = [f"{v:.4f}" if np.isfinite(v) else "nan" for v in check[i]]
            lines.append(",".join([f"{threshold:g}", *values]))
        diff = check - absolute["table"]
        lines.append(
            f"max |nwpeval - ours| = {np.nanmax(np.abs(diff)):.2e}"
        )
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--output-dir", type=Path, default=FIGURES)
    p.add_argument("--periods", nargs="+", default=["combined"], choices=list(PERIODS))
    p.add_argument("--experiments", nargs="+", default=["C1M"], choices=list(EXPERIMENTS))
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_iop_fss", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for key in args.periods:
        iops = IOP_PERIODS[key]
        for label in args.experiments:
            radar, model, footprint, times = load_period(args.work_dir, PERIODS[key], label)

            table, base_radar, base_model = score_table(
                radar, model, footprint, THRESHOLDS_MM_H, THRESHOLDS_MM_H
            )
            absolute = {
                "table": table,
                "base_radar": base_radar,
                "base_model": base_model,
                "useful": [useful_skill(f) for f in base_radar],
            }

            radar_thr = [float(np.percentile(radar[:, footprint], p)) for p in PERCENTILES]
            model_thr = [float(np.percentile(model[:, footprint], p)) for p in PERCENTILES]
            table, base_radar, _ = score_table(radar, model, footprint, radar_thr, model_thr)
            percentile = {
                "table": table,
                "radar_thresholds": radar_thr,
                "model_thresholds": model_thr,
                "useful": [useful_skill(f) for f in base_radar],
            }

            png, jpg, txt = figure_files(args.output_dir, FIGURE_NUMBER, f"fss_{label}", key)
            draw(label, absolute, percentile, png, args.dpi)
            convert_to_jpg(png, jpg, dpi=args.dpi)
            png.unlink()
            write_txt(
                txt, label, iops, times.size, int(footprint.sum()), absolute, percentile,
                nwpeval_table(radar, model, footprint),
            )
            print(f"[saved] {jpg}\n[saved] {txt}", flush=True)
    write_figure_index(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
