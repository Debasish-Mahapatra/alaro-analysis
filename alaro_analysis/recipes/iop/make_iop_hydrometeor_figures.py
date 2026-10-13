#!/usr/bin/env python3
"""Hydrometeor mass-fraction figures for the GoAmazon IOPs, with no normalisation.

Same construction as the paper's hydrometeor figure, extended by one column for
G2M-XCU. Five species rows (graupel, snow, cloud ice, cloud water, rain) by four
columns:

    C1M absolute | G1M - C1M | G2M - G1M | G2M-XCU - G2M

Every cell is the plain number of grid-point-hours whose mass fraction fell in that
log10 bin at that model level, over the SIPAM radar domain on the native 4 km grid
(2960 cells). NO
NORMALISATION: heights are not rescaled to 100 %, so this is not a CFAD and
heights, species and experiments stay on one footing.

Absolute panels use the WhiteBlueGreenYellowRed log scale; difference panels a
diverging RdBu_r symmetric about zero at the 98th percentile of |difference|. Each
panel carries the per-level median mass fraction of its later model and the 0 degC
level of its column's model. C1M is the 2-ice control with no prognostic graupel,
so its graupel panel reads "no data" and that row shows the G1M absolute instead.

Reads the caches from ``build_iop_hydrometeor_histograms``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.figio import DEFAULT_DPI, convert_to_jpg
from alaro_analysis.common.iop_settings import (
    FIGURES,
    GOAMAZON_IOPS,
    IOP_PERIODS,
    figure_files,
    write_figure_index,
)
from alaro_analysis.common.vertical import centers_to_edges
from alaro_analysis.recipes.iop.build_iop_hydrometeor_histograms import DEFAULT_WORK_DIR

EXPS = ("control", "graupel", "2mom", "2mom-xcu")
FIGURE_NUMBER = 10
FIGURE_STEM = "hydrometeor_mass_fraction"
ROWS = (
    (r"$q_\mathrm{g}$", "GRAUPEL", "Graupel"),
    (r"$q_\mathrm{s}$", "SNOW", "Snow"),
    (r"$q_\mathrm{i}$", "SOLID_WATER", "Cloud ice"),
    (r"$q_\mathrm{c}$", "LIQUID_WATER", "Cloud water"),
    (r"$q_\mathrm{r}$", "RAIN", "Rain"),
)
# Column -> model whose 0 degC level is drawn on it.
COL_FREEZE_EXP = EXPS
MAX_HEIGHT_KM = 20.0
# Below this the values are numerical residue rather than condensate.
MIN_MASS_FRACTION = 1e-8
COUNT_VMIN, COUNT_VMAX = 1e1, 2e6
FREEZING_K = 273.15

# Fonts, as in the paper figure: everything bold, including mathtext.
TICK_FS, AXIS_FS, TITLE_FS = 17, 20, 23
CBAR_FS, CBAR_TICK_FS = 20, 15
SYM_FS, LEG_FS, LETTER_FS = 30, 15, 28

plt.rcParams.update(
    {
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "mathtext.default": "bf",
    }
)


def sci_tick(v, _pos=None) -> str:
    """Tick label as m x 10^n, e.g. 7.5x10^5. Small numbers stay plain."""
    if v == 0:
        return "0"
    a = abs(v)
    if 1e-3 <= a < 1e4:
        return f"{v:g}"
    exponent = int(np.floor(np.log10(a)))
    mantissa = v / 10.0**exponent
    m = f"{mantissa:.1f}".rstrip("0").rstrip(".")
    if m == "1":
        return rf"$10^{{{exponent}}}$"
    if m == "-1":
        return rf"$-10^{{{exponent}}}$"
    return rf"${m}{{\times}}10^{{{exponent}}}$"


def decade_ticks(limit: float) -> list[float]:
    """Decade ticks for the symmetric log colourbar.

    Its default locator also labels the minor ticks, which crowds 1000/300/100/0
    together around the linear core.
    """
    low = int(round(np.log10(DIFF_LINTHRESH)))
    high = int(round(np.log10(limit)))
    decades = [10.0**k for k in range(low, high + 1)]
    return [-d for d in reversed(decades)] + [0.0] + decades


def load_period(work: Path, iops: tuple[int, ...]) -> dict:
    """Histogram caches of every experiment, summed over the IOPs of one period."""
    data: dict = {"counts": {}, "files": {}, "height_km": None, "temperature": {}}
    for exp in EXPS:
        label = EXPERIMENT_LABELS[exp]
        counts: dict[str, np.ndarray] = {}
        files = 0
        temp_total, temp_weight = None, 0
        for iop in iops:
            with np.load(work / f"iop{iop}" / f"{label}_histograms.npz") as cache:
                for _, species, _ in ROWS:
                    key = f"counts_{species}"
                    if key not in cache:
                        continue
                    counts[species] = counts.get(species, 0) + cache[key].astype(np.float64)
                n = int(cache["n_files"])
                temperature = np.asarray(cache["temperature_k"], dtype=np.float64)
                temp_total = temperature * n if temp_total is None else temp_total + temperature * n
                temp_weight += n
                files += n
                data["height_km"] = np.asarray(cache["height_km"], dtype=np.float64)
                data["log_edges"] = np.asarray(cache["log_edges"], dtype=np.float64)
        data["counts"][exp] = counts
        data["files"][exp] = files
        data["temperature"][exp] = temp_total / max(temp_weight, 1)
    return data


def freezing_level_km(temperature: np.ndarray, height_km: np.ndarray) -> float:
    """Height of the 0 degC level, scanning upward from the surface."""
    finite = np.isfinite(temperature) & np.isfinite(height_km)
    z, t = height_km[finite], temperature[finite]
    order = np.argsort(z)
    z, t = z[order], t[order]
    for i in range(z.size - 1):
        a, b = t[i] - FREEZING_K, t[i + 1] - FREEZING_K
        if a == 0.0:
            return float(z[i])
        if a * b < 0.0:
            return float(z[i] + (z[i + 1] - z[i]) * (FREEZING_K - t[i]) / (t[i + 1] - t[i]))
    return float("nan")


def level_median(counts: np.ndarray, log_edges: np.ndarray) -> np.ndarray:
    """Per-level median mass fraction, read off the cumulative distribution.

    A median is a percentile, so it is the same whether or not the row is
    normalised; dropping the normalisation cannot move it.
    """
    centers = 10.0 ** (0.5 * (log_edges[:-1] + log_edges[1:]))
    cumulative = np.cumsum(counts, axis=1)
    median = np.full(counts.shape[0], np.nan)
    for level in range(counts.shape[0]):
        total = cumulative[level, -1]
        if total <= 0:
            continue
        index = int(np.searchsorted(cumulative[level] / total, 0.5))
        median[level] = centers[min(index, centers.size - 1)]
    return median


def prepare(data: dict) -> dict:
    """Crop to the plotted height range and drop the sub-threshold bins."""
    log_edges_full = data["log_edges"]
    dbin = log_edges_full[1] - log_edges_full[0]
    first = int(round((np.log10(MIN_MASS_FRACTION) - log_edges_full[0]) / dbin))
    log_edges = log_edges_full[first:]

    height = data["height_km"]
    order = np.argsort(height)
    keep = np.isfinite(height[order]) & (height[order] >= 0.0) & (height[order] <= MAX_HEIGHT_KM)
    y = height[order][keep]

    fields: dict[tuple[str, str], np.ndarray] = {}
    medians: dict[tuple[str, str], np.ndarray] = {}
    for exp in EXPS:
        for _, species, _ in ROWS:
            counts = data["counts"][exp].get(species)
            if counts is None:
                continue
            counts = counts[:, first:]
            fields[(exp, species)] = counts[order][keep]
            medians[(exp, species)] = level_median(counts, log_edges)[order][keep]
    freezing = {exp: freezing_level_km(data["temperature"][exp][order][keep], y) for exp in EXPS}
    return {
        "q_edges": 10.0**log_edges,
        "y": y,
        "y_edges": centers_to_edges(y),
        "fields": fields,
        "medians": medians,
        "freezing": freezing,
    }


def column_plan(species: str) -> list[tuple]:
    """(kind, payload, title, median experiment) per column for one species row.

    Graupel is not prognostic in the 2-ice control, so that row shows "no data"
    and then the G1M absolute, exactly as the paper figure does.
    """
    labels = {exp: EXPERIMENT_LABELS[exp] for exp in EXPS}
    later = [("graupel", "control"), ("2mom", "graupel"), ("2mom-xcu", "2mom")]
    if species == "GRAUPEL":
        plan = [
            ("nodata", None, labels["control"], None),
            ("abs", "graupel", labels["graupel"], "graupel"),
        ]
        pairs = later[1:]
    else:
        plan = [("abs", "control", labels["control"], "control")]
        pairs = later
    for new, old in pairs:
        plan.append(("diff", (new, old), f"{labels[new]} − {labels[old]}", new))
    return plan


# The three difference columns differ by an order of magnitude (p90 of |difference|:
# G1M-C1M 7.2e3, G2M-G1M 2.5e4, G2M-XCU-G2M 1.0e5), so one linear scale set by the
# largest column leaves the first one blank, and one set by the smallest saturates a
# quarter of the last. A symmetric log scale keeps a single global range and shows
# every column: values below DIFF_LINTHRESH stay in the near-white linear core.
DIFF_TAIL_PERCENTILE = 99.9
DIFF_LINTHRESH = 1.0e3


def difference_limit(prepared: dict) -> float:
    """Symmetric difference limit: a power of ten above the tail of the drawn differences."""
    pooled = []
    for _, species, _ in ROWS:
        for kind, payload, _, _ in column_plan(species):
            if kind != "diff":
                continue
            new, old = payload
            a, b = prepared["fields"].get((new, species)), prepared["fields"].get((old, species))
            if a is not None and b is not None:
                pooled.append(a - b)
    if not pooled:
        return 10.0 * DIFF_LINTHRESH
    flat = np.abs(np.concatenate([d.ravel() for d in pooled]))
    tail = float(np.percentile(flat[np.isfinite(flat)], DIFF_TAIL_PERCENTILE))
    return float(10.0 ** np.ceil(np.log10(max(tail, 10.0 * DIFF_LINTHRESH))))


def draw(prepared: dict, png: Path, dpi: int) -> float:
    import cmaps

    fields, medians, freezing = prepared["fields"], prepared["medians"], prepared["freezing"]
    q_edges, y, y_edges = prepared["q_edges"], prepared["y"], prepared["y_edges"]
    count_norm = mcolors.LogNorm(vmin=COUNT_VMIN, vmax=COUNT_VMAX)
    diff_scale = difference_limit(prepared)
    diff_norm = mcolors.SymLogNorm(
        linthresh=DIFF_LINTHRESH, vmin=-diff_scale, vmax=diff_scale, base=10
    )

    # A zero-height spacer row widens only the graupel/snow gap, where the column
    # titles change meaning; the other rows stay tight.
    # Paper panel geometry for four columns, kept under Pillow's 179 Mpx ceiling
    # at 450 dpi (29.5 x 27.5 in -> 13275 x 12375 px).
    fig = plt.figure(figsize=(29.5, 27.5))
    grid = fig.add_gridspec(
        len(ROWS) + 1,
        len(EXPS),
        height_ratios=[1.0, 0.001] + [1.0] * (len(ROWS) - 1),
        left=0.085,
        # Each panel's colorbar hangs its tick labels to the right of the slot, so
        # the last column needs margin or they are clipped at the figure edge.
        right=0.950,
        top=0.975,
        bottom=0.040,
        hspace=0.16,
        wspace=0.40,
    )
    grid_rows = [0] + list(range(2, len(ROWS) + 1))
    axes = np.empty((len(ROWS), len(EXPS)), dtype=object)
    for r in range(len(ROWS)):
        for c in range(len(EXPS)):
            axes[r, c] = fig.add_subplot(grid[grid_rows[r], c])

    letters = iter("abcdefghijklmnopqrst")
    for r, (_symbol, species, _name) in enumerate(ROWS):
        for c, (kind, payload, title, median_exp) in enumerate(column_plan(species)):
            ax = axes[r, c]
            ax.set_xscale("log")
            ax.set_xlim(q_edges[0], q_edges[-1])
            ax.set_ylim(0.0, MAX_HEIGHT_KM)
            ax.set_yticks(np.arange(0.0, MAX_HEIGHT_KM + 0.1, 5.0))
            ax.tick_params(axis="both", labelsize=TICK_FS)
            if r <= 1:
                ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")
            if r == len(ROWS) - 1:
                ax.set_xlabel(r"Mass fraction (kg kg$^{-1}$)", fontsize=AXIS_FS, fontweight="bold")
            if c == 0:
                ax.set_ylabel("Height (km)", fontsize=AXIS_FS, fontweight="bold")
            ax.text(
                0.975,
                0.03,
                f"({next(letters)})",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=LETTER_FS,
                fontweight="bold",
                zorder=12,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2.0},
            )

            handles = []
            if kind == "nodata":
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=18,
                    color="0.5",
                )
                continue
            if kind == "abs":
                # Zero is a real value but LogNorm cannot draw it; leave it blank.
                field = np.ma.masked_less_equal(fields[(payload, species)], 0.0)
                mesh = ax.pcolormesh(
                    q_edges,
                    y_edges,
                    field,
                    cmap=cmaps.WhiteBlueGreenYellowRed,
                    norm=count_norm,
                    shading="flat",
                )
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
                cbar = fig.colorbar(mesh, cax=cax)
                cbar.set_label("Count", fontsize=CBAR_FS, fontweight="bold")
                cbar.ax.tick_params(labelsize=CBAR_TICK_FS)
                cbar.ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
            else:
                new, old = payload
                a, b = fields[(new, species)], fields[(old, species)]
                # A zero count is information, so it enters the subtraction;
                # blank only where both sides are empty.
                diff = np.ma.masked_where((a <= 0) & (b <= 0), a - b)
                mesh = ax.pcolormesh(
                    q_edges, y_edges, diff, cmap="RdBu_r", norm=diff_norm, shading="flat"
                )
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
                cbar = fig.colorbar(mesh, cax=cax, ticks=decade_ticks(diff_scale))
                cbar.set_label(r"$\Delta$ Count", fontsize=CBAR_FS, fontweight="bold")
                cbar.ax.tick_params(labelsize=CBAR_TICK_FS)
                cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(sci_tick))

            if median_exp is not None and (median_exp, species) in medians:
                (line,) = ax.plot(
                    medians[(median_exp, species)], y, color="black", lw=2.4, zorder=6,
                    label="Median",
                )
                handles.append(line)
            level = freezing[COL_FREEZE_EXP[c]]
            if np.isfinite(level):
                handles.append(
                    ax.axhline(level, color="black", lw=2.4, ls="--", zorder=7, label="0 °C level")
                )
            if handles:
                ax.legend(
                    handles=handles,
                    loc="upper right",
                    framealpha=0.9,
                    prop={"size": LEG_FS, "weight": "bold"},
                )

    # Species symbol in the wide left margin, clear of the "Height (km)" label.
    for r, (symbol, _species, _name) in enumerate(ROWS):
        box = axes[r, 0].get_position()
        fig.text(
            0.030,
            0.5 * (box.y0 + box.y1),
            symbol,
            rotation=90,
            va="center",
            ha="center",
            fontsize=SYM_FS,
            fontweight="bold",
        )

    # Tight bbox so the last column's colorbar label is inside the page rather
    # than clipped at the right edge.
    fig.savefig(png, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return diff_scale


def write_txt(txt: Path, data: dict, prepared: dict, iops: tuple[int, ...], diff_scale: float):
    title = "Hydrometeor mass-fraction histograms, raw counts (no normalisation)"
    names = " and ".join(f"IOP{iop}" for iop in iops)
    combined = " combined" if len(iops) > 1 else ""
    lines = [
        title,
        "=" * len(title),
        f"Period: GoAmazon2014/5 {names}{combined} (Martin et al. 2016, ACP 16, 4785).",
    ]
    for iop in iops:
        start, end = GOAMAZON_IOPS[iop]
        lines.append(f"IOP{iop}: {start} to {end} inclusive, {(end - start).days + 1} days.")
    files = ", ".join(f"{EXPERIMENT_LABELS[e]} {data['files'][e]}" for e in EXPS)
    lines += [
        f"Hourly files per experiment: {files}.",
        "NO NORMALISATION IS APPLIED. Each cell is the plain number of grid-point-hours whose",
        "mass fraction fell in that bin at that model level, over the SIPAM radar domain on the",
        "native 4 km grid (2960 cells, ALARO-RUNS/mask/Radar_mask_4km_native.nc), all runs from",
        "raw FA. Heights are NOT rescaled to 100 %, so this is not a CFAD and heights, species",
        "and experiments are directly comparable.",
        "Columns: C1M absolute; G1M - C1M; G2M - G1M; G2M-XCU - G2M.",
        "Graupel row: C1M (no data), G1M absolute, G2M - G1M, G2M-XCU - G2M.",
        f"Absolute: WhiteBlueGreenYellowRed LogNorm {COUNT_VMIN:g}..{COUNT_VMAX:g} counts.",
        f"Difference: RdBu_r symmetric log, +/-{diff_scale:,.0f} counts, linear within "
        f"+/-{DIFF_LINTHRESH:,.0f}. The three difference columns span an order of magnitude,",
        "so a linear scale set by the largest would leave G1M - C1M blank; the log scale shows",
        "every column on one global range.",
        f"Only mass fractions >= {MIN_MASS_FRACTION:g} kg/kg are plotted; lower bins are",
        "numerical residue and are dropped. Empty bins are left blank; a zero on one side of a",
        "difference is kept.",
        "Black solid line: per-level median mass fraction of that panel's later model.",
        "Black dashed line: 0 degC level of that column's model.",
        "The full count matrices are in the npz caches under IOP-runs/analysis/hydrometeors/.",
        "",
        "0 degC height (km)",
        "------------------",
        "series,z_freeze_km",
    ]
    lines += [
        f"{EXPERIMENT_LABELS[e]},{prepared['freezing'][e]:.4g}"
        if np.isfinite(prepared["freezing"][e])
        else f"{EXPERIMENT_LABELS[e]},nan"
        for e in EXPS
    ]

    heading = "Counted grid-point-hours per species"
    lines += ["", heading, "-" * len(heading)]
    lines.append("species," + ",".join(EXPERIMENT_LABELS[e] for e in EXPS))
    for _, species, name in ROWS:
        row = []
        for exp in EXPS:
            field = prepared["fields"].get((exp, species))
            row.append("nan" if field is None else f"{float(field.sum()):.0f}")
        lines.append(f"{name}," + ",".join(row))

    for _, species, name in ROWS:
        heading = f"{name}: per-level median mass fraction (kg kg-1)"
        lines += ["", heading, "-" * len(heading)]
        lines.append("height_km," + ",".join(EXPERIMENT_LABELS[e] for e in EXPS))
        for i, height in enumerate(prepared["y"]):
            row = [f"{height:.4f}"]
            for exp in EXPS:
                median = prepared["medians"].get((exp, species))
                value = np.nan if median is None else median[i]
                row.append("nan" if not np.isfinite(value) else f"{value:.6g}")
            lines.append(",".join(row))
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure_outputs(root: Path, period: str) -> tuple[Path, Path, Path]:
    return figure_files(root, FIGURE_NUMBER, FIGURE_STEM, period)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--output-dir", type=Path, default=FIGURES)
    p.add_argument("--periods", nargs="+", default=list(IOP_PERIODS), choices=list(IOP_PERIODS))
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_iop_hydrometeor_figures", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for key in args.periods:
        iops = IOP_PERIODS[key]
        data = load_period(args.work_dir, iops)
        prepared = prepare(data)
        png, jpg, txt = figure_outputs(args.output_dir, key)
        diff_scale = draw(prepared, png, args.dpi)
        convert_to_jpg(png, jpg, dpi=args.dpi)
        png.unlink()
        write_txt(txt, data, prepared, iops, diff_scale)
        print(f"[saved] {jpg}\n[saved] {txt}", flush=True)
    write_figure_index(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
