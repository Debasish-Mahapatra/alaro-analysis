#!/usr/bin/env python3
"""How the choice of convective columns changes the G2M-XCU minus G2M mass flux.

Supplementary figure beside IOP figure 6 (``iop_settings.supplementary_files``), rows
IOP1 and IOP2. G2M-XCU has no 3MT updraft, so its convective columns are picked by rain
alone (>= 1 mm in the hour); in the runs with 3MT a column also needs a 3MT updraft. So
the runs are not sampled alike, and the flux per convective box of figure 6 mixes the
effect of switching 3MT off with the effect of picking other columns. Three lines:

    G2M               rain and a 3MT updraft, as in figure 6
    G2M, rain alone   the same run, its columns picked the way G2M-XCU's have to be
    G2M-XCU           rain alone

Columns:

    (a) share of the 20 km boxes that hold a convective column, by local hour
    (b) upward mass flux per convective box: mean over the box-hours with convection,
        all hours (the quantity of figure 6)
    (c) upward mass flux over the whole domain: the same flux summed over the convective
        boxes and divided by all box-hours, i.e. (b) times the share in (a). Weak boxes
        add little to it, so it depends much less on how the columns are picked.

Fluxes are the column split of ``build_iop_updraft_partition`` (``box_up``; run with
``--rain-only --experiments 2mom`` for the rain-alone G2M).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_COLORS
from alaro_analysis.common.figio import DEFAULT_DPI, convert_to_jpg, legend_below
from alaro_analysis.common.iop_settings import (
    PLOT_TOP_KM,
    FIGURES,
    GOAMAZON_IOPS,
    supplementary_files,
    write_figure_index,
)
from alaro_analysis.recipes.iop.build_iop_convection_diurnal import DEFAULT_WORK_DIR
from alaro_analysis.recipes.iop.make_iop_convection_figures import (
    AXIS_FS,
    HEADER_FS,
    LETTER_FS,
    MAX_HEIGHT_KM,
    TICK_FS,
)

SUPPLEMENT_NAME = "convective_flux_sampling"
STEM = "convective_flux_sampling"
PERIOD = "iop1_iop2"
# line -> (cache prefix, label, style)
LINES = {
    "g2m": ("G2M", "G2M (rain and 3MT updraft)",
            {"color": EXPERIMENT_COLORS["2mom"], "lw": 2.8, "ls": "-"}),
    "g2m_rain": ("G2M_rainonly", "G2M, rain alone",
                 {"color": "#17becf", "lw": 2.4, "ls": "--"}),
    "xcu": ("G2M-XCU", "G2M-XCU (rain alone)",
            {"color": EXPERIMENT_COLORS["2mom-xcu"], "lw": 2.8, "ls": "-"}),
}
REFERENCE_KM = 5.0
COLUMN_TITLES = ("Boxes with convection", "Per convective box", "Whole domain")


def load_line(work: Path, iop: int, prefix: str) -> dict:
    """Hourly share of convective boxes and the two all-hour flux profiles of one line."""
    with np.load(work / f"iop{iop}" / f"{prefix}_box_frequency.npz") as cache:
        share = np.asarray(cache["mean"], float)[0]
        boxes = np.asarray(cache["counts"], float)[0]  # all boxes per local hour
    with np.load(work / f"iop{iop}" / f"{prefix}_box_up.npz") as cache:
        mean = np.asarray(cache["mean"], float)
        counts = np.asarray(cache["counts"], float)  # convective boxes per level and hour
    total = np.where(np.isfinite(mean) & (counts > 0), mean * counts, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_box = total / counts.sum(axis=1)
    return {
        "share": share,
        "per_box": per_box,
        "domain": total / boxes.sum(),
        "box_hours": int(counts[0].sum()),
        "all_box_hours": int(boxes.sum()),
    }


def load(work: Path) -> dict:
    with np.load(work / "iop1" / "G2M_height.npz") as cache:
        height = np.asarray(cache["height_m"], float) / 1000.0
    rows = {iop: {key: load_line(work, iop, prefix) for key, (prefix, _, _) in LINES.items()}
            for iop in GOAMAZON_IOPS}
    return {"height_km": height, "rows": rows}


def at_height(height: np.ndarray, profile: np.ndarray, z: float = REFERENCE_KM) -> float:
    return float(np.interp(z, height, profile))


def percent(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def signed(value: float) -> str:
    """Whole per cent with its sign, and a plain 0 rather than -0."""
    return "0" if round(value) == 0 else f"{value:+.0f}"


def draw(data: dict, png: Path, dpi: int) -> None:
    height = data["height_km"]
    keep = height <= MAX_HEIGHT_KM
    fig, axes = plt.subplots(2, 3, figsize=(20.0, 13.0), sharex="col")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.10, wspace=0.28, hspace=0.18)
    letters = iter("abcdef")
    hours = np.arange(24)
    for r, (iop, lines) in enumerate(data["rows"].items()):
        ax_share, ax_box, ax_dom = axes[r]
        for key, (_, _, style) in LINES.items():
            line = lines[key]
            ax_share.plot(hours, 100.0 * line["share"], **style)
            ax_box.plot(line["per_box"][keep], height[keep], **style)
            ax_dom.plot(line["domain"][keep], height[keep], **style)
        for ax, field in ((ax_box, "per_box"), (ax_dom, "domain")):
            g2m, g2m_rain, xcu = (at_height(height, lines[k][field]) for k in LINES)
            ax.text(  # upper right: the profiles are narrow aloft, so it stays clear of them
                0.97, 0.97,
                f"At {REFERENCE_KM:g} km, G2M-XCU vs\n"
                f"G2M: {signed(percent(xcu, g2m))} %\n"
                f"G2M, rain alone: {signed(percent(xcu, g2m_rain))} %",
                transform=ax.transAxes, ha="right", va="top", fontsize=TICK_FS - 3,
                bbox={"facecolor": "white", "edgecolor": "#b0b0b0", "alpha": 0.9, "pad": 4.0},
            )
            ax.set_ylim(0.0, PLOT_TOP_KM)
            ax.set_xlim(left=0.0)
            ax.axhline(REFERENCE_KM, color="#808080", lw=0.8, ls=":", zorder=0)
        ax_share.set_xlim(0, 23)
        ax_share.set_xticks(np.arange(0, 24, 6))
        ax_share.set_ylim(bottom=0.0)
        ax_share.set_ylabel("Share of 20 km boxes (%)", fontsize=AXIS_FS)
        ax_box.set_ylabel("Height (km)", fontsize=AXIS_FS)
        ax_dom.set_ylabel("Height (km)", fontsize=AXIS_FS)
        ax_share.annotate(
            f"IOP{iop}", xy=(-0.30, 0.5), xycoords="axes fraction", rotation=90,
            ha="center", va="center", fontsize=HEADER_FS + 2, fontweight="bold",
        )
        for c, ax in enumerate(axes[r]):
            ax.tick_params(axis="both", labelsize=TICK_FS)
            ax.grid(color="#e0e0e0", lw=0.8)
            ax.text(0.0, 1.02, f"({next(letters)})", transform=ax.transAxes, ha="left",
                    va="bottom", fontsize=LETTER_FS, fontweight="bold")
            if r == 0:
                ax.set_title(COLUMN_TITLES[c], fontsize=HEADER_FS, fontweight="bold", pad=12)
    axes[1, 0].set_xlabel("Hour (Amazon local time, UTC-4)", fontsize=AXIS_FS)
    axes[1, 1].set_xlabel("Upward mass flux (kg m$^{-2}$ s$^{-1}$)", fontsize=AXIS_FS)
    axes[1, 2].set_xlabel("Upward mass flux (kg m$^{-2}$ s$^{-1}$)", fontsize=AXIS_FS)
    handles = [Line2D([0], [0], label=label, **style) for _, label, style in LINES.values()]
    legend_below(fig, axes, handles, ncol=3)
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_txt(txt: Path, data: dict, jpg: Path) -> None:
    height = data["height_km"]
    title = f"{STEM}_{PERIOD}"
    lines = [title, "=" * len(title), f"Figure: {jpg}"]
    for iop, (start, end) in GOAMAZON_IOPS.items():
        lines.append(f"IOP{iop}: {start} to {end} inclusive, every hour of the campaign days.")
    lines += [
        "Upward mass flux (kg m-2 s-1): each column split into 3MT updraft, downdraft and the rest on the",
        "instantaneous hourly snapshots (UD_OMEGA relative, DD_OMEGA absolute, VITESSE_VERT the area mean);",
        "rising part of each piece x its area; convective columns only, other columns count as 0; 20 km boxes",
        "(5 x 5 cells, 116 boxes at least half inside the 4 km radar domain) divided by their full area.",
        "Convective column: rain >= 1 mm in the hour ending at the snapshot; G2M (as in figure 6) also needs a",
        "3MT updraft (mesh fraction > 1e-3 at some level); 'G2M, rain alone' and G2M-XCU (no 3MT) use rain alone.",
        "per_box = mean over the box-hours with a convective column (figure 6); domain = the same sum divided by",
        "all box-hours = per_box x share. Hour is Amazon local time (UTC-4).",
        "",
        "iop,line,convective_box_hours,all_box_hours,share",
    ]
    for iop, rows in data["rows"].items():
        for key, (_, label, _) in LINES.items():
            line = rows[key]
            lines.append(f"IOP{iop},{label},{line['box_hours']},{line['all_box_hours']},"
                         f"{line['box_hours'] / line['all_box_hours']:.4f}")
    lines += ["", f"Differences at {REFERENCE_KM:g} km (%)", "iop,quantity,XCU_vs_G2M,XCU_vs_G2M_rain_alone,"
              "G2M_rain_alone_vs_G2M"]
    for iop, rows in data["rows"].items():
        for field in ("per_box", "domain"):
            g2m, g2m_rain, xcu = (at_height(height, rows[k][field]) for k in LINES)
            lines.append(f"IOP{iop},{field},{percent(xcu, g2m):.1f},{percent(xcu, g2m_rain):.1f},"
                         f"{percent(g2m_rain, g2m):.1f}")
    labels = [label for _, label, _ in LINES.values()]
    for iop, rows in data["rows"].items():
        lines += ["", f"IOP{iop}: share of boxes with convection (%) by local hour",
                  ",".join(["hour", *labels])]
        for h in range(24):
            lines.append(",".join([str(h), *[f"{100.0 * rows[k]['share'][h]:.3f}" for k in LINES]]))
        for field in ("per_box", "domain"):
            lines += ["", f"IOP{iop}: upward mass flux, {field} (kg m-2 s-1)",
                      ",".join(["height_km", *labels])]
            for i in np.flatnonzero(height <= MAX_HEIGHT_KM):
                lines.append(",".join([f"{height[i]:.3f}", *[f"{rows[k][field][i]:.6g}" for k in LINES]]))
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--output-dir", type=Path, default=FIGURES)
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_iop_convective_flux_sampling", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = load(args.work_dir)
    png, jpg, txt = supplementary_files(args.output_dir, SUPPLEMENT_NAME, STEM, PERIOD)
    draw(data, png, args.dpi)
    convert_to_jpg(png, jpg, dpi=args.dpi)
    png.unlink()
    write_txt(txt, data, jpg)
    print(f"[saved] {jpg}\n[saved] {txt}", flush=True)
    write_figure_index(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
