#!/usr/bin/env python3
"""Paper figure 5, rebuilt on the GoAmazon IOPs: updraft extent, upward mass flux, cloud cover.

Same figure as ``make_paper_updraft_cloud_relative_panel`` (diurnal height sections,
C1M absolute beside the relative differences), with three changes:

1. The mass flux row is the total upward mass flux of the column split into 3MT
   updraft, downdraft and environment (the supervisor's definition, see
   ``build_iop_updraft_cloud_partition``), not the 3MT updraft piece computed as if
   UD_OMEGA were an absolute velocity.
2. A fourth column, G2M-XCU minus G2M, isolates switching the 3MT convection scheme
   off. With no scheme there is no parameterized updraft, so the extent panel of that
   column stays empty.
3. The period is the GoAmazon IOPs, not the continuous 2-year run: of the four runs
   only C1M still has 2 years of raw FA on this machine, and the 2-year masked-netcdf
   archive the published version read is gone.

Written beside the published figure 5 under a different name, so the 2-year version
stays as it is. Rows are quantities, columns are C1M and the three relative
differences, each against the previous rung of the ladder.
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

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.figio import DEFAULT_DPI, LEGEND_FONTSIZE
from alaro_analysis.common.iop_settings import GOAMAZON_IOPS, IOP_PERIODS
from alaro_analysis.common.paths import data_path
from alaro_analysis.common.models import VerticalAxis
from alaro_analysis.common.vertical import centers_to_edges, compute_freezing_line_km
from alaro_analysis.data.convection_sections import PARTITION_QUANTITIES
from alaro_analysis.plotting.scales import robust_anomaly_scale
from alaro_analysis.recipes.iop.build_iop_convection_diurnal import DEFAULT_WORK_DIR
from alaro_analysis.recipes.iop.make_iop_convection_figures import (
    AXIS_FS,
    CBAR_LABEL_FS,
    CBAR_TICK_FS,
    HEADER_FS,
    LEGEND_FS,
    LETTER_FS,
    MAX_HEIGHT_KM,
    TICK_FS,
    merge_sections,
    relative_bias,
)

OUTPUT_DIR = data_path("microphysics-paper/5. updraft and cloud cover")
# Beside, not over, the published 2-year figure of the same folder.
FIGURE_STEM = "5. updraft and cloud cover_column-split_iops"
PERIOD_NAMES = {"combined": "iop1+iop2", "iop1": "iop1", "iop2": "iop2"}
EXPS = ("control", "graupel", "2mom", "2mom-xcu")
# (quantity, row label, unit, unit interval, experiments that have it)
ROWS = (
    ("ud_extent", "Updraft extent", "Fraction", True, ("control", "graupel", "2mom")),
    ("flux_up", "Upward mass flux", "kg m$^{-2}$ s$^{-1}$", False, EXPS),
    ("cloud", "Cloud cover", "Fraction", True, EXPS),
)
# Column -> (numerator experiment, reference experiment); column 0 is absolute C1M.
DIFFS = (("graupel", "control"), ("2mom", "graupel"), ("2mom-xcu", "2mom"))
PLAIN_UNITS = {"Fraction": "fraction", "kg m$^{-2}$ s$^{-1}$": "kg m-2 s-1"}
COLUMN_TITLES = (
    "C1M (absolute)",
    "(G1M − C1M) / C1M",
    "(G2M − G1M) / G1M",
    "(G2M-XCU − G2M) / G2M",
)


def load_period(work: Path, iops: tuple[int, ...]) -> dict:
    """Cached partition sections of every experiment for one period."""
    data: dict = {"sections": {}, "files": {}}
    for exp in EXPS:
        label = EXPERIMENT_LABELS[exp]
        sections, files = {}, 0
        for quantity in PARTITION_QUANTITIES:
            parts = []
            for iop in iops:
                with np.load(work / f"iop{iop}" / f"{label}_partition_{quantity}.npz") as cache:
                    parts.append((cache["mean"], cache["counts"]))
                    files = max(files, int(cache["n_files"]))
            sections[quantity] = merge_sections(parts)
        with np.load(work / f"iop{iops[0]}" / f"{label}_partition_height.npz") as cache:
            height_km = np.asarray(cache["height_m"], dtype=np.float64) / 1000.0
        data["sections"][exp] = sections
        data["files"][exp] = files * len(iops)
        data.setdefault("height_km", height_km)

    keep = np.isfinite(data["height_km"]) & (data["height_km"] <= MAX_HEIGHT_KM)
    data["height_km"] = data["height_km"][keep]
    for exp in EXPS:
        data["sections"][exp] = {k: v[keep, :] for k, v in data["sections"][exp].items()}
    return data


def period_header(data: dict, iops: tuple[int, ...]) -> list[str]:
    names = " and ".join(f"IOP{iop}" for iop in iops)
    combined = " combined" if len(iops) > 1 else ""
    lines = [f"Period: GoAmazon2014/5 {names}{combined} (Martin et al. 2016, ACP 16, 4785)."]
    for iop in iops:
        start, end = GOAMAZON_IOPS[iop]
        lines.append(f"IOP{iop}: {start} to {end} inclusive, {(end - start).days + 1} days.")
    files = ", ".join(f"{EXPERIMENT_LABELS[e]} {data['files'][e]}" for e in EXPS)
    lines += [
        f"Hourly files reduced: {files}. Every hour of the campaign days is used.",
        "Domain mean over the SIPAM radar domain on the native 4 km grid (2960 cells, "
        "ALARO-RUNS/mask/Radar_mask_4km_native.nc); every cell counts, with or without a "
        "draft. Hour is Amazon local time (UTC-4).",
        "All runs from raw FA: C1M from the continuous control run, G1M, G2M and G2M-XCU "
        "from the IOP warm-restart runs.",
        "Upward mass flux: the column is split into the 3MT updraft, the downdraft and the "
        "remaining environment, and the rising part of each piece is weighted by its area "
        "(data.convection_sections.column_fluxes). UD_OMEGA is the updraft velocity relative "
        "to its environment (ACCVUD; Gerard et al. 2009, eq. 4), DD_OMEGA is absolute "
        "(ACMODO), and the resolved VITESSE_VERT is the area mean of all three, so this is "
        "not the updraft piece plus the resolved ascent.",
        "G2M-XCU runs without 3MT: its column is one piece, so its upward flux is the rising "
        "resolved motion, and it has no parameterized updraft extent (panel left empty).",
        "Draft fields cleaned as the 3MT physics does on entry (advection overshoots): mesh "
        "fraction < 0 -> 0, UD_OMEGA > 0 -> 0, DD_OMEGA < 0 -> 0.",
        "Relative differences are 100 * (model - reference) / reference, masked where the "
        "reference is below 5 % of its 98th percentile. Grey cells were never sampled or are "
        "masked that way.",
    ]
    return lines


def draw(data: dict, png: Path, dpi: int) -> list[dict]:
    """Rows are quantities, columns are C1M and the three relative differences.

    Column headers run along the top and the quantity labels down the left, so no
    panel repeats them. Each row carries two horizontal colour bars: the absolute
    scale under column 1 and the shared relative scale under columns 2 to 4.
    """
    import cmaps

    height = data["height_km"]
    y_edges = centers_to_edges(height)
    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    axis = VerticalAxis(values=height, label="Height (km)", is_height_km=True)
    freezing = {
        exp: compute_freezing_line_km(axis, data["sections"][exp]["temp"]) for exp in EXPS
    }

    fig, axes = plt.subplots(3, 4, figsize=(24.0, 16.5), constrained_layout=True)
    letters = iter("abcdefghijkl")
    records: list[dict] = []
    freezing_handle = None

    for r, (quantity, title, unit, unit_interval, available) in enumerate(ROWS):
        sections = {e: data["sections"][e][quantity] for e in EXPS}
        abs_ctrl = np.ma.masked_invalid(sections["control"])
        rels = []
        for numerator, reference in DIFFS:
            if numerator not in available or reference not in available:
                rels.append(None)
                continue
            rels.append(
                np.ma.masked_invalid(
                    relative_bias(sections[numerator] - sections[reference], sections[reference])
                )
            )

        valid = abs_ctrl.compressed()
        vmin = max(0.0, float(np.percentile(valid, 2.0))) if valid.size else 0.0
        vmax = float(np.percentile(valid, 98.0)) if valid.size else 1.0
        if unit_interval:
            vmax = min(1.0, vmax)
        if vmax <= vmin:
            vmax = vmin + 1e-9
        drawn = [rel.filled(np.nan) for rel in rels if rel is not None]
        anom = robust_anomaly_scale(*drawn) if drawn else 1.0
        if not np.isfinite(anom) or anom <= 0:
            anom = 1.0
        norms = (
            mcolors.Normalize(vmin=vmin, vmax=vmax),
            mcolors.TwoSlopeNorm(vmin=-anom, vcenter=0.0, vmax=anom),
        )
        record = {
            "label": title, "unit": unit, "abs": np.ma.filled(abs_ctrl, np.nan),
            "vmin": vmin, "vmax": vmax, "anom": anom,
        }
        for ci, rel in enumerate(rels):
            record[f"rel{ci + 1}"] = None if rel is None else np.ma.filled(rel, np.nan)
        records.append(record)

        fields = [abs_ctrl, *rels]
        cmaps_used = [cmaps.WhiteBlueGreenYellowRed, "RdBu_r", "RdBu_r", "RdBu_r"]
        pcms: list = []
        for ci, ax in enumerate(axes[r]):
            ax.set_facecolor("#d3d3d3")
            ax.set_xlim(-0.5, 23.5)
            ax.set_ylim(0.0, MAX_HEIGHT_KM)
            ax.set_xticks(np.arange(0, 24, 6))
            ax.tick_params(axis="both", labelsize=TICK_FS)
            if r == 0:
                ax.set_title(COLUMN_TITLES[ci], fontsize=HEADER_FS, fontweight="bold", pad=14)
            if ci == 0:
                ax.set_ylabel("Height (km)", fontsize=AXIS_FS)
                ax.annotate(
                    f"{title}\n({unit})", xy=(-0.30, 0.5), xycoords="axes fraction",
                    rotation=90, ha="center", va="center",
                    fontsize=HEADER_FS + 1, fontweight="bold",
                )
            else:
                ax.set_yticklabels([])
            if r == len(ROWS) - 1:
                ax.set_xlabel("Hour (Amazon local time, UTC-4)", fontsize=AXIS_FS)
            ax.text(
                0.025, 0.97, f"({next(letters)})", transform=ax.transAxes, ha="left", va="top",
                fontsize=LETTER_FS, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2.0},
            )
            if fields[ci] is None:  # no parameterized updraft without 3MT
                ax.set_facecolor("white")
                for side in ax.spines.values():
                    side.set_color("#b0b0b0")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(
                    0.5, 0.5,
                    f"{EXPERIMENT_LABELS['2mom-xcu']} runs without 3MT:\nno parameterized updraft",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=HEADER_FS, color="#505050",
                )
                pcms.append(None)
                continue
            pcms.append(
                ax.pcolormesh(
                    hour_edges, y_edges, fields[ci], cmap=cmaps_used[ci],
                    norm=norms[0] if ci == 0 else norms[1], shading="auto",
                )
            )
            line = freezing[EXPS[ci]]
            if line is not None and np.isfinite(line).any():
                handle, = ax.plot(
                    np.arange(24), line, color="black", lw=1.8, ls="--", zorder=10,
                    label="Freezing level",
                )
                freezing_handle = freezing_handle or handle

        cbar = fig.colorbar(
            pcms[0], ax=axes[r, 0], orientation="horizontal", fraction=0.075, pad=0.03
        )
        cbar.set_label(unit, fontsize=CBAR_LABEL_FS)
        cbar.ax.tick_params(labelsize=CBAR_TICK_FS)
        cbar.ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        cbar.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

        shown = [pcm for pcm in pcms[1:] if pcm is not None]
        cbar_rel = fig.colorbar(
            shown[0], ax=list(axes[r, 1:]), orientation="horizontal", fraction=0.075, pad=0.03
        )
        cbar_rel.set_label("Relative difference (%)", fontsize=CBAR_LABEL_FS)
        cbar_rel.ax.tick_params(labelsize=CBAR_TICK_FS)
        cbar_rel.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    if freezing_handle is not None:
        fig.legend(
            handles=[freezing_handle], loc="outside lower center", ncol=1,
            fontsize=LEGEND_FONTSIZE, frameon=False,
        )
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return records


def write_txt(txt: Path, data: dict, records: list[dict], header: list[str]) -> None:
    height = data["height_km"]
    lines = list(header)
    lines.append("")
    lines.append(
        "Panels: rows are updraft extent, upward mass flux and cloud cover; columns are "
        "C1M absolute and the relative differences "
        + ", ".join(f"{EXPERIMENT_LABELS[a]}-{EXPERIMENT_LABELS[b]}" for a, b in DIFFS)
        + "."
    )
    for record in records:
        unit = PLAIN_UNITS.get(record["unit"], record["unit"])
        panels = [(f"{EXPERIMENT_LABELS['control']} absolute ({unit})", record["abs"])]
        for ci, (a, b) in enumerate(DIFFS):
            panel = record.get(f"rel{ci + 1}")
            name = f"({EXPERIMENT_LABELS[a]} - {EXPERIMENT_LABELS[b]}) / {EXPERIMENT_LABELS[b]} (%)"
            if panel is None:
                lines += ["", f"{record['label']}: {name}", "  not applicable (no 3MT updraft)"]
                continue
            panels.append((name, panel))
        for name, values in panels:
            lines += ["", f"{record['label']}: {name}",
                      ",".join(["height_km", *[f"h{h:02d}" for h in range(24)]])]
            for z, row in zip(height, values):
                cells = ["" if not np.isfinite(v) else f"{v:.6g}" for v in row]
                lines.append(",".join([f"{z:.3f}", *cells]))
    txt.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--periods", nargs="+", default=list(IOP_PERIODS), choices=list(IOP_PERIODS))
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_paper_updraft_cloud_partition_panel", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for key in args.periods:
        iops = IOP_PERIODS[key]
        data = load_period(args.work_dir, iops)
        header = period_header(data, iops)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        name = f"{FIGURE_STEM}_{PERIOD_NAMES[key]}"
        png = args.output_dir / f"{name}_450dpi.png"
        txt = args.output_dir / f"{name}_data.txt"
        records = draw(data, png, args.dpi)
        write_txt(txt, data, records, header)
        print(f"[saved] {png}\n[saved] {txt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
