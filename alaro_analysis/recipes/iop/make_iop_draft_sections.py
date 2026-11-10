#!/usr/bin/env python3
"""IOP figures 6 and 7: diurnal height sections of the updrafts and of the downdrafts.

Two figures per period (combined IOPs, IOP1, IOP2), C1M absolute beside the relative
differences G1M-C1M, G2M-G1M and G2M-XCU-G2M:

    updraft_and_cloud_cover            updraft extent, convective upward mass flux,
                                       cloud cover
    downdraft_and_vertical_velocity    downdraft extent, convective downward mass flux,
                                       resolved downward velocity

The mass flux rows follow the supervisor's procedure (the profiler note, section 1):

* each column is split into the 3MT updraft, the downdraft and the remaining
  environment on the instantaneous hourly fields, before any averaging, and the rising
  (sinking) part of every piece is weighted by its area (``column_fluxes``). UD_OMEGA is
  relative to the updraft environment, DD_OMEGA absolute, and the resolved omega is the
  area mean of all three. The downward flux is computed on its own, not subtracted
  from the upward one;
* only convective columns count: rain >= 1 mm in the hour and a 3MT updraft (mesh
  fraction > 1e-3 at some level); G2M-XCU, without 3MT, by rain alone;
* fluxes are averaged over 20 km boxes (5 x 5 cells) and divided by the full box area;
* the mean is over the box-hours that hold convection.

The other rows are domain means over the 4 km radar domain. Caches:
``build_iop_updraft_partition`` (convective boxes), ``build_iop_updraft_cloud_partition``
(updraft extent, cloud cover, temperature) and ``build_iop_convection_diurnal``
(downdraft extent, resolved velocity).
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
from alaro_analysis.common.constants import EXPERIMENT_DISPLAY, EXPERIMENT_LABELS
from alaro_analysis.common.experiments import shown_runs
from alaro_analysis.common.figio import DEFAULT_DPI, LEGEND_FONTSIZE, convert_to_jpg
from alaro_analysis.common.iop_settings import PLOT_TOP_KM, FIGURES, GOAMAZON_IOPS, IOP_PERIODS, figure_files, write_figure_index
from alaro_analysis.common.models import VerticalAxis
from alaro_analysis.common.vertical import centers_to_edges, compute_freezing_line_km
from alaro_analysis.plotting.scales import robust_anomaly_scale
from alaro_analysis.recipes.iop.build_iop_convection_diurnal import DEFAULT_WORK_DIR
from alaro_analysis.recipes.iop.make_iop_convection_figures import (
    AXIS_FS,
    CBAR_LABEL_FS,
    CBAR_TICK_FS,
    HEADER_FS,
    LETTER_FS,
    MAX_HEIGHT_KM,
    TICK_FS,
    merge_sections,
    relative_bias,
)

EXPS = shown_runs(("control-nodiag", "control", "graupel", "2mom", "2mom-xcu"))
SCHEME_EXPS = shown_runs(("control-nodiag", "control", "graupel", "2mom"))
FLUX_UNIT = "kg m$^{-2}$ s$^{-1}$"
# figure (folder name and stem) -> (note on the empty G2M-XCU panel, rows); a row is
# (quantity, cache file, row label, unit, unit interval, runs that have it).
FIGURE_SPECS = {
    "updraft_and_cloud_cover": ("no parameterized updraft", (
        ("ud_extent", "{label}_partition_ud_extent", "Updraft extent", "Fraction", True, SCHEME_EXPS),
        ("box_up", "{label}_box_up", "Convective upward\nmass flux", FLUX_UNIT, False, EXPS),
        ("cloud", "{label}_partition_cloud", "Cloud cover", "Fraction", True, EXPS),
    )),
    "downdraft_and_vertical_velocity": ("no parameterized downdraft", (
        ("dd_extent", "{label}_dd_extent", "Downdraft extent", "Fraction", True, SCHEME_EXPS),
        ("box_down", "{label}_box_down", "Convective downward\nmass flux", FLUX_UNIT, False, EXPS),
        ("w_dn", "{label}_w_dn", "Resolved downward\nvelocity", "m s$^{-1}$", False, EXPS),
    )),
}
# Column -> (numerator experiment, reference experiment): each run against the one before it;
# column 0 is the first run absolute.
DIFFS = tuple(zip(EXPS[1:], EXPS[:-1]))
PLAIN_UNITS = {"Fraction": "fraction", FLUX_UNIT: "kg m-2 s-1", "m s$^{-1}$": "m s-1"}
COLUMN_TITLES = (
    f"{EXPERIMENT_DISPLAY[EXPS[0]]} (absolute)",
    *(f"({EXPERIMENT_DISPLAY[a]} − {EXPERIMENT_DISPLAY[b]})"
      f"{chr(10) if len(EXPERIMENT_DISPLAY[a] + EXPERIMENT_DISPLAY[b]) > 12 else ' '}/ {EXPERIMENT_DISPLAY[b]}"
      for a, b in DIFFS),
)
ROW_NOTES = {
    "ud_extent": "Updraft extent: 3MT UD_MESH_FRAC, domain mean over the SIPAM radar domain on the native 4 km "
                 "grid (2960 cells, ALARO-RUNS/mask/Radar_mask_4km_native.nc); every cell counts.",
    "dd_extent": "Downdraft extent: 3MT DD_MESH_FRAC, domain mean over the SIPAM radar domain on the native "
                 "4 km grid (2960 cells); every cell counts.",
    "cloud": "Cloud cover: CLOUD_FRACTI, domain mean over the same 2960 cells.",
    "w_dn": "Resolved downward velocity: domain mean of max(-w, 0) of the resolved VERT.VELOCIT over the same "
            "2960 cells; rising cells count 0, so it is the mean sinking motion, not a downdraft speed.",
    "box_up": "Convective upward mass flux (the supervisor's procedure): rising part of every piece x its area.",
    "box_down": "Convective downward mass flux (the supervisor's procedure): sinking part of every piece x its "
                "area (3MT downdraft, compensating subsidence of the updraft environment, resolved sinking), "
                "positive down, computed on its own and not subtracted from the upward flux.",
}


def load_period(work: Path, iops: tuple[int, ...], rows) -> dict:
    """Cached sections of every experiment and row for one period; NaN where a run lacks a row."""
    data: dict = {"sections": {}, "files": {}, "boxes": {}}
    for exp in EXPS:
        label = EXPERIMENT_LABELS[exp]

        def merged(name: str) -> np.ndarray:
            parts = []
            for iop in iops:
                with np.load(work / f"iop{iop}" / f"{name.format(label=label)}.npz") as cache:
                    parts.append((cache["mean"], cache["counts"]))
            return merge_sections(parts)

        sections = {"temp": merged("{label}_partition_temp")}
        for quantity, name, _, _, _, available in rows:
            sections[quantity] = merged(name) if exp in available else np.full_like(sections["temp"], np.nan)
        data["sections"][exp] = sections
        data["files"][exp] = sum(int(np.load(work / f"iop{iop}" / f"{label}_partition_temp.npz")["n_files"])
                                 for iop in iops)
        data["boxes"][exp] = sum(int(np.load(work / f"iop{iop}" / f"{label}_box_up.npz")["counts"][0].sum())
                                 for iop in iops)
        with np.load(work / f"iop{iops[0]}" / f"{label}_partition_height.npz") as cache:
            data.setdefault("height_km", np.asarray(cache["height_m"], dtype=np.float64) / 1000.0)

    keep = np.isfinite(data["height_km"]) & (data["height_km"] <= MAX_HEIGHT_KM)
    data["height_km"] = data["height_km"][keep]
    for exp in EXPS:
        data["sections"][exp] = {k: v[keep, :] for k, v in data["sections"][exp].items()}
    return data



def period_header(data: dict, iops: tuple[int, ...], rows) -> list[str]:
    names = " and ".join(f"IOP{iop}" for iop in iops)
    combined = " combined" if len(iops) > 1 else ""
    lines = [f"Period: GoAmazon2014/5 {names}{combined} (Martin et al. 2016, ACP 16, 4785)."]
    for iop in iops:
        start, end = GOAMAZON_IOPS[iop]
        lines.append(f"IOP{iop}: {start} to {end} inclusive, {(end - start).days + 1} days.")
    files = ", ".join(f"{EXPERIMENT_DISPLAY[e]} {data['files'][e]}" for e in EXPS)
    lines += [
        f"Hourly files reduced: {files}. Every hour of the campaign days is used; the hourly files "
        "are instantaneous snapshots. Hour is Amazon local time (UTC-4).",
        "All runs from raw FA: C1M from the continuous control run, G1M, G2M and G2M-XCU from the "
        "IOP warm-restart runs.",
        *[ROW_NOTES[quantity] for quantity, *_ in rows],
        "Mass flux rows: each column is split into the 3MT updraft, the downdraft and the remaining "
        "environment (data.convection_sections.column_fluxes). UD_OMEGA is the updraft velocity "
        "relative to its environment (ACCVUD; Gerard et al. 2009, eq. 4), DD_OMEGA is absolute "
        "(ACMODO), and the resolved VITESSE_VERT is the area mean of all three. Convective columns "
        "only: rain >= 1 mm in the hour ending at the snapshot and a 3MT updraft (mesh fraction > "
        "1e-3 at some level; G2M-XCU by rain alone); fluxes of the other columns count as 0; 20 km "
        "boxes (5 x 5 cells, 116 boxes at least half inside the domain) divided by the full box area; "
        "mean over the box-hours with at least one convective column. Box-hours: "
        + ", ".join(f"{EXPERIMENT_DISPLAY[e]} {data['boxes'][e]}" for e in EXPS) + ".",
        "G2M-XCU runs without 3MT: its column is one piece, so its fluxes are the resolved motion, it "
        "has no parameterized draft extent (panel left empty), and its convective columns are picked "
        "by rain alone: it keeps weak raining boxes that the 3MT runs drop for lack of an updraft, which lowers "
        "its flux per box (supplementary convective_flux_sampling picks G2M by rain alone too).",
        "Draft fields cleaned as the 3MT physics does on entry (advection overshoots): mesh fraction "
        "< 0 -> 0, UD_OMEGA > 0 -> 0, DD_OMEGA < 0 -> 0.",
        "Relative differences are 100 * (model - reference) / reference, masked where the reference "
        "is below 5 % of its 98th percentile. Grey cells were never sampled or are masked that way.",
    ]
    return lines


def draw(data: dict, rows, empty_note: str, png: Path, dpi: int) -> list[dict]:
    """Rows are quantities, columns are the first run absolute and the relative differences.

    Column headers run along the top and the quantity labels down the left, so no
    panel repeats them. Each row carries two horizontal colour bars: the absolute
    scale under column 1 and the shared relative scale under columns 2 to 5.
    """
    import cmaps

    height = data["height_km"]
    y_edges = centers_to_edges(height)
    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    axis = VerticalAxis(values=height, label="Height (km)", is_height_km=True)
    freezing = {
        exp: compute_freezing_line_km(axis, data["sections"][exp]["temp"]) for exp in EXPS
    }

    fig, axes = plt.subplots(3, len(EXPS), figsize=(6.0 * len(EXPS), 16.5), constrained_layout=True)
    letters = iter("abcdefghijklmno")
    records: list[dict] = []
    freezing_handle = None

    for r, (quantity, _, title, unit, unit_interval, available) in enumerate(rows):
        sections = {e: data["sections"][e][quantity] for e in EXPS}
        abs_ctrl = np.ma.masked_invalid(sections[EXPS[0]])
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
            "label": title.replace("\n", " "), "unit": unit, "abs": np.ma.filled(abs_ctrl, np.nan),
            "vmin": vmin, "vmax": vmax, "anom": anom,
        }
        for ci, rel in enumerate(rels):
            record[f"rel{ci + 1}"] = None if rel is None else np.ma.filled(rel, np.nan)
        records.append(record)

        fields = [abs_ctrl, *rels]
        cmaps_used = [cmaps.WhiteBlueGreenYellowRed, *["RdBu_r"] * len(DIFFS)]
        pcms: list = []
        for ci, ax in enumerate(axes[r]):
            ax.set_facecolor("#d3d3d3")
            ax.set_xlim(-0.5, 23.5)
            ax.set_ylim(0.0, PLOT_TOP_KM)
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
            if r == len(rows) - 1:
                ax.set_xlabel("Hour (Amazon local time, UTC-4)", fontsize=AXIS_FS)
            ax.text(
                0.025, 0.97, f"({next(letters)})", transform=ax.transAxes, ha="left", va="top",
                fontsize=LETTER_FS, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2.0},
            )
            if fields[ci] is None:  # no parameterized draft without 3MT
                ax.set_facecolor("white")
                for side in ax.spines.values():
                    side.set_color("#b0b0b0")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(
                    0.5, 0.5,
                    f"{EXPERIMENT_LABELS['2mom-xcu']} runs without 3MT:\n{empty_note}",
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


def write_txt(txt: Path, data: dict, records: list[dict], header: list[str], empty_note: str) -> None:
    height = data["height_km"]
    lines = list(header)
    lines.append("")
    lines.append(
        "Panels: rows are " + ", ".join(r["label"].lower() for r in records) + "; columns are "
        f"{EXPERIMENT_DISPLAY[EXPS[0]]} absolute and the relative differences "
        + ", ".join(f"{EXPERIMENT_DISPLAY[a]}-{EXPERIMENT_DISPLAY[b]}" for a, b in DIFFS)
        + "."
    )
    for record in records:
        unit = PLAIN_UNITS.get(record["unit"], record["unit"])
        panels = [(f"{EXPERIMENT_DISPLAY[EXPS[0]]} absolute ({unit})", record["abs"])]
        for ci, (a, b) in enumerate(DIFFS):
            name = f"({EXPERIMENT_DISPLAY[a]} - {EXPERIMENT_DISPLAY[b]}) / {EXPERIMENT_DISPLAY[b]} (%)"
            panels.append((name, record.get(f"rel{ci + 1}")))
        for name, values in panels:
            if values is None:
                lines += ["", f"{record['label']}: {name}", f"  not applicable ({empty_note})"]
                continue
            lines += ["", f"{record['label']}: {name}",
                      ",".join(["height_km", *[f"h{h:02d}" for h in range(24)]])]
            for z, row in zip(height, values):
                cells = ["" if not np.isfinite(v) else f"{v:.6g}" for v in row]
                lines.append(",".join([f"{z:.3f}", *cells]))
    txt.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--output-dir", type=Path, default=FIGURES)
    p.add_argument("--periods", nargs="+", default=list(IOP_PERIODS), choices=list(IOP_PERIODS))
    p.add_argument("--figures", nargs="+", default=list(FIGURE_SPECS), choices=list(FIGURE_SPECS))
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_iop_draft_sections", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for figure in args.figures:
        empty_note, rows = FIGURE_SPECS[figure]
        for key in args.periods:
            iops = IOP_PERIODS[key]
            data = load_period(args.work_dir, iops, rows)
            header = period_header(data, iops, rows)
            png, jpg, txt = figure_files(args.output_dir, figure, figure, key)
            records = draw(data, rows, empty_note, png, args.dpi)
            convert_to_jpg(png, jpg, dpi=args.dpi)
            png.unlink()
            write_txt(txt, data, records, header, empty_note)
            print(f"[saved] {jpg}\n[saved] {txt}", flush=True)
    write_figure_index(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
