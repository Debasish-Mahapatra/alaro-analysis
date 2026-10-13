#!/usr/bin/env python3
"""Updraft and downdraft figures for the GoAmazon IOPs, from the cached diurnal sections.

Three figures per period, for C1M, G1M and G2M (G2M-XCU has no parameterized
updraft or downdraft, so it is left out):

    7. updraft and downdraft profiles   six panels, all-hours mean profile per
                                        experiment: updraft and downdraft extent,
                                        mass flux and resolved velocity
    8. updraft and cloud cover          paper layout: rows updraft extent, updraft
                                        mass flux, cloud cover; columns C1M absolute,
                                        (G1M-C1M)/C1M, (G2M-G1M)/G1M
    9. downdraft and vertical velocity  same layout with downdraft extent, downdraft
                                        mass flux and resolved downdraft velocity

Sections come from ``build_iop_convection_diurnal``; periods are IOP1+IOP2
combined (count weighted), IOP1 and IOP2. Every hour of the campaign days counts,
so there is no observational time mask here; the file counts are in the data text.
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
from alaro_analysis.common.constants import EXPERIMENT_COLORS, EXPERIMENT_LABELS
from alaro_analysis.common.figio import DEFAULT_DPI, convert_to_jpg
from alaro_analysis.common.iop_settings import (
    FIGURES,
    GOAMAZON_IOPS,
    IOP_PERIODS,
    figure_files,
    write_figure_index,
)
from alaro_analysis.common.models import VerticalAxis
from alaro_analysis.common.vertical import centers_to_edges, compute_freezing_line_km
from alaro_analysis.plotting.scales import robust_anomaly_scale
from alaro_analysis.recipes.iop.build_iop_convection_diurnal import (
    DEFAULT_WORK_DIR,
    QUANTITIES,
)

EXPS = ("control", "graupel", "2mom")
FIGURE_NUMBERS = {"profiles": 7, "updraft": 8, "downdraft": 9}
SECTION_TITLES = {
    "updraft": "Updraft and cloud cover",
    "downdraft": "Downdraft and vertical velocity",
}
FIGURE_STEMS = {
    "profiles": "updraft_downdraft_profiles",
    "updraft": "updraft_and_cloud_cover",
    "downdraft": "downdraft_and_vertical_velocity",
}
LABELS = {
    "ud_extent": ("Updraft extent", "Fraction", True),
    "ud_flux": ("Updraft mass flux", "kg m$^{-2}$ s$^{-1}$", False),
    "w_up": ("Resolved updraft velocity", "m s$^{-1}$", False),
    "dd_extent": ("Downdraft extent", "Fraction", True),
    "dd_flux": ("Downdraft mass flux", "kg m$^{-2}$ s$^{-1}$", False),
    "w_dn": ("Resolved downdraft velocity", "m s$^{-1}$", False),
    "cloud": ("Cloud cover", "Fraction", True),
}
PLAIN_UNITS = {
    "Fraction": "fraction",
    "kg m$^{-2}$ s$^{-1}$": "kg m-2 s-1",
    "m s$^{-1}$": "m s-1",
}
PROFILE_PANELS = ("ud_extent", "ud_flux", "w_up", "dd_extent", "dd_flux", "w_dn")
SECTION_ROWS = {
    "updraft": ("ud_extent", "ud_flux", "cloud"),
    "downdraft": ("dd_extent", "dd_flux", "w_dn"),
}
MAX_HEIGHT_KM = 16.0
REF_FRACTION = 0.05  # mask the relative bias where the reference is below 5 % of its p98
TICK_FS, AXIS_FS, HEADER_FS, LETTER_FS = 19, 21, 18, 16
CBAR_LABEL_FS, CBAR_TICK_FS, LEGEND_FS = 19, 16, 14


def merge_sections(parts: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Count-weighted mean of one quantity over several IOPs.

    Each part is ``(mean, counts)`` on the same (level, hour) grid; cells no IOP
    sampled come back NaN.
    """
    total = None
    weight = None
    for mean, counts in parts:
        mean = np.asarray(mean, dtype=np.float64)
        counts = np.asarray(counts, dtype=np.float64)
        if total is None:
            total = np.zeros_like(mean)
            weight = np.zeros_like(mean)
        valid = np.isfinite(mean) & (counts > 0)
        total[valid] += mean[valid] * counts[valid]
        weight[valid] += counts[valid]
    out = np.full_like(total, np.nan)
    seen = weight > 0
    out[seen] = total[seen] / weight[seen]
    return out


def load_period(work: Path, iops: tuple[int, ...]) -> dict:
    """Cached sections of every experiment for one period, cropped to MAX_HEIGHT_KM."""
    data: dict = {"sections": {}, "files": {}}
    for exp in EXPS:
        label = EXPERIMENT_LABELS[exp]
        sections, files = {}, 0
        for quantity in QUANTITIES:
            parts = []
            for iop in iops:
                with np.load(work / f"iop{iop}" / f"{label}_{quantity}.npz") as cache:
                    parts.append((cache["mean"], cache["counts"]))
                    files = max(files, int(cache["n_files"]))
            sections[quantity] = merge_sections(parts)
        with np.load(work / f"iop{iops[0]}" / f"{label}_height.npz") as cache:
            height_km = np.asarray(cache["height_m"], dtype=np.float64) / 1000.0
        data["sections"][exp] = sections
        data["files"][exp] = files * len(iops)
        data.setdefault("height_km", height_km)

    keep = np.isfinite(data["height_km"]) & (data["height_km"] <= MAX_HEIGHT_KM)
    data["height_km"] = data["height_km"][keep]
    for exp in EXPS:
        data["sections"][exp] = {k: v[keep, :] for k, v in data["sections"][exp].items()}
    return data


def relative_bias(numerator: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """100 * (model - ref) / ref, masked where the reference is negligibly small."""
    ref = np.asarray(reference, dtype=np.float64)
    positive = ref[np.isfinite(ref) & (ref > 0.0)]
    threshold = REF_FRACTION * float(np.percentile(positive, 98.0)) if positive.size else np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = 100.0 * numerator / np.where(ref != 0.0, ref, np.nan)
    return np.where(np.isfinite(ref) & (ref > threshold), rel, np.nan)


def hour_mean(section: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(section, axis=1)


def freezing_km(temp_section: np.ndarray, height_km: np.ndarray) -> float:
    """Height where the all-hours mean temperature crosses 0 degC."""
    profile = hour_mean(temp_section) - 273.15
    finite = np.isfinite(profile) & np.isfinite(height_km)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    z, t = height_km[finite], profile[finite]
    crossings = np.flatnonzero((t[:-1] > 0.0) & (t[1:] <= 0.0))
    if crossings.size == 0:
        return float("nan")
    i = int(crossings[0])
    return float(z[i] + (0.0 - t[i]) * (z[i + 1] - z[i]) / (t[i + 1] - t[i]))


def figure_outputs(root: Path, kind: str, period: str) -> tuple[Path, Path, Path]:
    return figure_files(root, FIGURE_NUMBERS[kind], FIGURE_STEMS[kind], period)


def period_header(data: dict, iops: tuple[int, ...]) -> list[str]:
    names = " and ".join(f"IOP{iop}" for iop in iops)
    combined = " combined" if len(iops) > 1 else ""
    lines = [f"Period: GoAmazon2014/5 {names}{combined} (Martin et al. 2016, ACP 16, 4785)."]
    for iop in iops:
        start, end = GOAMAZON_IOPS[iop]
        lines.append(f"IOP{iop}: {start} to {end} inclusive, {(end - start).days + 1} days.")
    files = ", ".join(f"{EXPERIMENT_LABELS[e]} {data['files'][e]}" for e in EXPS)
    lines += [
        f"Hourly files reduced: {files}. Every hour of the campaign days is used; "
        "no observational time mask applies here.",
        "Domain mean over the SIPAM radar domain on the native 4 km grid (2960 cells, "
        "ALARO-RUNS/mask/Radar_mask_4km_native.nc); every cell counts, with or without a draft. "
        "Hour is Amazon local time (UTC-4).",
        "All runs from raw FA: C1M from the continuous control run, G1M and G2M from the IOP "
        "warm-restart runs.",
        "G2M-XCU is not shown: it has no parameterized updraft or downdraft.",
        "Draft fields cleaned as the 3MT physics does on entry (advection overshoots): mesh "
        "fraction < 0 -> 0, UD_OMEGA > 0 -> 0, DD_OMEGA < 0 -> 0.",
        "Updraft mass flux = mean(-min(UD_OMEGA, 0) * max(mesh, 0) / g), positive upward.",
        "Downdraft mass flux = mean(max(DD_OMEGA, 0) * max(mesh, 0) / g), positive downward.",
        "Resolved velocities from VERT.VELOCIT: w_up = mean(max(w, 0)), w_dn = mean(max(-w, 0)).",
    ]
    return lines


def draw_profiles(data: dict, png: Path, dpi: int) -> dict[str, dict[str, np.ndarray]]:
    """Six panels of all-hours mean profiles, one curve per experiment."""
    height = data["height_km"]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 12.0), sharey=True)
    curves: dict[str, dict[str, np.ndarray]] = {}
    freezing = {exp: freezing_km(data["sections"][exp]["temp"], height) for exp in EXPS}

    for index, quantity in enumerate(PROFILE_PANELS):
        ax = axes[index // 3, index % 3]
        title, unit, _ = LABELS[quantity]
        curves[quantity] = {}
        for exp in EXPS:
            profile = hour_mean(data["sections"][exp][quantity])
            curves[quantity][exp] = profile
            ax.plot(
                profile,
                height,
                color=EXPERIMENT_COLORS[exp],
                lw=2.0,
                label=EXPERIMENT_LABELS[exp],
            )
        for exp in EXPS:
            if np.isfinite(freezing[exp]):
                ax.axhline(freezing[exp], color=EXPERIMENT_COLORS[exp], lw=1.1, ls=":", alpha=0.9)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel(unit, fontsize=13)
        ax.set_ylim(0.0, MAX_HEIGHT_KM)
        ax.grid(True, color="0.88", linewidth=0.8)
        ax.tick_params(axis="both", labelsize=12)
        # Small-valued panels otherwise crowd their tick labels into each other.
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.text(
            0.0,
            1.015,
            f"({'abcdef'[index]})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )
    for row in range(2):
        axes[row, 0].set_ylabel("Height above mean sea level (km)", fontsize=13)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles = list(handles) + [plt.Line2D([0], [0], color="0.4", lw=1.1, ls=":")]
    labels = list(labels) + ["0 °C height"]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.11, wspace=0.16, hspace=0.26)
    fig.savefig(png, dpi=dpi, facecolor="white")
    plt.close(fig)
    curves["freezing"] = freezing
    return curves


def draw_sections(data: dict, kind: str, png: Path, dpi: int) -> list[dict]:
    """Paper layout: rows are quantities, columns are C1M and the two relative differences."""
    import cmaps

    height = data["height_km"]
    y_edges = centers_to_edges(height)
    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    axis = VerticalAxis(values=height, label="Height (km)", is_height_km=True)
    freezing = {
        exp: compute_freezing_line_km(axis, data["sections"][exp]["temp"]) for exp in EXPS
    }

    fig, axes = plt.subplots(3, 3, figsize=(20.0, 21.0), constrained_layout=True)
    letters = iter("abcdefghi")
    records: list[dict] = []

    for r, quantity in enumerate(SECTION_ROWS[kind]):
        title, unit, unit_interval = LABELS[quantity]
        ctrl = data["sections"]["control"][quantity]
        g1m = data["sections"]["graupel"][quantity]
        g2m = data["sections"]["2mom"][quantity]
        abs_ctrl = np.ma.masked_invalid(ctrl)
        rel1 = np.ma.masked_invalid(relative_bias(g1m - ctrl, ctrl))
        rel2 = np.ma.masked_invalid(relative_bias(g2m - g1m, g1m))

        valid = abs_ctrl.compressed()
        vmin = max(0.0, float(np.percentile(valid, 2.0))) if valid.size else 0.0
        vmax = float(np.percentile(valid, 98.0)) if valid.size else 1.0
        if unit_interval:
            vmax = min(1.0, vmax)
        if vmax <= vmin:
            vmax = vmin + 1e-9
        anom = robust_anomaly_scale(rel1.filled(np.nan), rel2.filled(np.nan))
        if not np.isfinite(anom) or anom <= 0:
            anom = 1.0
        norms = (
            mcolors.Normalize(vmin=vmin, vmax=vmax),
            mcolors.TwoSlopeNorm(vmin=-anom, vcenter=0.0, vmax=anom),
        )
        records.append(
            {
                "label": title,
                "unit": unit,
                "abs": np.ma.filled(abs_ctrl, np.nan),
                "rel1": np.ma.filled(rel1, np.nan),
                "rel2": np.ma.filled(rel2, np.nan),
                "vmin": vmin,
                "vmax": vmax,
                "anom": anom,
            }
        )

        fields = (abs_ctrl, rel1, rel2)
        titles = (
            f"{EXPERIMENT_LABELS['control']} ({title}, absolute)",
            f"({EXPERIMENT_LABELS['graupel']} − {EXPERIMENT_LABELS['control']})"
            f" / {EXPERIMENT_LABELS['control']}",
            f"({EXPERIMENT_LABELS['2mom']} − {EXPERIMENT_LABELS['graupel']})"
            f" / {EXPERIMENT_LABELS['graupel']}",
        )
        cmaps_used = (cmaps.WhiteBlueGreenYellowRed, "RdBu_r", "RdBu_r")
        pcms = []
        for ci, ax in enumerate(axes[r]):
            ax.set_facecolor("#d3d3d3")
            pcms.append(
                ax.pcolormesh(
                    hour_edges,
                    y_edges,
                    fields[ci],
                    cmap=cmaps_used[ci],
                    norm=norms[0] if ci == 0 else norms[1],
                    shading="auto",
                )
            )
            ax.set_title(titles[ci], fontsize=HEADER_FS, fontweight="bold", pad=18)
            ax.set_xlabel("Hour (Amazon UTC-4)", fontsize=AXIS_FS)
            ax.set_ylabel("Height (km)", fontsize=AXIS_FS)
            ax.set_xticks(np.arange(0, 24, 6))
            ax.set_xlim(-0.5, 23.5)
            ax.set_ylim(0.0, MAX_HEIGHT_KM)
            ax.tick_params(axis="both", labelsize=TICK_FS)
            ax.text(
                0.02,
                0.98,
                f"({next(letters)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=LETTER_FS,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2.0},
            )
            line = freezing[EXPS[ci]]
            if line is not None and np.isfinite(line).any():
                ax.plot(
                    np.arange(24), line, color="black", lw=1.8, ls="--", zorder=10,
                    label="Freezing level",
                )
                if r == 0 and ci == 0:
                    ax.legend(loc="upper right", fontsize=LEGEND_FS, framealpha=0.9)

        cbar = fig.colorbar(
            pcms[0], ax=axes[r, 0], orientation="horizontal", fraction=0.08, pad=0.04
        )
        cbar.set_label(unit, fontsize=CBAR_LABEL_FS)
        cbar.ax.tick_params(labelsize=CBAR_TICK_FS)
        cbar.ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        cbar.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

        cbar_rel = fig.colorbar(
            pcms[1], ax=[axes[r, 1], axes[r, 2]], orientation="horizontal", fraction=0.08, pad=0.04
        )
        cbar_rel.set_label("Relative difference (%)", fontsize=CBAR_LABEL_FS)
        cbar_rel.ax.tick_params(labelsize=CBAR_TICK_FS)
        cbar_rel.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return records


def write_profile_txt(txt: Path, data: dict, curves: dict, header: list[str]) -> None:
    height = data["height_km"]
    title = "Updraft and downdraft profiles"
    lines = [title, "=" * len(title), *header, "", "All-hours mean profiles.", ""]
    lines += ["0 degC height (km)", "------------------", "series,z_freeze_km"]
    lines += [
        f"{EXPERIMENT_LABELS[e]},{curves['freezing'][e]:.6g}"
        if np.isfinite(curves["freezing"][e])
        else f"{EXPERIMENT_LABELS[e]},nan"
        for e in EXPS
    ]
    for quantity in PROFILE_PANELS:
        name, unit, _ = LABELS[quantity]
        heading = f"{name} [{PLAIN_UNITS[unit]}]"
        lines += ["", heading, "-" * len(heading)]
        lines.append(",".join(["height_km", *[EXPERIMENT_LABELS[e] for e in EXPS]]))
        for i, z in enumerate(height):
            row = [f"{z:.4f}"]
            row += [
                "nan" if not np.isfinite(curves[quantity][e][i]) else f"{curves[quantity][e][i]:.6g}"
                for e in EXPS
            ]
            lines.append(",".join(row))
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_section_txt(txt: Path, data: dict, kind: str, records: list[dict], header: list[str]) -> None:
    height = data["height_km"]
    title = f"{SECTION_TITLES[kind]} - relative-bias panels"
    lines = [title, "=" * len(title), *header]
    lines += [
        f"Columns per quantity: C1M absolute; (G1M-C1M)/C1M [%]; (G2M-G1M)/G1M [%].",
        f"Relative bias masked where the reference is below {REF_FRACTION:g} x its 98th percentile.",
        f"Height cropped to {MAX_HEIGHT_KM:g} km.",
    ]
    for record in records:
        lines += [
            "",
            f"## {record['label']}",
            f"absolute C1M limits: vmin={record['vmin']:.6g} vmax={record['vmax']:.6g} "
            f"[{PLAIN_UNITS[record['unit']]}]; relative colour scale: +/-{record['anom']:.6g} %",
        ]
        for key, block in (
            ("abs", f"C1M absolute [{PLAIN_UNITS[record['unit']]}]"),
            ("rel1", "(G1M-C1M)/C1M [%]"),
            ("rel2", "(G2M-G1M)/G1M [%]"),
        ):
            lines += [
                f"-- {block} -- rows=height_km, cols=hour 0..23",
                "height_km," + ",".join(f"h{h:02d}" for h in range(24)),
            ]
            field = record[key]
            for i, z in enumerate(height):
                values = ",".join(
                    "nan" if not np.isfinite(field[i, j]) else f"{field[i, j]:.6g}"
                    for j in range(field.shape[1])
                )
                lines.append(f"{z:.4f},{values}")
            lines.append("")
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--output-dir", type=Path, default=FIGURES)
    p.add_argument("--periods", nargs="+", default=list(IOP_PERIODS), choices=list(IOP_PERIODS))
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_iop_convection_figures", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for key in args.periods:
        iops = IOP_PERIODS[key]
        data = load_period(args.work_dir, iops)
        header = period_header(data, iops)
        root = args.output_dir

        png, jpg, txt = figure_outputs(root, "profiles", key)
        curves = draw_profiles(data, png, args.dpi)
        convert_to_jpg(png, jpg, dpi=args.dpi)
        png.unlink()
        write_profile_txt(txt, data, curves, header)
        print(f"[saved] {jpg}\n[saved] {txt}", flush=True)

        for kind in ("updraft", "downdraft"):
            png, jpg, txt = figure_outputs(root, kind, key)
            records = draw_sections(data, kind, png, args.dpi)
            convert_to_jpg(png, jpg, dpi=args.dpi)
            png.unlink()
            write_section_txt(txt, data, kind, records, header)
            print(f"[saved] {jpg}\n[saved] {txt}", flush=True)
    write_figure_index(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
