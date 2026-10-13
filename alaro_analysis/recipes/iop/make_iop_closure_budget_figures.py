#!/usr/bin/env python3
"""Closure-budget compression figure for the GoAmazon IOPs, for all four experiments.

One panel per experiment (C1M, G1M, G2M, G2M-XCU), in the layout of the paper's
closure figure: total DDH condensation with its convection-scheme and
resolved-microphysics parts, the updraft mass flux on a green twin axis, the
condensation peak height and the 0 degC isotherm.

Condensation comes from the DDH QV budget aggregated by
``build_iop_closure_budgets`` (convective = max(-condcv, 0), resolved =
max(-condrs, 0)); the 0 degC line from the DDH temperature of the same days. The
updraft mass flux is the 4 km radar-domain profile already cached by
``build_iop_convection_diurnal``.

G2M-XCU carries no parameterized convection, so its convective curve is zero and
it has no updraft mass flux to draw.

A second figure per period stacks day (06-18 LT, Amazon local time UTC-4) over night
(18-06 LT), with one condensation and one updraft axis range for all panels, so the
two rows compare directly (``closure_budget_compression_day_night_<period>``). The
condensation comes from the day and night DDH windows of ``build_iop_closure_budgets``;
the updraft flux from the same local hours of the convection caches.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.figio import DEFAULT_DPI, convert_to_jpg, legend_below
from alaro_analysis.common.iop_settings import (
    FIGURES,
    GOAMAZON_IOPS,
    IOP_PERIODS,
    figure_files,
    write_figure_index,
)
from alaro_analysis.recipes.iop.build_iop_closure_budgets import DEFAULT_WORK_DIR, LABELS
from alaro_analysis.recipes.iop.build_iop_convection_diurnal import (
    DEFAULT_WORK_DIR as CONVECTION_WORK_DIR,
)

# Experiment directory -> key of the convection cache holding its updraft flux.
# G2M-XCU has no parameterized updraft, so it has no entry.
UPDRAFT_KEYS = {"control": "C1M", "graupel": "G1M", "2mom": "G2M"}
EXPERIMENTS = ("control", "graupel", "2mom", "NO3M")
FIGURE_NUMBER = 11
FIGURE_STEM = "closure_budget_compression"
DAY_NIGHT_STEM = "closure_budget_compression_day_night"
# Local-hour columns of the convection caches (Amazon local time, UTC-4).
DAY_HOURS = tuple(range(6, 18))
NIGHT_HOURS = tuple(h for h in range(24) if h not in DAY_HOURS)
HALF_LABELS = {"day": "Day, 06-18 LT", "night": "Night, 18-06 LT"}

Z_MAX_KM = 10.0
TOTAL_COLOR = "#111111"
CONVECTIVE_COLOR = "#D62728"
RESOLVED_COLOR = "#1F77B4"
UPDRAFT_COLOR = "#009E73"
PEAK_COLOR = "#4D4D4D"
FREEZING_COLOR = "#555555"
PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)")


def load_condensation(
    work: Path, iops: tuple[int, ...], experiment: str, half: str | None = None
) -> dict:
    """Day-weighted mean condensation partition of one experiment over a period.

    ``half`` = "day" or "night" reads that window's aggregate instead of the 24 h one.
    """
    totals: dict[str, np.ndarray] = {}
    temperature, weight, days = None, 0, 0
    altitude = None
    suffix = f"_{half}" if half else ""
    for iop in iops:
        path = work / "aggregates" / f"iop{iop}" / f"{experiment}_QV{suffix}.npz"
        with np.load(path, allow_pickle=True) as cache:
            n = int(np.asarray(cache["days"]).size)
            altitude = np.asarray(cache["altitude_km"], float)
            for block in ("condcv", "condrs"):
                values = np.asarray(cache[f"block__{block}"], float) * n
                totals[block] = values if block not in totals else totals[block] + values
            if "temperature_k" in cache.files:
                t = np.asarray(cache["temperature_k"], float) * n
                temperature = t if temperature is None else temperature + t
                weight += n
            days += n
    convective = np.maximum(-totals["condcv"] / days, 0.0)
    resolved = np.maximum(-totals["condrs"] / days, 0.0)
    return {
        "altitude_km": altitude,
        "convective": convective,
        "resolved": resolved,
        "total": convective + resolved,
        "temperature_k": None if temperature is None else temperature / max(weight, 1),
        "days": days,
    }


def load_updraft(
    convection_work: Path, iops: tuple[int, ...], experiment: str, hours=tuple(range(24))
):
    """Mean updraft mass-flux profile (kg m-2 s-1) over the given local hours, and height (km)."""
    label = UPDRAFT_KEYS.get(experiment)
    if label is None:
        return None
    total, weight, height = None, None, None
    for iop in iops:
        with np.load(convection_work / f"iop{iop}" / f"{label}_ud_flux.npz") as cache:
            mean = np.asarray(cache["mean"], float)[:, list(hours)]
            counts = np.asarray(cache["counts"], float)[:, list(hours)]
        valid = np.isfinite(mean) & (counts > 0)
        contribution = np.where(valid, mean * counts, 0.0)
        counted = np.where(valid, counts, 0.0)
        total = contribution if total is None else total + contribution
        weight = counted if weight is None else weight + counted
        with np.load(convection_work / f"iop{iop}" / f"{label}_height.npz") as cache:
            height = np.asarray(cache["height_m"], float) / 1000.0
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(weight.sum(axis=1) > 0, total.sum(axis=1) / weight.sum(axis=1), np.nan)
    return height, profile


def freezing_level_km(altitude_km, temperature_k) -> float:
    """Lowest 0 degC crossing of a temperature profile."""
    if temperature_k is None:
        return float("nan")
    z = np.asarray(altitude_km, float)
    t = np.asarray(temperature_k, float)
    ok = np.isfinite(z) & np.isfinite(t)
    z, t = z[ok], t[ok]
    if z.size < 2:
        return float("nan")
    order = np.argsort(z)
    z, t = z[order], t[order]
    d = t - 273.15
    cross = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if cross.size == 0:
        return float("nan")
    i = int(cross[0])
    return float(z[i] + d[i] / (d[i] - d[i + 1]) * (z[i + 1] - z[i]))


def peak_height_km(altitude_km, total) -> float:
    z, v = np.asarray(altitude_km, float), np.asarray(total, float)
    ok = np.isfinite(z) & np.isfinite(v) & (z <= Z_MAX_KM)
    if not ok.any():
        return float("nan")
    return float(z[ok][np.argmax(v[ok])])


def column_integral(altitude_km, values, z_top=Z_MAX_KM) -> float:
    z, v = np.asarray(altitude_km, float), np.asarray(values, float)
    ok = np.isfinite(z) & np.isfinite(v)
    z, v = z[ok], v[ok]
    order = np.argsort(z)
    z, v = z[order], v[order]
    inside = (z > 0.0) & (z < z_top)
    grid = np.r_[0.0, z[inside], z_top]
    return float(np.trapezoid(np.interp(grid, z, v), grid))


def gather(
    work: Path, convection_work: Path, iops: tuple[int, ...], half: str | None = None
) -> dict:
    hours = {"day": DAY_HOURS, "night": NIGHT_HOURS}.get(half, tuple(range(24)))
    panels = {}
    for experiment in EXPERIMENTS:
        data = load_condensation(work, iops, experiment, half)
        data["z_peak"] = peak_height_km(data["altitude_km"], data["total"])
        data["z_freeze"] = freezing_level_km(data["altitude_km"], data["temperature_k"])
        data["column"] = column_integral(data["altitude_km"], data["total"])
        data["updraft"] = load_updraft(convection_work, iops, experiment, hours)
        panels[experiment] = data
    return panels


def axis_ranges(rows: list[dict]) -> tuple[float, float]:
    """Shared condensation and updraft-flux axis maxima over every panel of ``rows``."""
    panels = [d for row in rows for d in row.values()]
    cond_max = max(np.nanmax(d["total"][d["altitude_km"] <= Z_MAX_KM]) for d in panels) * 1.08
    fluxes = [d["updraft"][1] for d in panels if d["updraft"] is not None]
    fluxes += [line[1] for d in panels for line in d.get("updraft_lines", ())]
    flux_max = max(np.nanmax(f) for f in fluxes) * 1.08 if fluxes else 1.0
    return cond_max, flux_max


def draw(panels: dict, png: Path, dpi: int) -> tuple[float, float]:
    return draw_rows({None: panels}, png, dpi)


def draw_rows(rows: dict, png: Path, dpi: int, updraft_handles=None) -> tuple[float, float]:
    """One row of experiment panels per entry of ``rows`` (row label -> panels).

    A panel with ``updraft_lines`` [(height_km, flux, plot kwargs), ...] draws those
    on the updraft axis instead of its single ``updraft`` line; ``updraft_handles``
    then replaces the one updraft entry of the legend.
    """
    cond_max, flux_max = axis_ranges(list(rows.values()))
    nrow = len(rows)
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 16})
    width = 5.2 * len(EXPERIMENTS)
    fig, axes = plt.subplots(
        nrow, len(EXPERIMENTS), figsize=(width, 8.0 * nrow), sharey=True, squeeze=False
    )
    for row, (row_label, panels) in enumerate(rows.items()):
        for index, experiment in enumerate(EXPERIMENTS):
            data = panels[experiment]
            ax = axes[row][index]
            z = data["altitude_km"]
            if np.isfinite(data["z_peak"]):
                ax.axhspan(0, data["z_peak"], color="#BFC7D5", alpha=0.18, zorder=0)
            ax.plot(data["total"], z, color=TOTAL_COLOR, lw=2.8)
            ax.plot(data["convective"], z, color=CONVECTIVE_COLOR, lw=2.0, ls="-.", alpha=0.9)
            ax.plot(data["resolved"], z, color=RESOLVED_COLOR, lw=2.0, ls="--", alpha=0.95)
            if np.isfinite(data["z_peak"]):
                ax.axhline(data["z_peak"], color=PEAK_COLOR, lw=1.2, ls=(0, (6, 3)), alpha=0.9)
            if np.isfinite(data["z_freeze"]):
                ax.axhline(data["z_freeze"], color=FREEZING_COLOR, lw=1.2, ls=":", alpha=0.95)

            twin = ax.twiny()
            if data.get("updraft_lines"):
                for height, flux, style in data["updraft_lines"]:
                    twin.plot(flux, height, **{"color": UPDRAFT_COLOR, **style})
            elif data["updraft"] is not None:
                height, flux = data["updraft"]
                twin.plot(flux, height, color=UPDRAFT_COLOR, lw=2.1, alpha=0.95)
            else:
                twin.text(
                    0.5,
                    0.94,
                    "no parameterized updraft",
                    transform=twin.transAxes,
                    ha="center",
                    va="top",
                    fontsize=10,
                    color=UPDRAFT_COLOR,
                )
            twin.set_xlim(0, flux_max)
            twin.set_xlabel(
                r"updraft mass flux (kg m$^{-2}$ s$^{-1}$)", color=UPDRAFT_COLOR, labelpad=8
            )
            twin.tick_params(axis="x", colors=UPDRAFT_COLOR)
            twin.spines["top"].set_color(UPDRAFT_COLOR)
            twin.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

            ax.set_xlim(0, cond_max)
            ax.set_ylim(0, Z_MAX_KM)
            if row == 0:
                ax.set_title(LABELS[experiment], pad=32)
            ax.set_xlabel(r"condensation (g kg$^{-1}$ day$^{-1}$)")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.grid(alpha=0.25)
            if index == 0:
                ax.set_ylabel(f"{row_label}\naltitude (km)" if row_label else "altitude (km)")
            ax.text(
                0.97,
                0.96,
                PANEL_LABELS[row * len(EXPERIMENTS) + index],
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=14,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
                zorder=20,
            )

    handles = [
        Line2D([0], [0], color=TOTAL_COLOR, lw=2.8, label="total condensation"),
        Line2D([0], [0], color=CONVECTIVE_COLOR, lw=2.0, ls="-.", label="convection-scheme part"),
        Line2D([0], [0], color=RESOLVED_COLOR, lw=2.0, ls="--", label="resolved-microphysics part"),
        *(updraft_handles or [Line2D([0], [0], color=UPDRAFT_COLOR, lw=2.1, label="updraft mass flux")]),
        Line2D(
            [0], [0], color=PEAK_COLOR, lw=1.2, ls=(0, (6, 3)), label="condensation peak height"
        ),
        Line2D([0], [0], color=FREEZING_COLOR, lw=1.2, ls=":", label=r"0 $^{\circ}$C isotherm"),
    ]
    if nrow == 1:
        fig.subplots_adjust(left=0.06, right=0.985, top=0.84, bottom=0.10, wspace=0.12)
    else:
        fig.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.06, wspace=0.12, hspace=0.30)
    legend_below(fig, axes, handles, handlelength=2.6, columnspacing=1.8)
    fig.savefig(png, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return cond_max, flux_max


def write_txt(txt: Path, panels: dict, iops: tuple[int, ...]) -> None:
    txt.write_text("\n".join(txt_lines(panels, iops)) + "\n", encoding="utf-8")


def write_day_night_txt(txt: Path, halves: dict, iops: tuple[int, ...]) -> None:
    lines = [
        "Closure-budget compression by time of day",
        "=========================================",
        "Day = 06-18 LT, night = 18-06 LT (Amazon local time, UTC-4). DDH windows: day = leads",
        "10-22 UTC of the 00 UTC runs (+0022 minus +0010, ddht DIFFE_EC2_EC1); night = 00-10 UTC",
        "(+0010) and 22-24 UTC (+0024 minus +0022), combined by duration. The three windows add",
        "back to the 24 h budget to within 0.1 %. Updraft mass flux: the same local hours of the",
        "convection caches. The 0 degC line is the 24 h one in both rows.",
    ]
    for half, panels in halves.items():
        lines += ["", "#" * 60]
        lines += txt_lines(panels, iops, title=f"{half.upper()} ({HALF_LABELS[half]})")
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def txt_lines(panels: dict, iops: tuple[int, ...], title: str | None = None) -> list[str]:
    names = " and ".join(f"IOP{iop}" for iop in iops)
    combined = " combined" if len(iops) > 1 else ""
    title = title or "Closure-budget compression: DDH condensation and updraft mass flux"
    lines = [
        title,
        "=" * len(title),
        f"Period: GoAmazon2014/5 {names}{combined} (Martin et al. 2016, ACP 16, 4785).",
    ]
    for iop in iops:
        start, end = GOAMAZON_IOPS[iop]
        lines.append(f"IOP{iop}: {start} to {end} inclusive, {(end - start).days + 1} days.")
    lines += [
        "Condensation from the DDH QV budget at lead 0024, domain mean of the DDH namelist",
        "domain: convection-scheme part = max(-condcv, 0), resolved-microphysics part =",
        "max(-condrs, 0), total = their sum, in g kg-1 day-1. Day-weighted mean over the",
        "campaign days.",
        "0 degC isotherm from the DDH temperature of the same days (T = VCT0/VPP0/cp).",
        "Updraft mass flux is the all-hours mean over the 4 km radar domain (2960 cells) from",
        "the convection caches. The condensation profiles are DDH domain means.",
        "G2M-XCU has no parameterized convection: its convective part is zero and it has no",
        "updraft mass flux.",
        "",
        "experiment,days,column_0_10km_gkgday_km,peak_height_km,freezing_level_km",
    ]
    for experiment in EXPERIMENTS:
        d = panels[experiment]
        lines.append(
            f"{LABELS[experiment]},{d['days']},{d['column']:.6g},{d['z_peak']:.4g},"
            f"{d['z_freeze']:.4g}"
        )
    lines += ["", "experiment,altitude_km,convective_gkgday,resolved_gkgday,total_gkgday"]
    for experiment in EXPERIMENTS:
        d = panels[experiment]
        for z, c, r, t in zip(d["altitude_km"], d["convective"], d["resolved"], d["total"]):
            if z <= Z_MAX_KM:
                lines.append(f"{LABELS[experiment]},{z:.4f},{c:.6e},{r:.6e},{t:.6e}")
    lines += ["", "experiment,height_km,updraft_mass_flux_kg_m2_s"]
    for experiment in EXPERIMENTS:
        updraft = panels[experiment]["updraft"]
        if updraft is None:
            lines.append(f"{LABELS[experiment]},nan,nan")
            continue
        height, flux = updraft
        for h, f in zip(height, flux):
            if h <= Z_MAX_KM:
                lines.append(f"{LABELS[experiment]},{h:.4f},{f:.6e}")
    return lines


def figure_outputs(root: Path, period: str) -> tuple[Path, Path, Path]:
    return figure_files(root, FIGURE_NUMBER, FIGURE_STEM, period)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--convection-work-dir", type=Path, default=CONVECTION_WORK_DIR)
    p.add_argument("--output-dir", type=Path, default=FIGURES)
    p.add_argument("--periods", nargs="+", default=list(IOP_PERIODS), choices=list(IOP_PERIODS))
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_iop_closure_budget_figures", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for key in args.periods:
        iops = IOP_PERIODS[key]
        panels = gather(args.work_dir, args.convection_work_dir, iops)
        png, jpg, txt = figure_outputs(args.output_dir, key)
        draw(panels, png, args.dpi)
        convert_to_jpg(png, jpg, dpi=args.dpi)
        png.unlink()
        write_txt(txt, panels, iops)
        print(f"[saved] {jpg}\n[saved] {txt}", flush=True)

        halves = {
            half: gather(args.work_dir, args.convection_work_dir, iops, half)
            for half in ("day", "night")
        }
        png, jpg, txt = figure_files(args.output_dir, FIGURE_NUMBER, DAY_NIGHT_STEM, key)
        draw_rows({HALF_LABELS[h]: p for h, p in halves.items()}, png, args.dpi)
        convert_to_jpg(png, jpg, dpi=args.dpi)
        png.unlink()
        write_day_night_txt(txt, halves, iops)
        print(f"[saved] {jpg}\n[saved] {txt}", flush=True)
    write_figure_index(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
