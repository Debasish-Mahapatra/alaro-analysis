#!/usr/bin/env python3
"""Hydrometeor species by height: the CHUVA X-band radar against the four runs (IOP1).

The polarimetric radar is the only instrument covering IOP1 that can say which species
were present, so it is the only direct test of where the runs put graupel.

Both sides are counted the same way. The radar panel is the CSU fuzzy-logic class of
each gate (``build_iop_xband_hid``); the model panels are the species holding most of
the precipitating mass in each cell (``build_iop_model_species``). Both keep only
detectable points - above 0 dBZ, inside the radar domain - and both are binned into
0.5 km layers, then normalised per layer, so each panel reads as "of the detectable
points at this height, what fraction was each species".

Radar classes are grouped onto the model's species:

    rain      drizzle + rain + big drops        -> qr
    ice       ice crystals + vertical ice       -> qi
    snow      aggregates                        -> qs
    graupel   low- and high-density graupel + hail -> qg
    melting   wet snow                          -> no model counterpart, radar only

What the comparison cannot do: the radar labels one dominant species per gate, so a
species confined to narrow convective cores can carry much of the mass while labelling
few gates. Read it as "where is each species found", not "how much of it there is".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.figio import DEFAULT_DPI, LEGEND_FONTSIZE
from alaro_analysis.common.iop_settings import CACHE_DIRS, GOAMAZON_IOPS
from alaro_analysis.common.paths import data_path

OUTPUT_DIR = data_path("microphysics-paper/6. hydrometeor mixing-ratio cfad")
FIGURE_STEM = "6b. hydrometeor species vs X-band radar_iop1"
RUNS = ("C1M", "G1M", "G2M", "G2M-XCU")
# radar class (CSU order) -> grouped species
RADAR_GROUPS = {
    "rain": ("drizzle", "rain", "big drops"),
    "ice": ("ice crystals", "vertical ice"),
    "snow": ("aggregates",),
    "graupel": ("low-density graupel", "high-density graupel", "hail"),
    "melting": ("wet snow",),
}
MODEL_GROUPS = {"rain": "qr", "ice": "qi", "snow": "qs", "graupel": "qg"}
COLORS = {"rain": "#1f77b4", "ice": "#9467bd", "snow": "#17becf", "graupel": "#d62728",
          "melting": "#7f7f7f"}
MAX_HEIGHT_KM = 14.0
SONDES = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ARM/radiosond_data/maosondewnpnM1.b1")
MIN_POINTS = 10_000     # a layer thinner than this is too noisy to plot
TICK_FS, AXIS_FS, HEADER_FS, LETTER_FS = 15, 17, 18, 14


def freezing_level_km(iop: int) -> float:
    """Mean 0 degC height of the ARM sondes launched inside the IOP."""
    from netCDF4 import Dataset

    start, end = GOAMAZON_IOPS[iop]
    levels = []
    for path in sorted(SONDES.glob("maosondewnpnM1.b1.*.cdf")):
        token = path.name.split(".")[2]
        if not start.strftime("%Y%m%d") <= token <= end.strftime("%Y%m%d"):
            continue
        try:
            with Dataset(path) as data:
                alt = np.asarray(data["alt"][:], float) / 1000.0
                temp = np.asarray(data["tdry"][:], float)
        except Exception:  # noqa: BLE001 - a broken sonde is simply skipped
            continue
        good = np.isfinite(alt) & np.isfinite(temp) & (temp > -120.0) & (temp < 60.0)
        if good.sum() < 50:
            continue
        alt, temp = alt[good], temp[good]
        order = np.argsort(alt)
        alt, temp = alt[order], temp[order]
        crossing = np.flatnonzero((temp[:-1] > 0.0) & (temp[1:] <= 0.0))
        if crossing.size:
            i = int(crossing[0])
            levels.append(alt[i] + (0.0 - temp[i]) * (alt[i + 1] - alt[i]) / (temp[i + 1] - temp[i]))
    return float(np.mean(levels)) if levels else float("nan")


def load_radar(work: Path, iop: int) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    with np.load(work / f"xband_hid_iop{iop}.npz", allow_pickle=True) as cache:
        counts = np.asarray(cache["counts"], float)
        edges = np.asarray(cache["height_edges_km"], float)
        names = [str(s) for s in cache["classes"]]
    totals = counts.sum(axis=1)
    groups = {}
    for group, members in RADAR_GROUPS.items():
        columns = [names.index(m) for m in members if m in names]
        groups[group] = counts[:, columns].sum(axis=1)
    return edges, groups, totals


def load_model(work: Path, iop: int, label: str):
    with np.load(work / f"model_species_{label}_iop{iop}.npz", allow_pickle=True) as cache:
        counts = np.asarray(cache["counts"], float)
        edges = np.asarray(cache["height_edges_km"], float)
        species = [str(s) for s in cache["species"]]
    totals = counts.sum(axis=1)
    groups = {g: counts[:, species.index(s)] for g, s in MODEL_GROUPS.items()}
    return edges, groups, totals


def draw(panels: list[tuple[str, dict[str, np.ndarray], np.ndarray]], edges, png: Path,
         dpi: int, iop: int, freezing_km: float) -> None:
    mid = 0.5 * (edges[1:] + edges[:-1])
    keep = mid <= MAX_HEIGHT_KM
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 8.4), sharey=True)
    for index, (ax, (title, groups, totals)) in enumerate(zip(axes, panels)):
        thin = totals < MIN_POINTS
        for group, values in groups.items():
            with np.errstate(invalid="ignore", divide="ignore"):
                share = 100.0 * values / np.where(totals > 0, totals, np.nan)
            share = np.where(thin, np.nan, share)
            ax.plot(share[keep], mid[keep], color=COLORS[group], lw=2.4,
                    ls="--" if group == "melting" else "-", label=group)
        ax.set_title(title, fontsize=HEADER_FS, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, MAX_HEIGHT_KM)
        ax.set_xlabel("share of detectable points (%)", fontsize=AXIS_FS)
        ax.grid(alpha=0.3, lw=0.6)
        ax.tick_params(labelsize=TICK_FS)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        if np.isfinite(freezing_km):
            ax.axhline(freezing_km, color="black", lw=1.4, ls=":", zorder=5,
                       label="0 °C (sondes)" if index == 0 else None)
        ax.text(0.03, 0.985, f"({'abcde'[index]})", transform=ax.transAxes, va="top",
                fontsize=LETTER_FS, fontweight="bold")
        if index == 0:
            ax.set_ylabel("height (km)", fontsize=AXIS_FS)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(0.5, -0.005))
    start, end = GOAMAZON_IOPS[iop]
    fig.suptitle(
        "Hydrometeor species by height, X-band radar against the runs\n"
        f"GoAmazon IOP{iop}, {start:%-d %b} - {end:%-d %b %Y}", fontsize=19,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.14, wspace=0.08)
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_txt(txt: Path, panels, edges, iop: int) -> None:
    mid = 0.5 * (edges[1:] + edges[:-1])
    start, end = GOAMAZON_IOPS[iop]
    lines = [
        f"Hydrometeor species by height, GoAmazon IOP{iop} ({start} to {end}).",
        "Radar: CHUVA X-band dual-pol at T3, CSU fuzzy-logic classification (Dolan & "
        "Rutledge 2009) with X-band membership functions and ARM radiosonde temperatures; "
        "gates with RhoHV <= 0.8, Z <= 0 dBZ or range > 60 km dropped.",
        "Model: species holding most of the precipitating mass (qr, qi, qs, qg) in cells "
        "with simulated reflectivity above the same 0 dBZ floor, inside the radar domain.",
        f"Dotted line: mean 0 degC height of the ARM sondes inside the IOP.",
        "Both are normalised per 0.5 km layer. Layers with fewer than "
        f"{MIN_POINTS:,} detectable points are left out.",
        "The radar labels one dominant species per gate, so a species confined to narrow "
        "convective cores can hold much of the mass while labelling few gates.",
        "",
    ]
    for title, groups, totals in panels:
        lines += [f"{title}", "height_km,points," + ",".join(groups) ]
        for j, z in enumerate(mid):
            if totals[j] < MIN_POINTS or z > MAX_HEIGHT_KM:
                continue
            shares = ",".join(f"{100.0 * groups[g][j] / totals[j]:.3f}" for g in groups)
            lines.append(f"{z:.2f},{int(totals[j])},{shares}")
        lines.append("")
    txt.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=CACHE_DIRS["microphysics"] / "xband")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--iop", type=int, default=1)
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_paper_hydrometeor_species_vs_radar", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    edges, radar_groups, radar_totals = load_radar(args.work_dir, args.iop)
    freezing_km = freezing_level_km(args.iop)
    panels = [("X-band radar", radar_groups, radar_totals)]
    for label in RUNS:
        _, groups, totals = load_model(args.work_dir, args.iop, label)
        panels.append((label, groups, totals))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / f"{FIGURE_STEM}_450dpi.png"
    txt = args.output_dir / f"{FIGURE_STEM}_data.txt"
    draw(panels, edges, png, args.dpi, args.iop, freezing_km)
    print(f"mean sonde 0 degC level: {freezing_km:.2f} km", flush=True)
    write_txt(txt, panels, edges, args.iop)
    print(f"[saved] {png}\n[saved] {txt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
