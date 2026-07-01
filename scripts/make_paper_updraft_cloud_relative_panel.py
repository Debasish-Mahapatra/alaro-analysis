#!/usr/bin/env python3
"""Relative-bias 9-panel figure: updraft extent, updraft mass flux, cloud cover.

Studies hydrometeor.py's ``plot_three_panels`` and reuses its colour scheme, but
(1) stacks the three quantities into a single 3x3 figure and (2) turns the
absolute anomaly columns into RELATIVE (percentage) differences.

Layout (rows = quantity, columns = comparison)::

    row 0  Updraft extent      |  C1M absolute  |  (G1M-C1M)/C1M  |  (G2M-G1M)/G1M
    row 1  Updraft mass flux   |  C1M absolute  |  (G1M-C1M)/C1M  |  (G2M-G1M)/G1M
    row 2  Cloud cover         |  C1M absolute  |  (G1M-C1M)/C1M  |  (G2M-G1M)/G1M

Colour scheme (identical to the original single-quantity panels):
  * absolute column : cmaps.WhiteBlueGreenYellowRed, linear Normalize(p2, p98)
  * relative columns: RdBu_r diverging, TwoSlopeNorm centred on 0, robust scale
  * grey #d3d3d3 facecolor, dashed black freezing-level overlay, %g colourbars

Each height-hour cell is a diurnal mean over the full 2-year run; the relative
difference is masked where the reference field is below 5 % of its 98th
percentile (otherwise dividing by a near-zero updraft/cloud value explodes).

Reads the same cached diurnal profiles the original panels use.
"""
from __future__ import annotations

from pathlib import Path

import cmaps
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.models import VerticalAxis
from alaro_analysis.common.vertical import centers_to_edges, compute_freezing_line_km
from alaro_analysis.data.cache import load_diurnal_profile_cache
from alaro_analysis.plotting.scales import robust_anomaly_scale

PROC = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data")
# Diurnal means recomputed directly from masked-netcdf by
# build_updraft_cloud_diurnal_from_netcdf.py (not the pre-existing caches).
NETCDF_DIR = PROC / "paper5_from_netcdf"
OUTPUT_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/5. updraft and cloud cover"
)
FIGURE_NAME = "5. updraft and cloud cover_450dpi.png"
DATA_NAME = "5. updraft and cloud cover_data.txt"
PERIOD = "2years"
STAG = "full-domain"
MAX_HEIGHT_KM = 16.0
REF_FRACTION = 0.05  # mask relative bias where reference < 5% of its p98
REL_SCALE_PCT = 70.0  # fixed +/- relative-bias colour scale (%) for every row

# Font sizes (enlarged for the paper figure)
TICK_FS = 19       # x/y tick labels
AXIS_FS = 21       # x/y axis labels
HEADER_FS = 18     # panel titles
LETTER_FS = 16     # (a)-(i) panel tags
CBAR_LABEL_FS = 19
CBAR_TICK_FS = 16
LEGEND_FS = 14

EXPS = ("control", "graupel", "2mom")
PANEL_EXPS = ("control", "graupel", "2mom")  # freezing-line source per column

ROWS = [
    {"key": "extent", "row_label": "Updraft extent", "abs_label": "Fraction", "unit_interval": True},
    {"key": "flux", "row_label": "Updraft mass flux", "abs_label": "kg m$^{-2}$ s$^{-1}$", "unit_interval": False},
    {"key": "cloud", "row_label": "Cloud cover", "abs_label": "Fraction", "unit_interval": True},
]

# Old caches kept only for an optional recompute-vs-cache cross-check.
OLD_CACHE_DIRS = {
    "extent": PROC / "total_updraft_extent" / PERIOD,
    "flux": PROC / "total_updraft_flux" / PERIOD,
    "cloud": PROC / "data" / "cloud_fracti" / PERIOD,
}


def load_diurnal(key: str, exp: str) -> np.ndarray:
    with np.load(NETCDF_DIR / f"{exp}_{key}.npz") as d:
        return np.asarray(d["mean"], dtype=np.float64)


def load_height_km(exp: str) -> np.ndarray:
    with np.load(NETCDF_DIR / f"{exp}_height.npz") as d:
        return np.asarray(d["height_m"], dtype=np.float64) / 1000.0


def load_temperature(exp: str) -> np.ndarray | None:
    path = NETCDF_DIR / f"{exp}_temp.npz"
    if not path.exists():
        return None
    with np.load(path) as d:
        return np.asarray(d["mean"], dtype=np.float64)


def crosscheck_against_cache() -> None:
    """Print max |recomputed - cached| so the netcdf recompute is verifiable."""
    for key, old_dir in OLD_CACHE_DIRS.items():
        for exp in EXPS:
            old_path = old_dir / f"{exp}_{STAG}_diurnal_profile.npz"
            new_path = NETCDF_DIR / f"{exp}_{key}.npz"
            if not old_path.exists() or not new_path.exists():
                continue
            old, _, _, _ = load_diurnal_profile_cache(str(old_path))
            with np.load(new_path) as d:
                new = np.asarray(d["mean"], dtype=np.float64)
            old = np.asarray(old, dtype=np.float64)
            n = min(old.shape[0], new.shape[0])
            diff = np.abs(new[:n] - old[:n])
            denom = float(np.nanmax(np.abs(old[:n]))) or 1.0
            print(
                f"  cross-check {key:7s} {exp:8s}: max|new-old|={np.nanmax(diff):.3e} "
                f"({100 * np.nanmax(diff) / denom:.3f}% of field max)",
                flush=True,
            )


def relative_bias(
    numerator: np.ndarray, reference: np.ndarray, frac: float = REF_FRACTION
) -> np.ndarray:
    """100 * (model - ref) / ref, masked where ref is negligibly small."""
    ref = np.asarray(reference, dtype=np.float64)
    positive = ref[np.isfinite(ref) & (ref > 0.0)]
    threshold = frac * float(np.percentile(positive, 98.0)) if positive.size else np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = 100.0 * numerator / np.where(ref != 0.0, ref, np.nan)
    return np.where(np.isfinite(ref) & (ref > threshold), rel, np.nan)


def main() -> None:
    print("Cross-check: netcdf recompute vs pre-existing cache", flush=True)
    crosscheck_against_cache()

    # Vertical axis from control geopotential: sort ascending, crop to MAX_HEIGHT.
    z = load_height_km("control")
    order = np.argsort(z)
    z_sorted = z[order]
    keep = np.isfinite(z_sorted) & (z_sorted >= 0.0) & (z_sorted <= MAX_HEIGHT_KM)
    y = z_sorted[keep]
    y_edges = centers_to_edges(y)
    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    axis_ctrl = VerticalAxis(values=z, label="Height (km)", is_height_km=True)

    freezing = {}
    for exp in EXPS:
        temp = load_temperature(exp)
        freezing[exp] = compute_freezing_line_km(axis_ctrl, temp) if temp is not None else None

    def crop(arr: np.ndarray) -> np.ndarray:
        return arr[order, :][keep, :]

    # Copy plot_three_panels aesthetics: constrained_layout does all the spacing,
    # horizontal colorbars added below each row with the same fraction/pad. Three
    # rows == three stacked (20x7) figures, so the panel aspect matches the original.
    fig, axes = plt.subplots(3, 3, figsize=(20.0, 21.0), constrained_layout=True)

    letters = iter("abcdefghi")
    txt_rows: list[dict] = []

    for r, row in enumerate(ROWS):
        ctrl = load_diurnal(row["key"], "control")
        g1 = load_diurnal(row["key"], "graupel")
        g2 = load_diurnal(row["key"], "2mom")

        abs_ctrl = np.ma.masked_invalid(crop(ctrl))
        rel1 = np.ma.masked_invalid(crop(relative_bias(g1 - ctrl, ctrl)))
        rel2 = np.ma.masked_invalid(crop(relative_bias(g2 - g1, g1)))

        valid = abs_ctrl.compressed()
        vmin = float(np.percentile(valid, 2.0)) if valid.size else 0.0
        vmax = float(np.percentile(valid, 98.0)) if valid.size else 1.0
        vmin = max(0.0, vmin)
        if row["unit_interval"]:
            vmax = min(1.0, vmax)
        if vmax <= vmin:
            vmax = vmin + 1e-9
        abs_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        anom_scale = robust_anomaly_scale(rel1.filled(np.nan), rel2.filled(np.nan))
        if not np.isfinite(anom_scale) or anom_scale <= 0:
            anom_scale = 1.0
        diff_norm = mcolors.TwoSlopeNorm(vmin=-anom_scale, vcenter=0.0, vmax=anom_scale)

        txt_rows.append(
            {
                "label": row["row_label"],
                "abs_label": row["abs_label"],
                "abs": np.ma.filled(abs_ctrl, np.nan),
                "rel1": np.ma.filled(rel1, np.nan),
                "rel2": np.ma.filled(rel2, np.nan),
                "vmin": vmin,
                "vmax": vmax,
                "anom_scale": anom_scale,
            }
        )

        ax0, ax1, ax2 = axes[r, 0], axes[r, 1], axes[r, 2]
        fields = (abs_ctrl, rel1, rel2)
        cmaps_used = (cmaps.WhiteBlueGreenYellowRed, "RdBu_r", "RdBu_r")
        norms = (abs_norm, diff_norm, diff_norm)
        titles = (
            f"{EXPERIMENT_LABELS['control']} ({row['row_label']}, absolute)",
            f"({EXPERIMENT_LABELS['graupel']} − {EXPERIMENT_LABELS['control']}) / {EXPERIMENT_LABELS['control']}",
            f"({EXPERIMENT_LABELS['2mom']} − {EXPERIMENT_LABELS['graupel']}) / {EXPERIMENT_LABELS['graupel']}",
        )
        pcms = []
        for ci, ax in enumerate((ax0, ax1, ax2)):
            ax.set_facecolor("#d3d3d3")
            pcms.append(
                ax.pcolormesh(
                    hour_edges, y_edges, fields[ci],
                    cmap=cmaps_used[ci], norm=norms[ci], shading="auto",
                )
            )
            ax.set_title(titles[ci], fontsize=HEADER_FS, fontweight="bold")
            ax.set_xlabel("Hour (Amazon UTC-4)", fontsize=AXIS_FS)
            ax.set_ylabel("Height (km)", fontsize=AXIS_FS)
            ax.set_xticks(np.arange(0, 24, 6))
            ax.set_xlim(-0.5, 23.5)
            ax.set_ylim(0.0, MAX_HEIGHT_KM)
            ax.tick_params(axis="both", labelsize=TICK_FS)
            ax.text(
                0.02, 0.98, f"({next(letters)})",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=LETTER_FS, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2.0},
            )
            line = freezing.get(PANEL_EXPS[ci])
            if line is not None and np.isfinite(line).any():
                ax.plot(
                    np.arange(24), line, color="black", lw=1.8, ls="--",
                    zorder=10, label="Freezing level",
                )
                if r == 0 and ci == 0:
                    ax.legend(loc="upper right", fontsize=LEGEND_FS, framealpha=0.9)

        cbar_abs = fig.colorbar(
            pcms[0], ax=ax0, orientation="horizontal", fraction=0.08, pad=0.04
        )
        cbar_abs.set_label(row["abs_label"], fontsize=CBAR_LABEL_FS)
        cbar_abs.ax.tick_params(labelsize=CBAR_TICK_FS)
        cbar_abs.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

        cbar_rel = fig.colorbar(
            pcms[1], ax=[ax1, ax2], orientation="horizontal", fraction=0.08, pad=0.04
        )
        cbar_rel.set_label("Relative difference (%)", fontsize=CBAR_LABEL_FS)
        cbar_rel.ax.tick_params(labelsize=CBAR_TICK_FS)
        cbar_rel.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / FIGURE_NAME
    fig.savefig(out, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")

    write_data_txt(OUTPUT_DIR / DATA_NAME, y, txt_rows, out)
    print(f"[saved] {OUTPUT_DIR / DATA_NAME}")


def write_data_txt(path: Path, y: np.ndarray, txt_rows: list[dict], figure_path: Path) -> None:
    hours = np.arange(24)
    with path.open("w", encoding="utf-8") as fh:
        title = "Updraft extent / updraft mass flux / cloud cover - relative-bias panels"
        fh.write(f"{title}\n{'=' * len(title)}\n")
        fh.write(f"Source figure: {figure_path}\n")
        fh.write("Diurnal-mean height-hour sections, full 2-year run, full-domain mean.\n")
        fh.write("Columns per quantity: C1M absolute; (G1M-C1M)/C1M [%]; (G2M-G1M)/G1M [%].\n")
        fh.write(f"Relative bias masked where reference < {REF_FRACTION:g} x its 98th percentile.\n")
        fh.write(f"Height cropped to {MAX_HEIGHT_KM:g} km. Hour is Amazon local time (UTC-4).\n\n")
        for rec in txt_rows:
            fh.write(f"## {rec['label']}\n")
            fh.write(
                f"absolute C1M limits: vmin={rec['vmin']:.6g} vmax={rec['vmax']:.6g} "
                f"[{rec['abs_label']}]; relative colour scale: +/-{rec['anom_scale']:.6g} %\n"
            )
            for key, blabel in (
                ("abs", f"C1M absolute [{rec['abs_label']}]"),
                ("rel1", "(G1M-C1M)/C1M [%]"),
                ("rel2", "(G2M-G1M)/G1M [%]"),
            ):
                fh.write(f"-- {blabel} -- rows=height_km, cols=hour 0..23\n")
                fh.write("height_km," + ",".join(f"h{h:02d}" for h in hours) + "\n")
                field = rec[key]
                for i, hk in enumerate(y):
                    vals = ",".join(
                        "nan" if not np.isfinite(field[i, j]) else f"{field[i, j]:.6g}"
                        for j in range(field.shape[1])
                    )
                    fh.write(f"{hk:.4f},{vals}\n")
                fh.write("\n")


if __name__ == "__main__":
    main()
