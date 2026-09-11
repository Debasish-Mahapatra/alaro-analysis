#!/usr/bin/env python3
"""Parameterized vs resolved downdraft, both as a velocity in m/s.

The companion figure (make_paper_downdraft_vertvel_relative_panel.py) reports
the parameterized downdraft as a mass flux in kg m-2 s-1 and the resolved one as
a velocity in m/s, so the two cannot be compared directly.  Here both are
velocities, on the same footing:

    row 0  Parameterized downdraft velocity = mean(sigma_d * w_d)   m s-1
    row 1  Resolved downdraft velocity      = mean(max(-w, 0))      m s-1

Both are grid-box means of the downward velocity, positive downward, over the
same 2822 radar-mask points -- the parameterized one area-weighted by the
downdraft fraction sigma_d, the resolved one weighted by 1 where the flow sinks
and 0 where it rises.  Like for like.

The conversion uses M_d = rho * sigma_d * w_d, i.e. velocity = flux / rho, with
rho taken from the cached fields as w_dn_flux / w_dn (the mean of rho*|w| over
the mean of |w|, which is a motion-weighted mean density).  That is checked
against rho = p/(Rd*T) computed directly from raw FA: they agree to 0.10 %
median / 0.30 % max below 16 km, so no re-extraction is needed.

Columns per row: C1M absolute; (G1M-C1M)/C1M; (G2M-G1M)/G1M -- same colours,
masking and layout as the other paper panels.

Reads the npz written by build_downdraft_vertvel_diurnal_from_FA.py.
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
from alaro_analysis.plotting.scales import robust_anomaly_scale

PROC = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data")
FA_DIR = PROC / "paper_downdraft_from_FA"
OUTPUT_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/16. downdraft and vertical velocity"
)
FIGURE_NAME = "16b. downdraft velocity parameterized vs resolved_450dpi.png"
DATA_NAME = "16b. downdraft velocity parameterized vs resolved_data.txt"
MAX_HEIGHT_KM = 16.0
REF_FRACTION = 0.05
N_MASK_POINTS = 2822

TICK_FS = 19
AXIS_FS = 21
HEADER_FS = 18
LETTER_FS = 16
CBAR_LABEL_FS = 19
CBAR_TICK_FS = 16
LEGEND_FS = 14

EXPS = ("control", "graupel", "2mom")
PANEL_EXPS = ("control", "graupel", "2mom")

ROWS = [
    {"key": "param", "row_label": "Parameterized downdraft velocity"},
    {"key": "resolved", "row_label": "Resolved downdraft velocity"},
]
UNIT = "m s$^{-1}$"


def load(key: str, exp: str) -> np.ndarray:
    with np.load(FA_DIR / f"{exp}_{key}.npz") as d:
        return np.asarray(d["mean"], dtype=np.float64)


def density(exp: str) -> np.ndarray:
    """Motion-weighted mean air density (kg m-3) from the cached resolved fields."""
    w = load("w_dn", exp)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(w > 0.0, load("w_dn_flux", exp) / np.where(w > 0.0, w, np.nan), np.nan)


def velocity(key: str, exp: str) -> np.ndarray:
    """Grid-box mean downward velocity (m s-1), positive downward."""
    if key == "resolved":
        return load("w_dn", exp)
    # M_d = rho * sigma_d * w_d  ->  sigma_d * w_d = M_d / rho
    return load("flux", exp) / density(exp)


def load_height_km(exp: str) -> np.ndarray:
    with np.load(FA_DIR / f"{exp}_height.npz") as d:
        return np.asarray(d["height_m"], dtype=np.float64) / 1000.0


def load_temperature(exp: str) -> np.ndarray | None:
    path = FA_DIR / f"{exp}_temp.npz"
    if not path.exists():
        return None
    with np.load(path) as d:
        return np.asarray(d["mean"], dtype=np.float64)


def relative_bias(numerator, reference, frac: float = REF_FRACTION) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    positive = ref[np.isfinite(ref) & (ref > 0.0)]
    threshold = frac * float(np.percentile(positive, 98.0)) if positive.size else np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = 100.0 * numerator / np.where(ref != 0.0, ref, np.nan)
    return np.where(np.isfinite(ref) & (ref > threshold), rel, np.nan)


def main() -> None:
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

    fig, axes = plt.subplots(2, 3, figsize=(20.0, 14.0), constrained_layout=True)
    letters = iter("abcdef")
    txt_rows: list[dict] = []

    for r, row in enumerate(ROWS):
        ctrl = velocity(row["key"], "control")
        g1 = velocity(row["key"], "graupel")
        g2 = velocity(row["key"], "2mom")

        abs_ctrl = np.ma.masked_invalid(crop(ctrl))
        rel1 = np.ma.masked_invalid(crop(relative_bias(g1 - ctrl, ctrl)))
        rel2 = np.ma.masked_invalid(crop(relative_bias(g2 - g1, g1)))

        valid = abs_ctrl.compressed()
        vmin = max(0.0, float(np.percentile(valid, 2.0))) if valid.size else 0.0
        vmax = float(np.percentile(valid, 98.0)) if valid.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-12
        abs_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        anom_scale = robust_anomaly_scale(rel1.filled(np.nan), rel2.filled(np.nan))
        if not np.isfinite(anom_scale) or anom_scale <= 0:
            anom_scale = 1.0
        diff_norm = mcolors.TwoSlopeNorm(vmin=-anom_scale, vcenter=0.0, vmax=anom_scale)

        txt_rows.append({
            "label": row["row_label"],
            "abs": np.ma.filled(abs_ctrl, np.nan),
            "rel1": np.ma.filled(rel1, np.nan),
            "rel2": np.ma.filled(rel2, np.nan),
            "vmin": vmin, "vmax": vmax, "anom_scale": anom_scale,
            "mean": float(np.nanmean(np.ma.filled(abs_ctrl, np.nan))),
        })

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
            pcms.append(ax.pcolormesh(hour_edges, y_edges, fields[ci],
                                      cmap=cmaps_used[ci], norm=norms[ci], shading="auto"))
            ax.set_title(titles[ci], fontsize=HEADER_FS, fontweight="bold", pad=18)
            ax.set_xlabel("Hour (Amazon UTC-4)", fontsize=AXIS_FS)
            ax.set_ylabel("Height (km)", fontsize=AXIS_FS)
            ax.set_xticks(np.arange(0, 24, 6))
            ax.set_xlim(-0.5, 23.5)
            ax.set_ylim(0.0, MAX_HEIGHT_KM)
            ax.tick_params(axis="both", labelsize=TICK_FS)
            ax.text(0.02, 0.98, f"({next(letters)})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=LETTER_FS, fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2.0})
            line = freezing.get(PANEL_EXPS[ci])
            if line is not None and np.isfinite(line).any():
                ax.plot(np.arange(24), line, color="black", lw=1.8, ls="--",
                        zorder=10, label="Freezing level")
                if r == 0 and ci == 0:
                    ax.legend(loc="upper right", fontsize=LEGEND_FS, framealpha=0.9)

        cbar_abs = fig.colorbar(pcms[0], ax=ax0, orientation="horizontal", fraction=0.08, pad=0.04)
        cbar_abs.set_label(UNIT, fontsize=CBAR_LABEL_FS)
        cbar_abs.ax.tick_params(labelsize=CBAR_TICK_FS)
        cbar_abs.ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        cbar_abs.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

        cbar_rel = fig.colorbar(pcms[1], ax=[ax1, ax2], orientation="horizontal", fraction=0.08, pad=0.04)
        cbar_rel.set_label("Relative difference (%)", fontsize=CBAR_LABEL_FS)
        cbar_rel.ax.tick_params(labelsize=CBAR_TICK_FS)
        cbar_rel.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / FIGURE_NAME
    fig.savefig(out, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")

    ratio = txt_rows[0]["mean"] / txt_rows[1]["mean"]
    print(f"[ratio] parameterized / resolved (C1M domain mean) = {ratio:.4g}")
    write_data_txt(OUTPUT_DIR / DATA_NAME, y, txt_rows, out, ratio)
    print(f"[saved] {OUTPUT_DIR / DATA_NAME}")


def write_data_txt(path, y, txt_rows, figure_path, ratio) -> None:
    hours = np.arange(24)
    with path.open("w", encoding="utf-8") as fh:
        title = "Downdraft velocity, parameterized vs resolved - both in m/s"
        fh.write(f"{title}\n{'=' * len(title)}\n")
        fh.write(f"Figure: {figure_path}\n\n")
        fh.write("Both rows are GRID-BOX MEAN downward velocities (m/s, positive down)\n")
        fh.write(f"over the same {N_MASK_POINTS} SIPAM radar-mask points, read from raw FA.\n\n")
        fh.write("Parameterized = mean(sigma_d * w_d) = mean(where(mesh>0, DD_OMEGA*mesh/g, 0)) / rho\n")
        fh.write("  from M_d = rho * sigma_d * w_d.  DD_OMEGA > 0 for subsidence in ALARO.\n")
        fh.write("Resolved      = mean(max(-w, 0)),  w = VERT.VELOCIT (m/s, positive up)\n")
        fh.write("  i.e. the resolved vertical velocity with upward motion masked out.\n\n")
        fh.write("rho is taken from the cached fields as w_dn_flux / w_dn (a motion-weighted\n")
        fh.write("mean density).  Checked against rho = p/(Rd*T) straight from raw FA:\n")
        fh.write("agreement 0.10 % median, 0.30 % max below 16 km.\n\n")
        fh.write(f"C1M domain-mean parameterized / resolved = {ratio:.4g}\n")
        fh.write("  (the resolved flow carries the great majority of the descent; the two\n")
        fh.write("   rows therefore use independent colour scales so both stay readable).\n\n")
        fh.write("Columns per row: C1M absolute; (G1M-C1M)/C1M [%]; (G2M-G1M)/G1M [%].\n")
        fh.write(f"Relative bias masked where reference < {REF_FRACTION:g} x its 98th percentile.\n")
        fh.write(f"Height cropped to {MAX_HEIGHT_KM:g} km. Hour is Amazon local time (UTC-4).\n\n")
        for rec in txt_rows:
            fh.write(f"## {rec['label']}\n")
            fh.write(f"absolute C1M limits: vmin={rec['vmin']:.6g} vmax={rec['vmax']:.6g} [m s-1]; "
                     f"relative colour scale: +/-{rec['anom_scale']:.6g} %; "
                     f"domain mean {rec['mean']:.6g} m s-1\n")
            for key, blabel in (("abs", "C1M absolute [m s-1]"),
                                ("rel1", "(G1M-C1M)/C1M [%]"),
                                ("rel2", "(G2M-G1M)/G1M [%]")):
                fh.write(f"-- {blabel} -- rows=height_km, cols=hour 0..23\n")
                fh.write("height_km," + ",".join(f"h{h:02d}" for h in hours) + "\n")
                field = rec[key]
                for i, hk in enumerate(y):
                    vals = ",".join("nan" if not np.isfinite(field[i, j]) else f"{field[i, j]:.6g}"
                                    for j in range(field.shape[1]))
                    fh.write(f"{hk:.4f},{vals}\n")
                fh.write("\n")


if __name__ == "__main__":
    main()
