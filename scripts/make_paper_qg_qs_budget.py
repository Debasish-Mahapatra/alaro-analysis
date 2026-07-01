#!/usr/bin/env python3
"""Snow + graupel (QS + QG) budget diagnostic, mirroring the CT plot.

Three panels of pairwise differences (G1M-C1M, G2M-G1M, G2M-C1M) of the
QS+QG mass tendency (g/kg/day), split into:
  * microphysics (resolved scheme)  = sum of the *-rs DDH budget blocks
  * convection (3MT scheme)         = sum of the *-cv DDH budget blocks
summed over the snow (QS) and graupel (QG) budgets, with the 0 C isotherm band.

Reads the already-extracted DDH block .dta files in lead0024_VZ (control is the
2-ice scheme: it has QS but no QG, so its graupel contribution is zero). Nothing
is recomputed from raw DDH.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VZ_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/lead0024_VZ")
TEMP_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data")
OUTPUT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/8.2 snow and graupel budget")
LEAD = "0024"
MAX_HEIGHT_KM = 20.0
FREEZING_K = 273.15
BUDGETS = ("QS", "QG")

RUNS = {
    "control": {"label": "C1M", "dir": "control"},
    "graupel": {"label": "G1M", "dir": "graupel"},
    "2mom": {"label": "G2M", "dir": "2mom"},
}
PAIRS = [("graupel", "control"), ("2mom", "graupel"), ("2mom", "control")]
COMPONENTS = {
    "microphysics": {
        "blocks": ("evap-rs", "auto-rs", "prec-rs"),
        "label": "Microphysics scheme",
        "color": "#7570b3",
        "linestyle": "-",
    },
    "convection": {
        "blocks": ("evap-cv", "auto-cv", "prec-cv"),
        "label": "Convection scheme 3MT",
        "color": "#d95f02",
        "linestyle": "--",
    },
}


def pair_label(a: str, b: str) -> str:
    return f"{RUNS[a]['label']} − {RUNS[b]['label']}"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QS+QG budget pairwise-difference panels.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--vz-root", type=Path, default=VZ_ROOT)
    parser.add_argument("--temp-root", type=Path, default=TEMP_ROOT)
    parser.add_argument("--max-height-km", type=float, default=MAX_HEIGHT_KM)
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args(argv)


def read_dta(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(str(path))
    return data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)


def average_block(vz_root: Path, run: str, budget: str, block: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Mean profile (over all DDH dates) of one budget block, or None if absent."""
    root = vz_root / str(RUNS[run]["dir"])
    files = sorted(root.glob(f"DDH*/{budget}/{budget}.DHFDLABOF+{LEAD}.{block}.dta"))
    if not files:
        return None
    profiles: List[np.ndarray] = []
    height: Optional[np.ndarray] = None
    for path in files:
        z, values = read_dta(path)
        if height is None:
            height = z
        profiles.append(values)
    assert height is not None
    order = np.argsort(height)
    return height[order], np.nanmean(np.vstack(profiles), axis=0)[order]


def component_profile(vz_root: Path, run: str, component: str) -> Tuple[np.ndarray, np.ndarray]:
    """QS+QG tendency from one scheme (sum of its blocks over both budgets)."""
    blocks = COMPONENTS[component]["blocks"]
    height: Optional[np.ndarray] = None
    total: Optional[np.ndarray] = None
    for budget in BUDGETS:
        for block in blocks:
            res = average_block(vz_root, run, budget, block)
            if res is None:
                continue  # e.g. QG absent for control, or auto-cv absent for QG
            z, prof = res
            if height is None:
                height, total = z, prof.copy()
            else:
                total = total + np.interp(height, z, prof, left=np.nan, right=np.nan)
    if height is None:
        raise FileNotFoundError(f"No {component} blocks found for {run}")
    return height, total


def interpolate_to(z_src: np.ndarray, prof: np.ndarray, z_tgt: np.ndarray) -> np.ndarray:
    valid = np.isfinite(z_src) & np.isfinite(prof)
    if np.sum(valid) < 2:
        return np.full(z_tgt.shape, np.nan)
    z, v = z_src[valid], prof[valid]
    order = np.argsort(z)
    z, v = z[order], v[order]
    uniq = np.concatenate(([True], np.diff(z) > 0.0))
    return np.interp(z_tgt, z[uniq], v[uniq], left=np.nan, right=np.nan)


def load_profiles(vz_root: Path, max_height_km: float):
    raw: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for run in RUNS:
        for comp in COMPONENTS:
            raw[(run, comp)] = component_profile(vz_root, run, comp)

    ctrl_height, _ = raw[("control", "microphysics")]
    height = np.sort(ctrl_height[np.isfinite(ctrl_height)])
    height = height[(height >= 0.0) & (height <= max_height_km)]

    interp: Dict[Tuple[str, str], np.ndarray] = {}
    for (run, comp), (z, prof) in raw.items():
        interp[(run, comp)] = interpolate_to(z, prof, height)
    return height, interp


def pair_diff(interp, a: str, b: str, comp: str) -> np.ndarray:
    return interp[(a, comp)] - interp[(b, comp)]


# --- 0 C isotherm band (reused from the CT diagnostic) ---
def load_temperature_cache(temp_root: Path, exp: str) -> np.ndarray:
    path = temp_root / "temperature" / "2years" / f"{exp}_full-domain_diurnal_profile.npz"
    with np.load(path) as d:
        return np.asarray(d["mean"], dtype=np.float64)


def load_height_axis(temp_root: Path) -> np.ndarray:
    path = temp_root / "geopotential" / "2years" / "control_full-domain_height_profile_first.npz"
    with np.load(path) as d:
        return np.asarray(d["height_m"], dtype=np.float64) / 1000.0


def freezing_line(height_km: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    n = min(height_km.size, temperature.shape[0])
    y = np.asarray(height_km[:n]); temp = np.asarray(temperature[:n, :])
    order = np.argsort(y); y = y[order]; temp = temp[order, :]
    out = np.full(temp.shape[1], np.nan)
    for h in range(temp.shape[1]):
        col = temp[:, h]; m = np.isfinite(y) & np.isfinite(col)
        yy, tt = y[m], col[m]
        for i in range(yy.size - 1):
            if (tt[i] - FREEZING_K) * (tt[i + 1] - FREEZING_K) < 0.0:
                f = (FREEZING_K - tt[i]) / (tt[i + 1] - tt[i])
                out[h] = yy[i] + f * (yy[i + 1] - yy[i]); break
    return out


def freezing_lines(temp_root: Path) -> Dict[str, np.ndarray]:
    axis = load_height_axis(temp_root)
    return {run: freezing_line(axis, load_temperature_cache(temp_root, run)) for run in RUNS}


def plot_panels(output_path: Path, height, interp, lines, dpi) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.2), sharey=True)
    all_diffs = np.concatenate(
        [pair_diff(interp, a, b, comp) for a, b in PAIRS for comp in COMPONENTS]
    )
    xlim = float(np.nanmax(np.abs(all_diffs))) * 1.05
    letters = ["(a)", "(b)", "(c)"]
    for i, (ax, (a, b)) in enumerate(zip(axes, PAIRS)):
        for comp, cfg in COMPONENTS.items():
            ax.plot(pair_diff(interp, a, b, comp), height, color=str(cfg["color"]),
                    linestyle=str(cfg["linestyle"]), linewidth=2.8, label=str(cfg["label"]))
        fl = np.concatenate([lines[a], lines[b]])
        ax.axhspan(float(np.nanmin(fl)), float(np.nanmax(fl)), color="0.85", alpha=0.55, linewidth=0)
        ax.axhline(float(np.nanmean(fl)), color="black", linestyle=":", linewidth=2.0,
                   label="0 °C isotherm mean")
        ax.axvline(0.0, color="0.35", linewidth=1.0)
        ax.set_xlim(-xlim, xlim)
        ax.set_xlabel("QG + QS tendency (g kg$^{-1}$ day$^{-1}$)", fontsize=12)
        ax.set_title(pair_label(a, b), fontsize=15, fontweight="bold")
        ax.text(0.96, 0.97, letters[i], transform=ax.transAxes, ha="right", va="top",
                fontsize=15, fontweight="bold")
        ax.grid(True, alpha=0.28)
    axes[0].set_ylabel("Altitude (km)", fontsize=13)
    axes[0].set_ylim(0.0, float(np.nanmax(height)))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=11,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.04))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_text(path: Path, height, interp, lines) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("Snow + graupel (QS+QG) budget diagnostic (pairwise differences)\n")
        f.write("Units: g/kg/day. microphysics = sum(*-rs blocks); convection = sum(*-cv blocks),\n")
        f.write("each summed over the QS and QG DDH budgets (control 2-ice has QS only).\n")
        f.write("Each panel = minuend - subtrahend.\n\n")
        for run, line in lines.items():
            f.write(f"{RUNS[run]['label']}_mean_freezing_level_km,{float(np.nanmean(line)):.6g}\n")
        for a, b in PAIRS:
            f.write(f"\n=== {pair_label(a, b)} ===\n")
            f.write("height_km,microphysics_gkgday,convection_gkgday\n")
            micro = pair_diff(interp, a, b, "microphysics")
            conv = pair_diff(interp, a, b, "convection")
            for z, m, c in zip(height, micro, conv):
                f.write(f"{z:.6g},{m:.6g},{c:.6g}\n")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    out = args.output_dir.resolve()
    height, interp = load_profiles(args.vz_root.resolve(), args.max_height_km)
    lines = freezing_lines(args.temp_root.resolve())

    figure = out / "8.2 snow and graupel budget_450dpi.png"
    text_path = out / "8.2 snow and graupel budget_data.txt"
    plot_panels(figure, height, interp, lines, args.dpi)
    write_text(text_path, height, interp, lines)
    print(f"[saved] {figure}")
    print(f"[saved] {text_path}")
    for a, b in PAIRS:
        micro = pair_diff(interp, a, b, "microphysics")
        conv = pair_diff(interp, a, b, "convection")
        fl = float(np.nanmean(np.concatenate([lines[a], lines[b]])))
        print(f"{pair_label(a, b)}: |micro| peak {np.nanmax(np.abs(micro)):.3g}, "
              f"|conv| peak {np.nanmax(np.abs(conv)):.3g} g/kg/day; 0C~{fl:.2f} km")


if __name__ == "__main__":
    main()
