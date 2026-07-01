#!/usr/bin/env python3
"""Graupel/freezing-level CT budget diagnostic for paper1.

Three panels of pairwise CT-tendency differences (microphysics + 3MT convection)
around the 0 C isotherm: G1M-C1M, G1M-G2M, C1M-G2M.

CT note: C1M is read from the 2-ice CT FBL split (CT3.fbl-2ice); G1M and G2M use
the same 3-ice split (CT.fbl-3ice). All three are already extracted/cached, so
nothing is recomputed here.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAPER_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper")
OUTPUT_DIR = PAPER_ROOT / "8. temperature tendencies"
DDH_CACHE = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/"
    "DDH-0024-only-CT-extracted"
)
TEMP_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data")
FBL_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ddhtoolbox/ddh_budget_lists")
LEAD = "0024"
MAX_HEIGHT_KM = 20.0
FREEZING_K = 273.15

RUNS = {
    "control": {
        "label": "C1M",
        "cache_tag": "lead0024_VZ_CT3_fbl_2ice_micro",
        "experiment_dir": "control",
        "fbl": FBL_ROOT / "alaro" / "CT3.fbl-2ice",
    },
    "graupel": {
        "label": "G1M",
        "cache_tag": "lead0024_VZ_CT_fbl_3ice",
        "experiment_dir": "graupel",
        "fbl": FBL_ROOT / "alaro" / "CT.fbl-3ice",
    },
    "2mom": {
        "label": "G2M",
        "cache_tag": "lead0024_VZ_CT_fbl_3ice",
        "experiment_dir": "2mom",
        "fbl": FBL_ROOT / "alaro" / "CT.fbl-3ice",
    },
}
# pairwise differences to plot, each (minuend, subtrahend) -> minuend - subtrahend
PAIRS = [("graupel", "control"), ("2mom", "graupel"), ("2mom", "control")]
TERMS = {
    "microphysics": {
        "component": "micro-rs",
        "label": "Microphysics scheme",
        "color": "#7570b3",
        "linestyle": "-",
    },
    "convection": {
        "component": "micro-cv",
        "label": "Convection scheme 3MT",
        "color": "#d95f02",
        "linestyle": "--",
    },
}


def pair_label(a: str, b: str) -> str:
    return f"{RUNS[a]['label']} − {RUNS[b]['label']}"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot pairwise CT microphysics/convection differences and 0 C isotherm."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--ddh-cache", type=Path, default=DDH_CACHE)
    parser.add_argument("--temp-root", type=Path, default=TEMP_ROOT)
    parser.add_argument("--max-height-km", type=float, default=MAX_HEIGHT_KM)
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args(argv)


def read_dta(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(str(path))
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected two-column .dta file: {path}")
    return data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)


def component_files(ddh_cache: Path, run: str, component: str) -> List[Path]:
    info = RUNS[run]
    root = ddh_cache / str(info["cache_tag"]) / str(info["experiment_dir"])
    pattern = f"DDH*/CT/CT.DHFDLABOF+{LEAD}.{component}.dta"
    return sorted(root.glob(pattern))


def average_component(ddh_cache: Path, run: str, component: str) -> Tuple[np.ndarray, np.ndarray, int]:
    files = component_files(ddh_cache, run, component)
    if not files:
        raise FileNotFoundError(f"No {component} CT .dta files found for {run}")

    profiles: List[np.ndarray] = []
    height: Optional[np.ndarray] = None
    for path in files:
        z, values = read_dta(path)
        if height is None:
            height = z
        elif z.shape != height.shape:
            raise ValueError(f"Height shape mismatch in {path}")
        profiles.append(values)

    assert height is not None
    stack = np.vstack(profiles)
    order = np.argsort(height)
    return height[order], np.nanmean(stack[:, order], axis=0), len(files)


def interpolate_to(height_src: np.ndarray, profile: np.ndarray, height_target: np.ndarray) -> np.ndarray:
    valid = np.isfinite(height_src) & np.isfinite(profile)
    if np.sum(valid) < 2:
        return np.full(height_target.shape, np.nan, dtype=np.float64)
    z = height_src[valid]
    v = profile[valid]
    order = np.argsort(z)
    z = z[order]
    v = v[order]
    unique = np.concatenate(([True], np.diff(z) > 0.0))
    return np.interp(height_target, z[unique], v[unique], left=np.nan, right=np.nan)


def load_budget_profiles(ddh_cache: Path, max_height_km: float):
    """Return (height, interp{(run,component)->profile}, nfiles{run})."""
    raw: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, int]] = {}
    for run in RUNS:
        for term in TERMS.values():
            component = str(term["component"])
            raw[(run, component)] = average_component(ddh_cache, run, component)

    control_height, _, _ = raw[("control", "micro-rs")]
    height = np.sort(control_height[np.isfinite(control_height)])
    keep = (height >= 0.0) & (height <= max_height_km)
    height = height[keep]

    interp: Dict[Tuple[str, str], np.ndarray] = {}
    nfiles: Dict[str, int] = {}
    for (run, component), (z, prof, n) in raw.items():
        interp[(run, component)] = interpolate_to(z, prof, height)
        nfiles[run] = n
    return height, interp, nfiles


def pair_diff(interp, a: str, b: str, component: str) -> np.ndarray:
    return interp[(a, component)] - interp[(b, component)]


def load_temperature_cache(temp_root: Path, experiment: str) -> np.ndarray:
    path = temp_root / "temperature" / "2years" / f"{experiment}_full-domain_diurnal_profile.npz"
    with np.load(path) as data:
        return np.asarray(data["mean"], dtype=np.float64)


def load_height_axis(temp_root: Path) -> np.ndarray:
    path = temp_root / "geopotential" / "2years" / "control_full-domain_height_profile_first.npz"
    with np.load(path) as data:
        height_km = np.asarray(data["height_m"], dtype=np.float64) / 1000.0
    return height_km


def compute_freezing_line_km(height_km: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    n_levels = min(height_km.size, temperature.shape[0])
    y = np.asarray(height_km[:n_levels], dtype=np.float64)
    temp = np.asarray(temperature[:n_levels, :], dtype=np.float64)
    order = np.argsort(y)
    y = y[order]
    temp = temp[order, :]
    out = np.full(24, np.nan, dtype=np.float64)
    for hour in range(24):
        column = temp[:, hour]
        finite = np.isfinite(y) & np.isfinite(column)
        if np.sum(finite) < 2:
            continue
        yy = y[finite]
        tt = column[finite]
        for idx in range(yy.size - 1):
            t1, t2 = tt[idx], tt[idx + 1]
            y1, y2 = yy[idx], yy[idx + 1]
            if np.isclose(t1, FREEZING_K):
                out[hour] = y1
                break
            if np.isclose(t2, FREEZING_K):
                out[hour] = y2
                break
            if (t1 - FREEZING_K) * (t2 - FREEZING_K) < 0.0 and not np.isclose(t1, t2):
                frac = (FREEZING_K - t1) / (t2 - t1)
                out[hour] = y1 + frac * (y2 - y1)
                break
    return out


def freezing_lines(temp_root: Path) -> Dict[str, np.ndarray]:
    axis = load_height_axis(temp_root)
    lines: Dict[str, np.ndarray] = {}
    for run in RUNS:
        lines[run] = compute_freezing_line_km(axis, load_temperature_cache(temp_root, run))
    return lines


def parse_fbl_blocks(path: Path) -> Dict[str, List[str]]:
    blocks: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("BEGIN BLOCK"):
            current = line.split("BEGIN BLOCK", 1)[1].strip()
            blocks[current] = []
            continue
        if current is None:
            continue
        token = line.split("#", 1)[0].strip().split()
        if token:
            blocks[current].append(token[0])
    return blocks


def peak_cooling(height: np.ndarray, diff: np.ndarray, center: float) -> Tuple[float, float]:
    window = np.isfinite(height) & np.isfinite(diff) & (height >= center - 1.0) & (height <= center + 1.0)
    if not np.any(window):
        window = np.isfinite(height) & np.isfinite(diff)
    idxs = np.flatnonzero(window)
    idx = idxs[int(np.nanargmin(diff[idxs]))]
    return float(height[idx]), float(diff[idx])


def plot_budget_difference(output_path, height, interp, lines, dpi) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.2), sharey=True)
    all_diffs = np.concatenate(
        [pair_diff(interp, a, b, str(term["component"])) for a, b in PAIRS for term in TERMS.values()]
    )
    xlim = float(np.nanmax(np.abs(all_diffs))) * 1.05
    letters = ["(a)", "(b)", "(c)"]
    for i, (ax, (a, b)) in enumerate(zip(axes, PAIRS)):
        for name, term in TERMS.items():
            comp = str(term["component"])
            ax.plot(
                pair_diff(interp, a, b, comp), height,
                color=str(term["color"]), linestyle=str(term["linestyle"]),
                linewidth=2.8, label=str(term["label"]),
            )
        fl = np.concatenate([lines[a], lines[b]])
        ax.axhspan(float(np.nanmin(fl)), float(np.nanmax(fl)), color="0.85", alpha=0.55, linewidth=0)
        ax.axhline(float(np.nanmean(fl)), color="black", linestyle=":", linewidth=2.0,
                   label="0 °C isotherm mean")
        ax.axvline(0.0, color="0.35", linewidth=1.0)
        ax.set_xlim(-xlim, xlim)
        ax.set_xlabel("CT tendency (K day$^{-1}$)", fontsize=12)
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


def write_text(path, height, interp, lines, nfiles) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("Graupel freezing-level CT budget diagnostic (pairwise differences)\n")
        f.write("DDH CT terms are read after ddhb conversion; units are K/day.\n")
        f.write("CT: C1M uses the 2-ice FBL split; G1M and G2M use the 3-ice split.\n")
        f.write("Each panel = (minuend - subtrahend); negative = extra cooling in the minuend.\n\n")
        for run, info in RUNS.items():
            f.write(f"{info['label']} FBL: {info['fbl']}  (days={int(nfiles[run])})\n")
        f.write("\n0 C isotherm (mean over local hours) from 3D temperature data\n")
        for run, line in lines.items():
            f.write(f"{RUNS[run]['label']}_mean_freezing_level_km,{float(np.nanmean(line)):.12g}\n")

        for a, b in PAIRS:
            tag = pair_label(a, b)
            fl = np.concatenate([lines[a], lines[b]])
            center = float(np.nanmean(fl))
            micro = pair_diff(interp, a, b, "micro-rs")
            conv = pair_diff(interp, a, b, "micro-cv")
            ph, pv = peak_cooling(height, micro, center)
            conv_at = float(np.interp(ph, height, conv))
            f.write(f"\n=== {tag} ===\n")
            f.write(f"microphysics_peak_height_km,{ph:.12g}\n")
            f.write(f"microphysics_peak_K_day,{pv:.12g}\n")
            f.write(f"convection_at_microphysics_peak_K_day,{conv_at:.12g}\n")
            if pv < 0.0 and not (conv_at < 0.0 and abs(conv_at) >= 0.5 * abs(pv)):
                f.write("test_result,pass: cooling peak is mainly in microphysics\n")
            elif pv < 0.0:
                f.write("test_result,flag: convection also cools strongly at that level\n")
            else:
                f.write("test_result,note: no microphysics cooling peak near freezing level\n")
            f.write("height_km,microphysics_K_day,convection_K_day\n")
            for z, m, c in zip(height, micro, conv):
                f.write(f"{z:.12g},{m:.12g},{c:.12g}\n")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    height, interp, nfiles = load_budget_profiles(args.ddh_cache.resolve(), args.max_height_km)
    lines = freezing_lines(args.temp_root.resolve())

    figure = output_dir / "8. temperature tendencies_450dpi.png"
    text_path = output_dir / "8. temperature tendencies_data.txt"
    plot_budget_difference(figure, height, interp, lines, args.dpi)
    write_text(text_path, height, interp, lines, nfiles)

    print(f"[saved] {figure}")
    print(f"[saved] {text_path}")
    for a, b in PAIRS:
        fl = np.concatenate([lines[a], lines[b]])
        ph, pv = peak_cooling(height, pair_diff(interp, a, b, "micro-rs"), float(np.nanmean(fl)))
        print(f"{pair_label(a, b)}: microphysics peak {pv:+.4g} K/day at {ph:.3f} km")


if __name__ == "__main__":
    main()
