#!/usr/bin/env python3
"""Graupel/freezing-level CT budget diagnostic for paper1."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAPER_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/paper1")
OUTPUT_DIR = PAPER_ROOT / "11_graupel_freezing_level_diagnostic"
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
}
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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot G1M-C1M CT microphysics/convection differences and 0 C isotherm."
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
    root = (
        ddh_cache
        / str(info["cache_tag"])
        / str(info["experiment_dir"])
    )
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


def load_budget_differences(ddh_cache: Path, max_height_km: float):
    profiles: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, int]] = {}
    for run in RUNS:
        for term in TERMS.values():
            component = str(term["component"])
            profiles[(run, component)] = average_component(ddh_cache, run, component)

    control_height, _, _ = profiles[("control", "micro-rs")]
    height = control_height[np.isfinite(control_height)]
    height = np.sort(height)
    keep = (height >= 0.0) & (height <= max_height_km)
    height = height[keep]

    out = {}
    for name, term in TERMS.items():
        component = str(term["component"])
        c_z, c_profile, c_n = profiles[("control", component)]
        g_z, g_profile, g_n = profiles[("graupel", component)]
        c_interp = interpolate_to(c_z, c_profile, height)
        g_interp = interpolate_to(g_z, g_profile, height)
        out[name] = {
            "control": c_interp,
            "graupel": g_interp,
            "difference": g_interp - c_interp,
            "control_n": c_n,
            "graupel_n": g_n,
        }
    return height, out


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
            t1 = tt[idx]
            t2 = tt[idx + 1]
            y1 = yy[idx]
            y2 = yy[idx + 1]
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
        temp = load_temperature_cache(temp_root, run)
        lines[run] = compute_freezing_line_km(axis, temp)
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


def plot_budget_difference(
    output_path: Path,
    height: np.ndarray,
    differences,
    freeze_line: np.ndarray,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    for name, term in TERMS.items():
        diff = np.asarray(differences[name]["difference"], dtype=np.float64)
        ax.plot(
            diff,
            height,
            color=str(term["color"]),
            linestyle=str(term["linestyle"]),
            linewidth=2.8,
            label=str(term["label"]),
        )
    freeze_mean = float(np.nanmean(freeze_line))
    freeze_min = float(np.nanmin(freeze_line))
    freeze_max = float(np.nanmax(freeze_line))
    ax.axhspan(freeze_min, freeze_max, color="0.85", alpha=0.55, linewidth=0)
    ax.axhline(freeze_mean, color="black", linestyle=":", linewidth=2.0, label="0 C isotherm mean")
    ax.axvline(0.0, color="0.35", linewidth=1.0)
    ax.set_ylim(0.0, float(np.nanmax(height)))
    ax.set_xlabel("G1M - C1M CT tendency (K day$^{-1}$)", fontsize=13)
    ax.set_ylabel("Altitude (km)", fontsize=13)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best", fontsize=11, framealpha=0.95)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_text(
    path: Path,
    height: np.ndarray,
    differences,
    lines: Dict[str, np.ndarray],
    peak_height_value: Tuple[float, float],
) -> None:
    peak_height_km, peak_value = peak_height_value
    conv_at_peak = float(np.interp(peak_height_km, height, differences["convection"]["difference"]))
    freeze_combined = np.nanmean(np.vstack([lines["control"], lines["graupel"]]), axis=0)
    freeze_mean = float(np.nanmean(freeze_combined))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("Graupel freezing-level CT budget diagnostic\n")
        f.write("DDH CT terms are read after ddhb conversion; units are K/day.\n")
        f.write("Sign convention: negative G1M-C1M means extra cooling in G1M.\n")
        f.write("Run difference: G1M - C1M.\n\n")
        f.write("Candidate CT DDH fields from FBL files\n")
        for run, info in RUNS.items():
            f.write(f"{info['label']} FBL: {info['fbl']}\n")
            for block, fields in parse_fbl_blocks(Path(info["fbl"])).items():
                f.write(f"  {block}: {', '.join(fields)}\n")
        f.write("\nSelected terms\n")
        f.write("microphysics = micro-rs = Microphysics scheme\n")
        f.write("convection = micro-cv = Convection scheme 3MT\n\n")
        f.write("Availability\n")
        for name, term in TERMS.items():
            f.write(
                f"{name},{term['component']},"
                f"C1M_days={int(differences[name]['control_n'])},"
                f"G1M_days={int(differences[name]['graupel_n'])}\n"
            )
        f.write("\n0 C isotherm from 3D temperature data\n")
        f.write(f"combined_mean_freezing_level_km,{freeze_mean:.12g}\n")
        for run, line in lines.items():
            f.write(
                f"{RUNS[run]['label']}_mean_freezing_level_km,"
                f"{float(np.nanmean(line)):.12g}\n"
            )
        f.write("\nMinimum microphysics difference near freezing level\n")
        f.write(f"microphysics_peak_height_km,{peak_height_km:.12g}\n")
        f.write(f"microphysics_peak_G1M_minus_C1M_K_day,{peak_value:.12g}\n")
        f.write(f"convection_at_microphysics_peak_K_day,{conv_at_peak:.12g}\n")
        f.write(f"microphysics_cooling_peak_present,{str(peak_value < 0.0).lower()}\n")
        if peak_value < 0.0 and not (conv_at_peak < 0.0 and abs(conv_at_peak) >= 0.5 * abs(peak_value)):
            f.write("test_result,pass: cooling peak is mainly in microphysics\n")
        elif peak_value < 0.0:
            f.write("test_result,flag: convection also cools strongly at that level\n")
        else:
            f.write("test_result,fail: no microphysics cooling peak found near freezing level\n")

        f.write("\nDifference profiles\n")
        f.write("height_km,microphysics_G1M_minus_C1M_K_day,convection_G1M_minus_C1M_K_day\n")
        micro = np.asarray(differences["microphysics"]["difference"], dtype=np.float64)
        conv = np.asarray(differences["convection"]["difference"], dtype=np.float64)
        for z, m, c in zip(height, micro, conv):
            f.write(f"{z:.12g},{m:.12g},{c:.12g}\n")

        f.write("\nFreezing level by local hour\n")
        f.write("hour,C1M_freezing_km,G1M_freezing_km\n")
        for hour in range(24):
            f.write(f"{hour},{lines['control'][hour]:.12g},{lines['graupel'][hour]:.12g}\n")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    height, differences = load_budget_differences(args.ddh_cache.resolve(), args.max_height_km)
    lines = freezing_lines(args.temp_root.resolve())
    combined_freezing = np.nanmean(np.vstack([lines["control"], lines["graupel"]]), axis=0)
    peak = peak_cooling(
        height,
        np.asarray(differences["microphysics"]["difference"], dtype=np.float64),
        float(np.nanmean(combined_freezing)),
    )

    budget_figure = output_dir / "graupel_ct_microphysics_convection_difference_450dpi.png"
    text_path = output_dir / "graupel_freezing_diagnostic_data.txt"
    plot_budget_difference(
        budget_figure,
        height,
        differences,
        combined_freezing,
        args.dpi,
    )
    write_text(text_path, height, differences, lines, peak)

    conv_at_peak = float(np.interp(peak[0], height, differences["convection"]["difference"]))
    print(f"[saved] {budget_figure}")
    print(f"[saved] {text_path}")
    if peak[1] < 0.0:
        print(f"Peak microphysics cooling near 0 C: {peak[1]:.4g} K/day at {peak[0]:.3f} km")
    else:
        print(f"Minimum microphysics difference near 0 C: {peak[1]:+.4g} K/day at {peak[0]:.3f} km")
        print("No microphysics cooling peak found near the 0 C level.")
    print(f"Convection difference at that height: {conv_at_peak:.4g} K/day")


if __name__ == "__main__":
    main()
