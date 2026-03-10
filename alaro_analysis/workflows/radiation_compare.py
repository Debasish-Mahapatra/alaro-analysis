#!/usr/bin/env python3
"""
Compare ALARO radiation terms against SURFEX net radiation.

The primary check is:
    ALARO_RN = SURFFLU.RAY.SOLA + SURFFLU.RAY.THER

and whether that roughly matches SURFEX SFX.RN over the masked domain.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.common.constants import EXPERIMENTS, EXPERIMENT_LABELS, SEASONS
from alaro_analysis.common.naming import safe_name
from alaro_analysis.common.seasons import build_period_specs, resolve_seasons
from alaro_analysis.common.spatial import build_spatial_window, spatial_window_tag
from alaro_analysis.common.timeparse import has_pf_subdirs
from alaro_analysis.data.cache import build_cache_file, load_cache, save_cache
from alaro_analysis.data.dataset_io import read_time_level_yx
from alaro_analysis.data.discovery import collect_file_records

DEFAULT_ALARO_CONTROL_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/masked-netcdf"
)
DEFAULT_ALARO_GRAUPEL_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/graupel/masked-netcdf"
)
DEFAULT_ALARO_2MOM_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/2mom/masked-netcdf"
)

DEFAULT_SURFEX_CONTROL_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/SURFEX/control/masked-netcdf"
)
DEFAULT_SURFEX_GRAUPEL_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/SURFEX/graupel/masked-netcdf"
)
DEFAULT_SURFEX_2MOM_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/SURFEX/2mom/masked-netcdf"
)

DEFAULT_OUTPUT_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/radiation_compare"
)
DEFAULT_INTERMEDIATE_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/radiation_compare"
)

VAR_TOKEN_RE = re.compile(r"[^A-Za-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ALARO radiation-derived net radiation against SURFEX SFX.RN."
    )
    parser.add_argument("--alaro-control-dir", type=Path, default=DEFAULT_ALARO_CONTROL_DIR)
    parser.add_argument("--alaro-graupel-dir", type=Path, default=DEFAULT_ALARO_GRAUPEL_DIR)
    parser.add_argument("--alaro-twomom-dir", type=Path, default=DEFAULT_ALARO_2MOM_DIR)

    parser.add_argument("--surfex-control-dir", type=Path, default=DEFAULT_SURFEX_CONTROL_DIR)
    parser.add_argument("--surfex-graupel-dir", type=Path, default=DEFAULT_SURFEX_GRAUPEL_DIR)
    parser.add_argument("--surfex-twomom-dir", type=Path, default=DEFAULT_SURFEX_2MOM_DIR)

    parser.add_argument("--shortwave-down-var", default="SURFRF.SHORT.DO")
    parser.add_argument("--longwave-down-var", default="SURFRF.LONG.DO")
    parser.add_argument("--solar-net-var", default="SURFFLU.RAY.SOLA")
    parser.add_argument("--thermal-net-var", default="SURFFLU.RAY.THER")
    parser.add_argument("--surfex-net-var", default="SFX.RN")

    parser.add_argument(
        "--seasons",
        nargs="+",
        default=list(SEASONS.keys()),
        help="Subset of seasons (wet dry ...). Use 'all' for all seasons.",
    )
    parser.add_argument(
        "--analysis-modes",
        nargs="+",
        default=("full", "seasonal"),
        choices=("full", "seasonal"),
        help="Run full 2-year analysis, seasonal analysis, or both.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--intermediate-dir", type=Path, default=DEFAULT_INTERMEDIATE_DIR)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--overwrite-intermediate",
        action="store_true",
        help="Overwrite existing intermediate cache files.",
    )
    parser.add_argument(
        "--recompute",
        dest="overwrite_intermediate",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--utc-offset-hours",
        type=int,
        default=-4,
        help="Local timezone offset from UTC (Amazon is -4).",
    )
    parser.add_argument(
        "--y-slice",
        default=None,
        help="Optional Python-style Y slice start:end for spatial averaging.",
    )
    parser.add_argument(
        "--x-slice",
        default=None,
        help="Optional Python-style X slice start:end for spatial averaging.",
    )
    parser.add_argument(
        "--list-variables",
        action="store_true",
        help="List discovered variables per experiment/root and exit.",
    )
    return parser.parse_args()


def normalize_var_token(name: str) -> str:
    return VAR_TOKEN_RE.sub("", name).upper()


def discover_variable_maps(experiment_dirs: dict[str, Path]) -> dict[str, dict[str, str]]:
    maps: dict[str, dict[str, str]] = {}
    for exp, exp_dir in experiment_dirs.items():
        token_map: dict[str, str] = {}
        for p in sorted(exp_dir.iterdir()):
            if not p.is_dir() or p.name.startswith(".") or not has_pf_subdirs(p):
                continue
            token = normalize_var_token(p.name)
            if token and token not in token_map:
                token_map[token] = p.name
        maps[exp] = token_map
    return maps


def resolve_var_name(
    variable_maps: dict[str, dict[str, str]],
    experiment: str,
    candidates: Sequence[str],
) -> str | None:
    token_map = variable_maps[experiment]
    for cand in candidates:
        token = normalize_var_token(cand)
        if token in token_map:
            return token_map[token]
    return None


def safe_scalar_mean(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def get_peer_file(base_file: Path, experiment_dir: Path, var_name: str) -> Path:
    return experiment_dir / var_name / base_file.parent.name / base_file.name


def read_mean_scalar(
    file_path: Path,
    variable_name: str,
    spatial_window,
) -> float:
    arr = read_time_level_yx(
        file_path,
        variable_name,
        spatial_window=spatial_window,
        token_normalizer=normalize_var_token,
    )
    return safe_scalar_mean(arr)


def finalize_line_means(
    sums: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, sum_arr in sums.items():
        cnt_arr = counts[key]
        mean = np.full(sum_arr.shape, np.nan, dtype=np.float64)
        valid = cnt_arr > 0
        mean[valid] = sum_arr[valid] / cnt_arr[valid]
        out[key] = mean
    return out


def compute_alaro_lines(
    *,
    experiment: str,
    experiment_dir: Path,
    records: list[tuple[int, Path]],
    names: dict[str, str | None],
    spatial_window,
) -> dict[str, np.ndarray]:
    sums = {
        "SW_DOWN": np.zeros((24,), dtype=np.float64),
        "LW_DOWN": np.zeros((24,), dtype=np.float64),
        "SW_NET": np.zeros((24,), dtype=np.float64),
        "LW_NET": np.zeros((24,), dtype=np.float64),
        "SW_UP": np.zeros((24,), dtype=np.float64),
        "LW_UP": np.zeros((24,), dtype=np.float64),
        "ALARO_RN": np.zeros((24,), dtype=np.float64),
    }
    counts = {key: np.zeros((24,), dtype=np.int64) for key in sums}

    used = 0
    skipped_bad = 0
    for idx, (hour, base_file) in enumerate(records, start=1):
        sw_net_file = get_peer_file(base_file, experiment_dir, names["SW_NET"])
        lw_net_file = get_peer_file(base_file, experiment_dir, names["LW_NET"])
        if not sw_net_file.exists() or not lw_net_file.exists():
            continue

        try:
            sw_net = read_mean_scalar(sw_net_file, names["SW_NET"], spatial_window)
            lw_net = read_mean_scalar(lw_net_file, names["LW_NET"], spatial_window)
        except Exception as exc:  # noqa: BLE001
            skipped_bad += 1
            print(
                f"[warn] {experiment}/alaro: skipping unreadable net-radiation pair "
                f"{sw_net_file.name} ({exc})",
                flush=True,
            )
            continue

        used += 1
        values = {
            "SW_NET": sw_net,
            "LW_NET": lw_net,
            "ALARO_RN": sw_net + lw_net if np.isfinite(sw_net) and np.isfinite(lw_net) else np.nan,
        }

        sw_down_name = names["SW_DOWN"]
        lw_down_name = names["LW_DOWN"]
        values["SW_DOWN"] = np.nan
        values["LW_DOWN"] = np.nan
        if sw_down_name is not None:
            sw_down_file = get_peer_file(base_file, experiment_dir, sw_down_name)
            if sw_down_file.exists():
                try:
                    values["SW_DOWN"] = read_mean_scalar(sw_down_file, sw_down_name, spatial_window)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[warn] {experiment}/alaro: could not read optional {sw_down_file.name} ({exc})",
                        flush=True,
                    )
        if lw_down_name is not None:
            lw_down_file = get_peer_file(base_file, experiment_dir, lw_down_name)
            if lw_down_file.exists():
                try:
                    values["LW_DOWN"] = read_mean_scalar(lw_down_file, lw_down_name, spatial_window)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[warn] {experiment}/alaro: could not read optional {lw_down_file.name} ({exc})",
                        flush=True,
                    )
        values["SW_UP"] = (
            values["SW_DOWN"] - values["SW_NET"]
            if np.isfinite(values["SW_DOWN"]) and np.isfinite(values["SW_NET"])
            else np.nan
        )
        values["LW_UP"] = (
            values["LW_DOWN"] - values["LW_NET"]
            if np.isfinite(values["LW_DOWN"]) and np.isfinite(values["LW_NET"])
            else np.nan
        )

        for key, value in values.items():
            if np.isfinite(value):
                sums[key][hour] += float(value)
                counts[key][hour] += 1

        if idx % 2000 == 0 or idx == len(records):
            print(f"[{experiment}/alaro] {idx}/{len(records)} files", flush=True)

    print(f"[{experiment}/alaro] used files: {used}/{len(records)} | unreadable skipped: {skipped_bad}", flush=True)
    return finalize_line_means(sums, counts)


def compute_single_variable_line(
    *,
    experiment: str,
    variable_dir: Path,
    variable_name: str,
    records: list[tuple[int, Path]],
    spatial_window,
) -> dict[str, np.ndarray]:
    sums = {"SURFEX_RN": np.zeros((24,), dtype=np.float64)}
    counts = {"SURFEX_RN": np.zeros((24,), dtype=np.int64)}
    used = 0
    skipped_bad = 0

    for idx, (hour, file_path) in enumerate(records, start=1):
        try:
            value = read_mean_scalar(file_path, variable_name, spatial_window)
        except Exception as exc:  # noqa: BLE001
            skipped_bad += 1
            print(
                f"[warn] {experiment}/surfex: skipping unreadable file {file_path.name} ({exc})",
                flush=True,
            )
            continue
        used += 1
        if np.isfinite(value):
            sums["SURFEX_RN"][hour] += float(value)
            counts["SURFEX_RN"][hour] += 1

        if idx % 2000 == 0 or idx == len(records):
            print(f"[{experiment}/surfex] {idx}/{len(records)} files", flush=True)

    print(f"[{experiment}/surfex] used files: {used}/{len(records)} | unreadable skipped: {skipped_bad}", flush=True)
    return finalize_line_means(sums, counts)


def compute_mae(left: np.ndarray, right: np.ndarray) -> float | None:
    mask = np.isfinite(left) & np.isfinite(right)
    if not np.any(mask):
        return None
    return float(np.mean(np.abs(left[mask] - right[mask])))


def plot_net_comparison(
    *,
    period_label: str,
    output_file: Path,
    alaro_lines: dict[str, dict[str, np.ndarray]],
    surfex_lines: dict[str, dict[str, np.ndarray]],
    utc_offset_hours: int,
) -> None:
    hours = np.arange(24, dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True, constrained_layout=True)

    for idx, exp in enumerate(EXPERIMENTS):
        ax = axes[idx]
        alaro_rn = alaro_lines[exp]["ALARO_RN"]
        surfex_rn = surfex_lines[exp]["SURFEX_RN"]

        ax.plot(hours, alaro_rn, color="black", linewidth=2.6, label="ALARO RN = SOLA + THER")
        ax.plot(hours, surfex_rn, color="#0b7285", linewidth=2.4, linestyle="--", label="SURFEX SFX.RN")
        ax.set_title(EXPERIMENT_LABELS[exp], fontsize=12, fontweight="bold")
        ax.set_xlabel(f"Hour (local UTC{utc_offset_hours:+d})", fontsize=11)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xlim(0.0, 23.0)
        ax.grid(alpha=0.25, linestyle="--")

        mae = compute_mae(alaro_rn, surfex_rn)
        if mae is not None:
            ax.text(
                0.02,
                0.95,
                f"MAE = {mae:.1f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
            )

    axes[0].set_ylabel("Net radiation [W m-2]", fontsize=12)
    axes[0].legend(loc="upper left", fontsize=10, framealpha=0.9)
    fig.suptitle(f"{period_label} - ALARO net radiation vs SURFEX SFX.RN", fontsize=15, fontweight="bold")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def plot_alaro_components(
    *,
    period_label: str,
    output_file: Path,
    alaro_lines: dict[str, dict[str, np.ndarray]],
    utc_offset_hours: int,
) -> None:
    hours = np.arange(24, dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True, constrained_layout=True)

    component_styles = {
        "SW_DOWN": {"color": "#f59f00", "linestyle": "-", "label": "SWD = SURFRF.SHORT.DO"},
        "LW_DOWN": {"color": "#5f3dc4", "linestyle": "-", "label": "LWD = SURFRF.LONG.DO"},
        "SW_NET": {"color": "#ff922b", "linestyle": "--", "label": "SWnet = SURFFLU.RAY.SOLA"},
        "LW_NET": {"color": "#1c7ed6", "linestyle": "--", "label": "LWnet = SURFFLU.RAY.THER"},
        "SW_UP": {"color": "#e8590c", "linestyle": ":", "label": "SWU = SWD - SWnet"},
        "LW_UP": {"color": "#6741d9", "linestyle": ":", "label": "LWU = LWD - LWnet"},
        "ALARO_RN": {"color": "black", "linestyle": "-", "label": "RN = SWnet + LWnet"},
    }

    for idx, exp in enumerate(EXPERIMENTS):
        ax = axes[idx]
        for key, style in component_styles.items():
            ax.plot(
                hours,
                alaro_lines[exp][key],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.2 if key == "ALARO_RN" else 2.0,
                label=style["label"],
            )
        ax.set_title(EXPERIMENT_LABELS[exp], fontsize=12, fontweight="bold")
        ax.set_xlabel(f"Hour (local UTC{utc_offset_hours:+d})", fontsize=11)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xlim(0.0, 23.0)
        ax.grid(alpha=0.25, linestyle="--")

    axes[0].set_ylabel("Radiative flux [W m-2]", fontsize=12)
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.suptitle(f"{period_label} - ALARO radiation components", fontsize=15, fontweight="bold")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def main() -> None:
    args = parse_args()
    spatial_window = build_spatial_window(args.y_slice, args.x_slice)
    spatial_tag = spatial_window_tag(spatial_window)

    alaro_dirs = {
        "control": args.alaro_control_dir.resolve(),
        "graupel": args.alaro_graupel_dir.resolve(),
        "2mom": args.alaro_twomom_dir.resolve(),
    }
    surfex_dirs = {
        "control": args.surfex_control_dir.resolve(),
        "graupel": args.surfex_graupel_dir.resolve(),
        "2mom": args.surfex_twomom_dir.resolve(),
    }
    for label, roots in (("ALARO", alaro_dirs), ("SURFEX", surfex_dirs)):
        for exp, d in roots.items():
            if not d.exists():
                raise FileNotFoundError(f"{label} {exp} data dir not found: {d}")

    alaro_variable_maps = discover_variable_maps(alaro_dirs)
    surfex_variable_maps = discover_variable_maps(surfex_dirs)

    if args.list_variables:
        for label, variable_maps in (("ALARO", alaro_variable_maps), ("SURFEX", surfex_variable_maps)):
            print(f"\n{label} variables:", flush=True)
            for exp in EXPERIMENTS:
                names = sorted(variable_maps[exp].values())
                print(f"- {exp}: {', '.join(names) if names else '(none)'}", flush=True)
        return

    selected_seasons = resolve_seasons(args.seasons)
    periods = build_period_specs(set(args.analysis_modes), selected_seasons)
    if not periods:
        raise RuntimeError("No analysis periods selected.")

    resolved_alaro = {
        exp: {
            "SW_DOWN": resolve_var_name(alaro_variable_maps, exp, (args.shortwave_down_var,)),
            "LW_DOWN": resolve_var_name(alaro_variable_maps, exp, (args.longwave_down_var,)),
            "SW_NET": resolve_var_name(alaro_variable_maps, exp, (args.solar_net_var,)),
            "LW_NET": resolve_var_name(alaro_variable_maps, exp, (args.thermal_net_var,)),
        }
        for exp in EXPERIMENTS
    }
    resolved_surfex = {
        exp: resolve_var_name(surfex_variable_maps, exp, (args.surfex_net_var,))
        for exp in EXPERIMENTS
    }

    print("\nALARO directories:", flush=True)
    for exp in EXPERIMENTS:
        print(f"- {exp}: {alaro_dirs[exp]}", flush=True)
    print("\nSURFEX directories:", flush=True)
    for exp in EXPERIMENTS:
        print(f"- {exp}: {surfex_dirs[exp]}", flush=True)
    print("\nResolved ALARO variable names:", flush=True)
    for exp in EXPERIMENTS:
        print(f"- {exp}: {resolved_alaro[exp]}", flush=True)
    print("\nResolved SURFEX variable names:", flush=True)
    for exp in EXPERIMENTS:
        print(f"- {exp}: {resolved_surfex[exp]}", flush=True)
    print("\nOutput directory:", args.output_dir.resolve(), flush=True)
    print("Intermediate directory:", args.intermediate_dir.resolve(), flush=True)
    print("Periods:", [p.key for p in periods], flush=True)
    print("Spatial averaging tag:", spatial_tag, flush=True)
    print(
        "Spatial Y slice:",
        f"{spatial_window.y_start}:{spatial_window.y_end}",
        "| X slice:",
        f"{spatial_window.x_start}:{spatial_window.x_end}",
        flush=True,
    )
    print("Ignoring +0024 files by design.", flush=True)

    output_dir = args.output_dir.resolve()
    intermediate_dir = args.intermediate_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    required_alaro_keys = ("SW_DOWN", "LW_DOWN", "SW_NET", "LW_NET", "SW_UP", "LW_UP", "ALARO_RN")

    for period in periods:
        print(f"\n===== Comparing radiation for {period.label} ({period.key}) =====", flush=True)
        alaro_payloads: dict[str, dict[str, np.ndarray]] = {}
        surfex_payloads: dict[str, dict[str, np.ndarray]] = {}

        for exp in EXPERIMENTS:
            if resolved_surfex[exp] is None:
                print(f"[warn] {period.key}/{exp}: SURFEX variable '{args.surfex_net_var}' not found.", flush=True)
                continue
            if resolved_alaro[exp]["SW_NET"] is None or resolved_alaro[exp]["LW_NET"] is None:
                print(
                    f"[warn] {period.key}/{exp}: missing required ALARO variables "
                    f"{args.solar_net_var}/{args.thermal_net_var}.",
                    flush=True,
                )
                continue

            alaro_ref_dir = alaro_dirs[exp] / resolved_alaro[exp]["SW_NET"]
            surfex_ref_dir = surfex_dirs[exp] / resolved_surfex[exp]

            alaro_records = collect_file_records(
                variable_dir=alaro_ref_dir,
                max_days=args.max_days,
                allowed_months=period.allowed_months,
                utc_offset_hours=args.utc_offset_hours,
            )
            surfex_records = collect_file_records(
                variable_dir=surfex_ref_dir,
                max_days=args.max_days,
                allowed_months=period.allowed_months,
                utc_offset_hours=args.utc_offset_hours,
            )
            if not alaro_records:
                print(f"[warn] {period.key}/{exp}: no ALARO records found in {alaro_ref_dir}", flush=True)
                continue
            if not surfex_records:
                print(f"[warn] {period.key}/{exp}: no SURFEX records found in {surfex_ref_dir}", flush=True)
                continue

            cache_file = build_cache_file(
                intermediate_dir=intermediate_dir,
                analysis_name="radiation_compare",
                period_subdir=period.output_subdir,
                experiment=exp,
                spatial_tag=spatial_tag,
            )
            use_cache = cache_file.exists() and not args.overwrite_intermediate
            if use_cache:
                payload = load_cache(cache_file)
                if all(key in payload for key in required_alaro_keys) and "SURFEX_RN" in payload:
                    alaro_payloads[exp] = {
                        key: np.asarray(payload[key], dtype=np.float64)
                        for key in required_alaro_keys
                    }
                    surfex_payloads[exp] = {"SURFEX_RN": np.asarray(payload["SURFEX_RN"], dtype=np.float64)}
                    continue
                print(f"[{period.key}/{exp}] cache missing new radiation keys; recomputing.", flush=True)

            if exp not in alaro_payloads or exp not in surfex_payloads:
                alaro_payload = compute_alaro_lines(
                    experiment=exp,
                    experiment_dir=alaro_dirs[exp],
                    records=alaro_records,
                    names=resolved_alaro[exp],  # type: ignore[arg-type]
                    spatial_window=spatial_window,
                )
                surfex_payload = compute_single_variable_line(
                    experiment=exp,
                    variable_dir=surfex_ref_dir,
                    variable_name=resolved_surfex[exp],  # type: ignore[arg-type]
                    records=surfex_records,
                    spatial_window=spatial_window,
                )
                alaro_payloads[exp] = alaro_payload
                surfex_payloads[exp] = surfex_payload
                if args.max_days is None:
                    save_cache(
                        cache_file,
                        {
                            **{key: value for key, value in alaro_payload.items()},
                            "SURFEX_RN": surfex_payload["SURFEX_RN"],
                        },
                    )

        if not all(exp in alaro_payloads and exp in surfex_payloads for exp in EXPERIMENTS):
            print(f"[warn] {period.key}: incomplete experiment set; skipping plots.", flush=True)
            continue

        suffix = f"_{spatial_tag}" if spatial_tag != "full-domain" else ""
        net_out = (
            output_dir
            / "net_radiation_compare"
            / period.output_subdir
            / f"net_radiation_compare{suffix}.png"
        )
        comp_out = (
            output_dir
            / "alaro_components"
            / period.output_subdir
            / f"alaro_radiation_components{suffix}.png"
        )

        plot_net_comparison(
            period_label=period.label,
            output_file=net_out,
            alaro_lines=alaro_payloads,
            surfex_lines=surfex_payloads,
            utc_offset_hours=args.utc_offset_hours,
        )
        plot_alaro_components(
            period_label=period.label,
            output_file=comp_out,
            alaro_lines=alaro_payloads,
            utc_offset_hours=args.utc_offset_hours,
        )

    print("\nCompleted ALARO/SURFEX radiation comparison.", flush=True)


if __name__ == "__main__":
    main()
