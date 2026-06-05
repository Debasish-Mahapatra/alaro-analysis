"""Vertical profiles of DSD diagnostics and DDH QV for C1M, G1M, and G2M."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from netCDF4 import Dataset

from alaro_analysis.common.cli_config import add_config_argument, parse_configured_args
from alaro_analysis.common.constants import (
    EXPERIMENT_COLORS,
    EXPERIMENT_LABELS,
    EXPERIMENTS,
    FREEZING_K,
    RD,
)
from alaro_analysis.common.dsd import (
    MP_FIXED_N0_PER_M3_MM,
    gamma_dsd_from_q_n_per_kg,
    mp_from_q_abel_boutle,
    mp_from_q_fixed_n0,
    mp_from_q_n_per_kg,
)
from alaro_analysis.ddh.io import AGG_DIR as DDH_AGG_DIR
from alaro_analysis.workflows.disdrometer_comparison import (
    RUNS_ROOT,
    lead_label,
    parse_lead_selection,
)
from alaro_analysis.workflows.disdrometer_dsd import (
    MASK_FILE,
    NETCDF_ROOT,
    PF_DAY_RE,
    PF_FILE_RE,
    DomainMask,
    build_domain_mask_from_netcdf,
)


OUTPUT_DIR = RUNS_ROOT / "figures" / "dsd_vertical_profiles"
DATA_TXT_DIR = OUTPUT_DIR / "data_txt"
CACHE_DIR = RUNS_ROOT / "processed-data" / "dsd_vertical_profiles"
FIGURE_NAME = "dsd_vertical_profiles_D0_logNw_DDH_QV.png"
TEXT_NAME = "dsd_vertical_profiles_D0_logNw_DDH_QV.txt"

DSD_FIELDS = ("d0_mm", "log_nw")
PANEL_SPECS = (
    ("d0_mm", "D$_0$", "D$_0$ (mm)"),
    ("log_nw", "log$_{10}$ N$_w$", "log$_{10}$ N$_w$ (m$^{-3}$ mm$^{-1}$)"),
    ("qv", "DDH QV", "DDH QV tendency (g kg$^{-1}$ day$^{-1}$)"),
)
PANEL_LABELS = ("a", "b", "c")


@dataclass(frozen=True)
class DsdVerticalProfile:
    height_km: np.ndarray
    temperature_k: np.ndarray
    temperature_count: np.ndarray
    values: dict[str, np.ndarray]
    counts: dict[str, np.ndarray]
    n_files: int
    source: str


@dataclass(frozen=True)
class DdhQvProfile:
    height_km: np.ndarray
    values: np.ndarray
    n_days: int
    source: str
    block: str


_WORKER_MASK: np.ndarray | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot vertical profiles of D0, log10(Nw), and DDH QV for "
            "C1M, G1M, and G2M."
        )
    )
    add_config_argument(parser)
    parser.add_argument("--netcdf-root", type=Path, default=NETCDF_ROOT)
    parser.add_argument("--ddh-agg-dir", type=Path, default=DDH_AGG_DIR)
    parser.add_argument("--mask-file", type=Path, default=MASK_FILE)
    parser.add_argument("--mask-var", default=None)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data-txt-dir", type=Path, default=DATA_TXT_DIR)
    parser.add_argument(
        "--lead",
        default="all",
        help="Forecast leads used for D0/log10(Nw). Use 'all' for all model timesteps.",
    )
    parser.add_argument(
        "--ddh-lead",
        default=None,
        help="DDH lead used for the QV panel. Defaults to 0024.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS),
        choices=list(EXPERIMENTS),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tasks-per-child", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--min-qr", type=float, default=1.0e-7)
    parser.add_argument(
        "--onemom-closure",
        choices=("abel_boutle", "fixed_n0"),
        default="abel_boutle",
    )
    parser.add_argument(
        "--twomom-closure",
        choices=("gamma_mu1", "marshall_palmer"),
        default="gamma_mu1",
        help="DSD closure for G2M from prognostic rain mass and number.",
    )
    parser.add_argument("--n0-fixed", type=float, default=MP_FIXED_N0_PER_M3_MM)
    parser.add_argument("--qv-block", default="VQVM")
    parser.add_argument(
        "--qv-label",
        default=PANEL_SPECS[-1][2],
        help="X-axis label for the DDH QV panel.",
    )
    parser.add_argument("--max-height-km", type=float, default=9.0)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--recompute", action="store_true")
    return parse_configured_args(parser, "dsd_vertical_profiles", argv=argv)


def _lead_text(value: str) -> str:
    return f"{int(value):04d}"


def ddh_lead_text(leads: tuple[int, ...] | None, ddh_lead: str | None) -> str:
    if ddh_lead is not None:
        return _lead_text(ddh_lead)
    return "0024"


def cache_tag(
    leads: tuple[int, ...] | None,
    *,
    onemom_closure: str,
    twomom_closure: str,
    max_days: int | None,
) -> str:
    tag = f"{lead_label(leads)}_{onemom_closure}_{twomom_closure}"
    if max_days is not None:
        tag += f"_first{max_days}days"
    return tag


def dsd_cache_path(cache_dir: Path, experiment: str, tag: str) -> Path:
    return cache_dir / f"{experiment}_{tag}.npz"


def discover_vertical_records(
    experiment: str,
    leads: tuple[int, ...] | None,
    netcdf_root: Path,
    max_days: int | None,
) -> list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]]:
    base = netcdf_root / experiment / "masked-netcdf"
    ref_dir = base / "RAIN"
    if not ref_dir.exists():
        raise FileNotFoundError(f"Missing variable folder: {ref_dir}")

    needed = ["RAIN", "TEMPERATURE", "PRESSURE", "GEOPOTENTIEL"]
    if experiment == "2mom":
        needed.append("PNR")

    day_dirs = sorted(d for d in ref_dir.iterdir() if d.is_dir() and PF_DAY_RE.match(d.name))
    if max_days is not None:
        day_dirs = day_dirs[:max_days]
    lead_set = set(leads) if leads is not None else None

    records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]] = []
    for day_dir in day_dirs:
        day_match = PF_DAY_RE.match(day_dir.name)
        if not day_match:
            continue
        init_dt = datetime.strptime(day_match.group(1), "%Y%m%d")
        for path in sorted(day_dir.iterdir()):
            file_match = PF_FILE_RE.match(path.name)
            if not file_match:
                continue
            lead = int(file_match.group(1))
            if lead_set is not None and lead not in lead_set:
                continue

            paths: dict[str, Path] = {}
            for var in needed:
                candidate = base / var / day_dir.name / path.name
                if not candidate.exists():
                    break
                paths[var] = candidate
            else:
                valid_dt = init_dt + timedelta(hours=lead)
                records.append(
                    (np.datetime64(valid_dt, "s"), np.datetime64(init_dt, "s"), lead, paths)
                )
    records.sort(key=lambda rec: rec[0])
    return records


def _init_worker(mask: np.ndarray) -> None:
    global _WORKER_MASK
    _WORKER_MASK = np.asarray(mask, dtype=bool)


def _as_float_array(raw: Any) -> np.ndarray:
    if np.ma.isMaskedArray(raw):
        return np.asarray(raw.filled(np.nan), dtype=np.float64)
    return np.asarray(raw, dtype=np.float64)


def _to_level_yx(arr: np.ndarray, mask_shape: tuple[int, int], path: Path, var: str) -> np.ndarray:
    if arr.ndim == 4:
        if arr.shape[-2:] != mask_shape:
            raise ValueError(f"{var} in {path} has incompatible shape {arr.shape}")
        if arr.shape[0] == 1:
            return arr[0]
        valid = np.isfinite(arr)
        counts = np.sum(valid, axis=0)
        sums = np.sum(np.where(valid, arr, 0.0), axis=0)
        out = np.full(counts.shape, np.nan, dtype=np.float64)
        ok = counts > 0
        out[ok] = sums[ok] / counts[ok]
        return out
    if arr.ndim == 3:
        if arr.shape[-2:] != mask_shape:
            raise ValueError(f"{var} in {path} has incompatible shape {arr.shape}")
        return arr
    if arr.ndim == 2:
        if arr.shape != mask_shape:
            raise ValueError(f"{var} in {path} has incompatible shape {arr.shape}")
        return arr[np.newaxis, :, :]
    raise ValueError(f"Unexpected ndim for {var} in {path}: {arr.ndim}")


def read_masked_level_mean(path: Path, var: str, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(path) as ds:
        arr = _as_float_array(ds.variables[var][:])
    field = _to_level_yx(arr, mask.shape, path, var)
    valid = np.isfinite(field) & mask[np.newaxis, :, :]
    counts = np.sum(valid, axis=(1, 2)).astype(np.int64)
    sums = np.sum(np.where(valid, field, 0.0), axis=(1, 2))
    profile = np.full(counts.shape, np.nan, dtype=np.float64)
    ok = counts > 0
    profile[ok] = sums[ok] / counts[ok]
    return profile, counts


def empty_partial(n_levels: int) -> dict[str, Any]:
    return {
        "sums": {field: np.zeros(n_levels, dtype=np.float64) for field in DSD_FIELDS},
        "counts": {field: np.zeros(n_levels, dtype=np.int64) for field in DSD_FIELDS},
        "height_sum": np.zeros(n_levels, dtype=np.float64),
        "height_count": np.zeros(n_levels, dtype=np.int64),
        "temperature_sum": np.zeros(n_levels, dtype=np.float64),
        "temperature_count": np.zeros(n_levels, dtype=np.int64),
        "n_files": 0,
    }


def add_partial(total: dict[str, Any], part: dict[str, Any]) -> dict[str, Any]:
    if not total:
        total.update(empty_partial(part["height_sum"].size))
    for field in DSD_FIELDS:
        total["sums"][field] += part["sums"][field]
        total["counts"][field] += part["counts"][field]
    total["height_sum"] += part["height_sum"]
    total["height_count"] += part["height_count"]
    total["temperature_sum"] += part["temperature_sum"]
    total["temperature_count"] += part["temperature_count"]
    total["n_files"] += int(part["n_files"])
    return total


def _process_profile_task(
    task: tuple[str, dict[str, str], float, str, str, float],
) -> tuple[dict[str, Any] | None, list[str]]:
    experiment, paths, min_qr, onemom_closure, twomom_closure, n0_fixed = task
    if _WORKER_MASK is None:
        raise RuntimeError("Worker mask not initialised")
    mask = _WORKER_MASK
    warnings: list[str] = []

    try:
        qr, _ = read_masked_level_mean(Path(paths["RAIN"]), "RAIN", mask)
        temp, _ = read_masked_level_mean(Path(paths["TEMPERATURE"]), "TEMPERATURE", mask)
        pres, _ = read_masked_level_mean(Path(paths["PRESSURE"]), "PRESSURE", mask)
        height_m, height_count = read_masked_level_mean(
            Path(paths["GEOPOTENTIEL"]), "GEOPOTENTIEL", mask
        )
        if experiment == "2mom":
            pnr, _ = read_masked_level_mean(Path(paths["PNR"]), "PNR", mask)
        else:
            pnr = np.full(qr.shape, np.nan, dtype=np.float64)
    except Exception as exc:  # pragma: no cover - defensive worker logging
        label = paths.get("RAIN", "<missing RAIN>")
        return None, [f"WARNING {experiment} {label}: {exc}"]

    rho = np.full(qr.shape, np.nan, dtype=np.float64)
    ok_rho = np.isfinite(pres) & np.isfinite(temp) & (temp > 0.0)
    rho[ok_rho] = pres[ok_rho] / (RD * temp[ok_rho])

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if experiment == "2mom":
            if twomom_closure == "gamma_mu1":
                diag = gamma_dsd_from_q_n_per_kg(qr, pnr, rho, mu=1.0)
            else:
                diag = mp_from_q_n_per_kg(qr, pnr, rho)
        elif onemom_closure == "abel_boutle":
            diag = mp_from_q_abel_boutle(qr, rho)
        else:
            diag = mp_from_q_fixed_n0(qr, rho, n0_per_m3_mm=n0_fixed)

    part = empty_partial(qr.size)
    rainy = np.isfinite(qr) & (qr >= min_qr)
    for field in DSD_FIELDS:
        values = np.asarray(diag[field], dtype=np.float64)
        valid = rainy & np.isfinite(values)
        part["sums"][field][valid] = values[valid]
        part["counts"][field][valid] = 1

    height_valid = np.isfinite(height_m)
    part["height_sum"][height_valid] = height_m[height_valid] / 1000.0
    part["height_count"][height_valid] = height_count[height_valid] > 0
    temp_valid = np.isfinite(temp)
    part["temperature_sum"][temp_valid] = temp[temp_valid]
    part["temperature_count"][temp_valid] = 1
    part["n_files"] = 1
    return part, warnings


def profile_from_accumulator(acc: dict[str, Any], source: str) -> DsdVerticalProfile:
    height = np.full(acc["height_sum"].shape, np.nan, dtype=np.float64)
    height_ok = acc["height_count"] > 0
    height[height_ok] = acc["height_sum"][height_ok] / acc["height_count"][height_ok]
    temperature = np.full(acc["temperature_sum"].shape, np.nan, dtype=np.float64)
    temp_ok = acc["temperature_count"] > 0
    temperature[temp_ok] = acc["temperature_sum"][temp_ok] / acc["temperature_count"][temp_ok]

    values: dict[str, np.ndarray] = {}
    for field in DSD_FIELDS:
        counts = acc["counts"][field]
        vals = np.full(counts.shape, np.nan, dtype=np.float64)
        ok = counts > 0
        vals[ok] = acc["sums"][field][ok] / counts[ok]
        values[field] = vals

    return DsdVerticalProfile(
        height_km=height,
        temperature_k=temperature,
        temperature_count=acc["temperature_count"].copy(),
        values=values,
        counts={field: acc["counts"][field].copy() for field in DSD_FIELDS},
        n_files=int(acc["n_files"]),
        source=source,
    )


def compute_dsd_profile(
    experiment: str,
    records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]],
    domain_mask: DomainMask,
    *,
    min_qr: float,
    onemom_closure: str,
    twomom_closure: str,
    n0_fixed: float,
    workers: int,
    progress_every: int,
    tasks_per_child: int,
) -> DsdVerticalProfile:
    if not records:
        raise RuntimeError(f"No masked-NetCDF records found for {experiment}")

    tasks = [
        (
            experiment,
            {name: str(path) for name, path in rec[3].items()},
            min_qr,
            onemom_closure,
            twomom_closure,
            n0_fixed,
        )
        for rec in records
    ]
    print(f"  [{experiment}] processing {len(tasks):,} vertical profile files", flush=True)

    acc: dict[str, Any] = {}
    if workers <= 1:
        _init_worker(domain_mask.mask)
        iterator = (_process_profile_task(task) for task in tasks)
        for idx, (part, warnings) in enumerate(iterator, start=1):
            if part is not None:
                add_partial(acc, part)
            for warning in warnings:
                print(warning, flush=True)
            if idx % progress_every == 0 or idx == len(tasks):
                print(f"  [{experiment}] processed {idx}/{len(tasks)}", flush=True)
    else:
        with get_context("fork").Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(domain_mask.mask,),
            maxtasksperchild=tasks_per_child,
        ) as pool:
            for idx, (part, warnings) in enumerate(
                pool.imap_unordered(_process_profile_task, tasks),
                start=1,
            ):
                if part is not None:
                    add_partial(acc, part)
                for warning in warnings:
                    print(warning, flush=True)
                if idx % progress_every == 0 or idx == len(tasks):
                    print(f"  [{experiment}] processed {idx}/{len(tasks)}", flush=True)

    if not acc:
        raise RuntimeError(f"No readable profile data for {experiment}")
    return profile_from_accumulator(acc, source=str(records[0][3]["RAIN"].parent.parent))


def save_dsd_profile(path: Path, profile: DsdVerticalProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        height_km=profile.height_km,
        temperature_k=profile.temperature_k,
        temperature_count=profile.temperature_count,
        n_files=np.asarray([profile.n_files], dtype=np.int64),
        source=np.asarray([profile.source]),
        **{field: profile.values[field] for field in DSD_FIELDS},
        **{f"count__{field}": profile.counts[field] for field in DSD_FIELDS},
    )


def load_dsd_profile(path: Path) -> DsdVerticalProfile:
    with np.load(path, allow_pickle=False) as data:
        source = str(data["source"][0]) if "source" in data.files else str(path)
        n_files_raw = data["n_files"] if "n_files" in data.files else np.asarray([0])
        n_files = int(np.ravel(n_files_raw)[0]) if np.size(n_files_raw) else 0
        height = np.asarray(data["height_km"], dtype=np.float64)
        if "temperature_k" in data.files:
            temperature = np.asarray(data["temperature_k"], dtype=np.float64)
        else:
            temperature = np.full(height.shape, np.nan, dtype=np.float64)
        if "temperature_count" in data.files:
            temperature_count = np.asarray(data["temperature_count"], dtype=np.int64)
        else:
            temperature_count = np.zeros(height.shape, dtype=np.int64)
        return DsdVerticalProfile(
            height_km=height,
            temperature_k=temperature,
            temperature_count=temperature_count,
            values={field: np.asarray(data[field], dtype=np.float64) for field in DSD_FIELDS},
            counts={
                field: np.asarray(data[f"count__{field}"], dtype=np.int64)
                for field in DSD_FIELDS
            },
            n_files=n_files,
            source=source,
        )


def get_dsd_profile(
    *,
    cache_dir: Path,
    experiment: str,
    tag: str,
    records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]],
    domain_mask: DomainMask,
    min_qr: float,
    onemom_closure: str,
    twomom_closure: str,
    n0_fixed: float,
    workers: int,
    progress_every: int,
    tasks_per_child: int,
    recompute: bool,
) -> DsdVerticalProfile:
    path = dsd_cache_path(cache_dir, experiment, tag)
    if path.exists() and not recompute:
        return load_dsd_profile(path)
    profile = compute_dsd_profile(
        experiment,
        records,
        domain_mask,
        min_qr=min_qr,
        onemom_closure=onemom_closure,
        twomom_closure=twomom_closure,
        n0_fixed=n0_fixed,
        workers=workers,
        progress_every=progress_every,
        tasks_per_child=tasks_per_child,
    )
    save_dsd_profile(path, profile)
    return profile


def load_ddh_qv_profile(
    agg_dir: Path,
    experiment: str,
    *,
    lead_text_value: str,
    block: str,
) -> DdhQvProfile:
    path = agg_dir / f"lead{lead_text_value}_VZ" / f"{experiment}_QV.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing DDH QV aggregate: {path}")

    key = f"block__{block}"
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            blocks = sorted(k[len("block__"):] for k in data.files if k.startswith("block__"))
            raise KeyError(f"{path} has no {key}. Available QV blocks: {blocks}")
        height = np.asarray(data["altitude_km"], dtype=np.float64)
        values = np.asarray(data[key], dtype=np.float64)
        n_days = int(data["days"].shape[0]) if "days" in data.files else 0

    order = np.argsort(height)
    return DdhQvProfile(
        height_km=height[order],
        values=values[order],
        n_days=n_days,
        source=str(path),
        block=block,
    )


def sorted_profile_xy(height: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(height) & np.isfinite(values)
    if not ok.any():
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    z = height[ok]
    x = values[ok]
    order = np.argsort(z)
    return x[order], z[order]


def freezing_level_km(height_km: np.ndarray, temperature_k: np.ndarray) -> float:
    z = np.asarray(height_km, dtype=np.float64)
    t = np.asarray(temperature_k, dtype=np.float64)
    ok = np.isfinite(z) & np.isfinite(t)
    if np.count_nonzero(ok) < 2:
        return float("nan")
    z = z[ok]
    t = t[ok]
    order = np.argsort(z)
    z = z[order]
    t = t[order]
    diff = t - FREEZING_K
    crossings = np.where(np.signbit(diff[:-1]) != np.signbit(diff[1:]))[0]
    if crossings.size == 0:
        return float("nan")
    idx = int(crossings[0])
    denom = diff[idx] - diff[idx + 1]
    if denom == 0.0:
        return float(z[idx])
    weight = diff[idx] / denom
    return float(z[idx] + weight * (z[idx + 1] - z[idx]))


def freezing_levels_by_experiment(
    dsd_profiles: dict[str, DsdVerticalProfile],
    experiments: list[str],
) -> dict[str, float]:
    return {
        experiment: freezing_level_km(
            dsd_profiles[experiment].height_km,
            dsd_profiles[experiment].temperature_k,
        )
        for experiment in experiments
    }


def plot_profiles(
    output_path: Path,
    dsd_profiles: dict[str, DsdVerticalProfile],
    qv_profiles: dict[str, DdhQvProfile],
    experiments: list[str],
    *,
    qv_label: str,
    max_height_km: float,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        len(PANEL_SPECS),
        figsize=(5.0 * len(PANEL_SPECS), 8.0),
        sharey=True,
    )
    freezing_levels = freezing_levels_by_experiment(dsd_profiles, experiments)

    for panel_idx, (field, title, xlabel) in enumerate(PANEL_SPECS):
        ax = axes[panel_idx]
        for experiment in experiments:
            color = EXPERIMENT_COLORS.get(experiment, "black")
            label = EXPERIMENT_LABELS.get(experiment, experiment)
            if field == "qv":
                values = qv_profiles[experiment].values
                height = qv_profiles[experiment].height_km
            else:
                values = dsd_profiles[experiment].values[field]
                height = dsd_profiles[experiment].height_km
            x, z = sorted_profile_xy(height, values)
            ax.plot(x, z, color=color, linewidth=2.4, label=label)
            freeze_z = freezing_levels[experiment]
            if np.isfinite(freeze_z):
                ax.axhline(
                    freeze_z,
                    color=color,
                    linestyle="--",
                    linewidth=1.3,
                    alpha=0.75,
                    zorder=1,
                )

        ax.set_title(f"({PANEL_LABELS[panel_idx]}) {title}", fontsize=13)
        ax.set_xlabel(qv_label if field == "qv" else xlabel)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        if panel_idx == 0:
            ax.set_ylabel("Height (km)")
            handles, labels = ax.get_legend_handles_labels()
            handles.append(
                plt.Line2D([], [], color="0.25", linestyle="--", linewidth=1.3)
            )
            labels.append("0 $^{\\circ}$C level")
            ax.legend(handles=handles, labels=labels, loc="best", frameon=False)

    axes[0].set_ylim(0.0, max_height_km)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_data_txt(
    path: Path,
    dsd_profiles: dict[str, DsdVerticalProfile],
    qv_profiles: dict[str, DdhQvProfile],
    experiments: list[str],
    *,
    lead_tag: str,
    ddh_lead: str,
    onemom_closure: str,
    twomom_closure: str,
    qv_block: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Vertical DSD diagnostics and DDH QV profiles\n")
        fh.write("=" * 72 + "\n")
        fh.write(f"DSD lead selection: {lead_tag}\n")
        fh.write(f"DDH QV lead: lead{ddh_lead}_VZ\n")
        fh.write(f"1-moment closure: {onemom_closure}\n")
        fh.write(f"2-moment closure: {twomom_closure}\n")
        fh.write(f"DDH QV block: {qv_block}\n\n")

        fh.write("DSD profiles\n")
        fh.write(
            "experiment,experiment_label,level_index,height_km,"
            "temperature_k,d0_mm,log_nw,"
            "count_temperature,count_d0,count_log_nw,n_files,source\n"
        )
        for experiment in experiments:
            profile = dsd_profiles[experiment]
            label = EXPERIMENT_LABELS.get(experiment, experiment)
            for idx, height in enumerate(profile.height_km):
                fh.write(
                    f"{experiment},{label},{idx},{height:.10e},"
                    f"{profile.temperature_k[idx]:.10e},"
                    f"{profile.values['d0_mm'][idx]:.10e},"
                    f"{profile.values['log_nw'][idx]:.10e},"
                    f"{int(profile.temperature_count[idx])},"
                    f"{int(profile.counts['d0_mm'][idx])},"
                    f"{int(profile.counts['log_nw'][idx])},"
                    f"{profile.n_files},{profile.source}\n"
                )

        fh.write("\nFreezing levels\n")
        fh.write("experiment,experiment_label,freezing_level_km\n")
        for experiment, freeze_z in freezing_levels_by_experiment(
            dsd_profiles, experiments
        ).items():
            label = EXPERIMENT_LABELS.get(experiment, experiment)
            fh.write(f"{experiment},{label},{freeze_z:.10e}\n")

        fh.write("\nDDH QV profile\n")
        fh.write("experiment,experiment_label,level_index,height_km,qv,n_days,source\n")
        for experiment in experiments:
            profile = qv_profiles[experiment]
            label = EXPERIMENT_LABELS.get(experiment, experiment)
            for idx, height in enumerate(profile.height_km):
                fh.write(
                    f"{experiment},{label},{idx},{height:.10e},"
                    f"{profile.values[idx]:.10e},{profile.n_days},{profile.source}\n"
                )


def build_profiles(args: argparse.Namespace) -> tuple[
    dict[str, DsdVerticalProfile],
    dict[str, DdhQvProfile],
    str,
    str,
]:
    leads = parse_lead_selection(args.lead)
    lead_tag = lead_label(leads)
    ddh_lead = ddh_lead_text(leads, args.ddh_lead)
    tag = cache_tag(
        leads,
        onemom_closure=args.onemom_closure,
        twomom_closure=args.twomom_closure,
        max_days=args.max_days,
    )

    records_by_experiment = {
        experiment: discover_vertical_records(
            experiment,
            leads,
            args.netcdf_root,
            args.max_days,
        )
        for experiment in args.experiments
    }
    sample_records = next((records for records in records_by_experiment.values() if records), None)
    if not sample_records:
        raise RuntimeError("No masked-NetCDF records found for any requested experiment")
    domain_mask = build_domain_mask_from_netcdf(
        sample_records[0][3]["RAIN"],
        args.mask_file,
        mask_var=args.mask_var,
        mask_threshold=args.mask_threshold,
    )
    print(
        f"Radar mask: {domain_mask.n_cells} / {domain_mask.mask.size} cells "
        f"({domain_mask.selected_var})",
        flush=True,
    )

    dsd_profiles: dict[str, DsdVerticalProfile] = {}
    qv_profiles: dict[str, DdhQvProfile] = {}
    for experiment in args.experiments:
        print(f"Processing {EXPERIMENT_LABELS.get(experiment, experiment)}", flush=True)
        dsd_profiles[experiment] = get_dsd_profile(
            cache_dir=args.cache_dir,
            experiment=experiment,
            tag=tag,
            records=records_by_experiment[experiment],
            domain_mask=domain_mask,
            min_qr=args.min_qr,
            onemom_closure=args.onemom_closure,
            twomom_closure=args.twomom_closure,
            n0_fixed=args.n0_fixed,
            workers=max(1, int(args.workers)),
            progress_every=max(1, int(args.progress_every)),
            tasks_per_child=max(1, int(args.tasks_per_child)),
            recompute=args.recompute,
        )
        qv_profiles[experiment] = load_ddh_qv_profile(
            args.ddh_agg_dir,
            experiment,
            lead_text_value=ddh_lead,
            block=args.qv_block,
        )

    return dsd_profiles, qv_profiles, lead_tag, ddh_lead


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dsd_profiles, qv_profiles, lead_tag, ddh_lead = build_profiles(args)

    output_path = args.output_dir / FIGURE_NAME
    text_path = args.data_txt_dir / TEXT_NAME
    plot_profiles(
        output_path,
        dsd_profiles,
        qv_profiles,
        args.experiments,
        qv_label=args.qv_label,
        max_height_km=args.max_height_km,
        dpi=args.dpi,
    )
    write_data_txt(
        text_path,
        dsd_profiles,
        qv_profiles,
        args.experiments,
        lead_tag=lead_tag,
        ddh_lead=ddh_lead,
        onemom_closure=args.onemom_closure,
        twomom_closure=args.twomom_closure,
        qv_block=args.qv_block,
    )
    print(f"Saved figure: {output_path}", flush=True)
    print(f"Saved data:   {text_path}", flush=True)


if __name__ == "__main__":
    main()
