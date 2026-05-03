"""Per-cell sampling variant of the disdrometer DSD comparison.

Reads the same masked netCDF outputs the domain-mean workflow uses, but instead
of averaging q_r, T, p (and PNR for 2mom) over the radar mask before applying
the DSD closure, it keeps each rainy cell as its own sample.  This lets us see
per-grid features like the breakup-induced "spike" discussed in Dolan et al.
2023 (Section 3.1, RAMS Figure 3 / 6) that domain averaging would smear out.

Same Marshall-Palmer closure as the domain-mean workflow:

* 2-moment (G2M): MP from prognostic ``q_r`` and ``PNR`` per cell.
* 1-moment (C1M, G1M): Abel & Boutle 2012 closure (``arpifs/adiab/gpprs0d.F90``)
  applied per cell.

Outputs four figures into ``figures/disdrometer_dsd/`` with the ``percell_``
prefix and a samples NPZ.  Reuses the existing obs reductions (Path A, Path B)
from the previous run's NPZ.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from alaro_analysis.common.constants import EXPERIMENTS, RD
from alaro_analysis.common.dsd import (
    gamma_dsd_from_q_n_per_kg,
    mp_from_q_abel_boutle,
    mp_from_q_fixed_n0,
)
from alaro_analysis.converter.pipeline import _regrid_mask_to_model
from alaro_analysis.workflows.disdrometer_dsd import (
    FIGURE_DIR,
    MASK_FILE,
    NETCDF_ROOT,
    OBS_PARAMETERS,
    PROCESSED_DIR,
    build_domain_mask_from_netcdf,
    plot_2x2_with_marginals,
)


PF_DAY_RE = re.compile(r"^pf(\d{8})$")
PF_FILE_RE = re.compile(r"^pfABOFABOF\+(\d{4})\.nc$")


def discover_records(
    experiment: str,
    leads: tuple[int, ...] | None,
    netcdf_root: Path,
    max_days: int | None,
) -> list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]]:
    base = netcdf_root / experiment / "masked-netcdf"
    needed = ["RAIN", "TEMPERATURE", "PRESSURE"]
    if experiment == "2mom":
        needed.append("PNR")
    ref_dir = base / "RAIN"
    day_dirs = sorted(d for d in ref_dir.iterdir() if d.is_dir() and PF_DAY_RE.match(d.name))
    if max_days is not None:
        day_dirs = day_dirs[:max_days]
    lead_set = set(leads) if leads is not None else None
    out: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]] = []
    for day_dir in day_dirs:
        m = PF_DAY_RE.match(day_dir.name)
        if not m:
            continue
        init_dt = datetime.strptime(m.group(1), "%Y%m%d")
        for path in sorted(day_dir.iterdir()):
            fm = PF_FILE_RE.match(path.name)
            if not fm:
                continue
            lead = int(fm.group(1))
            if lead_set is not None and lead not in lead_set:
                continue
            paths: dict[str, Path] = {}
            ok = True
            for var in needed:
                cand = base / var / day_dir.name / path.name
                if not cand.exists():
                    ok = False
                    break
                paths[var] = cand
            if not ok:
                continue
            valid_dt = init_dt + timedelta(hours=lead)
            out.append((np.datetime64(valid_dt, "s"), np.datetime64(init_dt, "s"), lead, paths))
    out.sort(key=lambda x: x[0])
    return out


_WORKER_MASK: np.ndarray | None = None


def _init_worker(mask: np.ndarray) -> None:
    global _WORKER_MASK
    _WORKER_MASK = np.asarray(mask, dtype=bool)


def _read_masked_field(path: Path, var: str) -> np.ndarray:
    """Return the bottom-level field (level=0) cropped to the radar mask cells."""
    if _WORKER_MASK is None:
        raise RuntimeError("worker mask not initialised")
    with Dataset(path) as ds:
        data = ds.variables[var]
        if data.ndim == 4:
            field = np.asarray(data[0, 0], dtype=np.float64)
        elif data.ndim == 3:
            field = np.asarray(data[0], dtype=np.float64)
        elif data.ndim == 2:
            field = np.asarray(data[:], dtype=np.float64)
        else:
            raise ValueError(f"Unexpected ndim for {var} in {path}: {data.ndim}")
    return field[_WORKER_MASK]


def _process_task(
    task: tuple[str, dict[str, str], float, str, float, float],
) -> tuple[dict[str, np.ndarray], list[str]]:
    experiment, paths, min_qr, onemom_closure, n0_fixed, d0_min_mm = task
    warnings: list[str] = []
    empty = {k: np.empty(0, dtype=np.float32) for k in OBS_PARAMETERS}
    try:
        qr = _read_masked_field(Path(paths["RAIN"]), "RAIN")
        temp = _read_masked_field(Path(paths["TEMPERATURE"]), "TEMPERATURE")
        pres = _read_masked_field(Path(paths["PRESSURE"]), "PRESSURE")
        if experiment == "2mom":
            pnr = _read_masked_field(Path(paths["PNR"]), "PNR")
        else:
            pnr = None
    except Exception as exc:
        warnings.append(f"WARNING {experiment} {paths['RAIN']}: {exc}")
        return empty, warnings

    rainy = np.isfinite(qr) & np.isfinite(temp) & np.isfinite(pres) & (qr > min_qr)
    if pnr is not None:
        rainy &= np.isfinite(pnr) & (pnr > 0)
    if not rainy.any():
        return empty, warnings
    qr_v = qr[rainy]
    rho_v = pres[rainy] / (RD * temp[rainy])
    rho_v = np.where(np.isfinite(rho_v) & (rho_v > 0), rho_v, np.nan)

    if experiment == "2mom":
        # ALARO 2-mom: gamma DSD with native shape mu = 1 (ZSHAPER default).
        diag = gamma_dsd_from_q_n_per_kg(qr_v, pnr[rainy], rho_v, mu=1.0)
    elif onemom_closure == "abel_boutle":
        # ALARO 1-mom: Abel & Boutle 2012 exponential (mu = 0).
        diag = mp_from_q_abel_boutle(qr_v, rho_v)
    else:
        diag = mp_from_q_fixed_n0(qr_v, rho_v, n0_per_m3_mm=n0_fixed)

    keep = (
        np.isfinite(diag["dm_mm"]) & (diag["dm_mm"] > 0.0)
        & np.isfinite(diag["d0_mm"]) & (diag["d0_mm"] >= d0_min_mm)
        & np.isfinite(diag["sigma_m_mm"])
        & np.isfinite(diag["log_nw"])
        & np.isfinite(diag["lwc_g_m3"]) & (diag["lwc_g_m3"] > 0.0)
        & np.isfinite(diag["nt_m3"]) & (diag["nt_m3"] > 0.0)
    )
    if not keep.any():
        return empty, warnings
    return {
        "dm_mm": diag["dm_mm"][keep].astype(np.float32),
        "d0_mm": diag["d0_mm"][keep].astype(np.float32),
        "sigma_m_mm": diag["sigma_m_mm"][keep].astype(np.float32),
        "log_nw": diag["log_nw"][keep].astype(np.float32),
        "lwc_g_m3": diag["lwc_g_m3"][keep].astype(np.float32),
        "nt_m3": diag["nt_m3"][keep].astype(np.float32),
    }, warnings


def gather_experiment(
    experiment: str,
    records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]],
    domain_mask,
    *,
    min_qr: float,
    onemom_closure: str,
    n0_fixed: float,
    d0_min_mm: float,
    workers: int,
    progress_every: int,
    tasks_per_child: int,
) -> dict[str, np.ndarray]:
    if not records:
        return {k: np.empty(0, dtype=np.float32) for k in OBS_PARAMETERS}
    tasks = [
        (
            experiment,
            {var: str(path) for var, path in rec[3].items()},
            min_qr,
            onemom_closure,
            n0_fixed,
            d0_min_mm,
        )
        for rec in records
    ]
    print(f"  [{experiment}] processing {len(tasks):,} timesteps -> per-cell samples", flush=True)
    accumulators: dict[str, list[np.ndarray]] = {k: [] for k in OBS_PARAMETERS}
    if workers <= 1:
        _init_worker(domain_mask.mask)
        for idx, task in enumerate(tasks, 1):
            samples, warnings = _process_task(task)
            for k in OBS_PARAMETERS:
                accumulators[k].append(samples[k])
            for w in warnings:
                print(w, flush=True)
            if idx % progress_every == 0 or idx == len(tasks):
                running = sum(a.size for a in accumulators["dm_mm"])
                print(f"  [{experiment}] processed {idx}/{len(tasks)} (samples so far: {running:,})", flush=True)
    else:
        with get_context("fork").Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(domain_mask.mask,),
            maxtasksperchild=tasks_per_child,
        ) as pool:
            for idx, (samples, warnings) in enumerate(
                pool.imap_unordered(_process_task, tasks),
                1,
            ):
                for k in OBS_PARAMETERS:
                    accumulators[k].append(samples[k])
                for w in warnings:
                    print(w, flush=True)
                if idx % progress_every == 0 or idx == len(tasks):
                    running = sum(a.size for a in accumulators["dm_mm"])
                    print(f"  [{experiment}] processed {idx}/{len(tasks)} (samples so far: {running:,})", flush=True)
    out = {
        k: (np.concatenate(parts).astype(np.float32) if parts else np.empty(0, dtype=np.float32))
        for k, parts in accumulators.items()
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-cell sampling DSD comparison")
    parser.add_argument("--lead", default="all")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS),
        choices=list(EXPERIMENTS),
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--tasks-per-child", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--min-qr", type=float, default=1.0e-7)
    parser.add_argument("--d0-min-mm", type=float, default=0.2)
    parser.add_argument(
        "--onemom-closure",
        choices=("abel_boutle", "fixed_n0"),
        default="abel_boutle",
    )
    parser.add_argument("--n0-fixed", type=float, default=8000.0)
    parser.add_argument("--netcdf-root", type=Path, default=NETCDF_ROOT)
    parser.add_argument("--mask-file", type=Path, default=MASK_FILE)
    parser.add_argument("--mask-var", default=None)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument(
        "--obs-samples-npz",
        type=Path,
        default=PROCESSED_DIR / "disdrometer_dsd_samples_all_leads.npz",
        help="Existing samples NPZ to take obs Path A and Path B from.",
    )
    args = parser.parse_args()

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    if args.lead.lower() in {"all", "*"}:
        leads = None
    else:
        leads_list: list[int] = []
        for piece in args.lead.split(","):
            piece = piece.strip()
            if not piece:
                continue
            if "-" in piece:
                a, b = piece.split("-", 1)
                leads_list.extend(range(int(a), int(b) + 1))
            else:
                leads_list.append(int(piece))
        leads = tuple(sorted(set(leads_list)))

    sample_records = None
    for exp in args.experiments:
        recs = discover_records(exp, leads, args.netcdf_root, args.max_days)
        if recs:
            sample_records = recs
            break
    if not sample_records:
        sys.exit("No netCDF records found for any experiment")

    sample_path = sample_records[0][3]["RAIN"]
    print(f"building radar mask from {args.mask_file} (sample {sample_path})", flush=True)
    domain_mask = build_domain_mask_from_netcdf(
        sample_path,
        args.mask_file,
        mask_var=args.mask_var,
        mask_threshold=args.mask_threshold,
    )
    print(f"  mask cells kept: {domain_mask.n_cells} / {domain_mask.mask.size}", flush=True)

    samples_per_exp: dict[str, dict[str, np.ndarray]] = {}
    for exp in args.experiments:
        records = discover_records(exp, leads, args.netcdf_root, args.max_days)
        s = gather_experiment(
            exp, records, domain_mask,
            min_qr=args.min_qr,
            onemom_closure=args.onemom_closure,
            n0_fixed=args.n0_fixed,
            d0_min_mm=args.d0_min_mm,
            workers=max(1, int(args.workers)),
            progress_every=args.progress_every,
            tasks_per_child=args.tasks_per_child,
        )
        print(f"  [{exp}] kept {s['dm_mm'].size:,} per-cell samples", flush=True)
        samples_per_exp[exp] = s

    print(f"loading observation samples from {args.obs_samples_npz}", flush=True)
    obs_npz = np.load(args.obs_samples_npz)
    obs = {k: np.asarray(obs_npz[f"obs_pathA__{k}"]) for k in OBS_PARAMETERS}

    out_npz = args.processed_dir / "disdrometer_dsd_percell_samples_all_leads.npz"
    payload: dict[str, np.ndarray] = {f"obs__{k}": v for k, v in obs.items()}
    for exp, s in samples_per_exp.items():
        payload.update({f"{exp}__{k}": v for k, v in s.items()})
    np.savez_compressed(out_npz, **payload)
    print(f"samples NPZ -> {out_npz}", flush=True)

    samples_all = {"obs": obs, **samples_per_exp}

    for x_field, x_label, suffix in (
        ("d0_mm", "D$_0$ (mm)", "logNw_D0"),
        ("dm_mm", "D$_m$ (mm)", "logNw_Dm"),
    ):
        title = (
            f"log$_{{10}}$ N$_w$ vs {'D$_0$' if x_field == 'd0_mm' else 'D$_m$'}, "
            "normalised gamma, per-cell"
        )
        out = args.figure_dir / f"dsd_percell_{suffix}_all_leads.png"
        plot_2x2_with_marginals(
            out_path=out,
            samples=samples_all,
            x_field=x_field,
            x_label=x_label,
            title=title,
        )
        print(f"  rendered {out}", flush=True)


if __name__ == "__main__":
    main()
