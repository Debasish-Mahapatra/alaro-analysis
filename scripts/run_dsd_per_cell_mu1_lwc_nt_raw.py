"""Raw masked-netCDF per-cell DSD sampling with mu=1 normalized gamma.

This is the strict rerun variant for the fixed-shape comparison.  It reads the
masked model netCDF fields directly, computes per-cell LWC and Nt, then fits
all model experiments to a normalized gamma distribution with ``mu=1``.

For C1M/G1M, Nt is diagnosed from the native one-moment rain closure
(``abel_boutle`` by default).  For G2M, Nt comes from prognostic PNR.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from alaro_analysis.common.constants import EXPERIMENTS, RD
from alaro_analysis.common.dsd import (
    mp_from_q_abel_boutle,
    mp_from_q_fixed_n0,
    normalized_gamma_diagnostics_from_lwc_nt_mu,
)
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


def _empty_samples() -> dict[str, np.ndarray]:
    out = {k: np.empty(0, dtype=np.float32) for k in OBS_PARAMETERS}
    out["mu"] = np.empty(0, dtype=np.float32)
    return out


def _normalised_gamma_from_lwc_nt(
    lwc_g_m3: np.ndarray,
    nt_m3: np.ndarray,
    *,
    d0_min_mm: float,
) -> dict[str, np.ndarray]:
    diag = normalized_gamma_diagnostics_from_lwc_nt_mu(lwc_g_m3, nt_m3, 1.0)
    keep = (
        np.isfinite(diag["dm_mm"]) & (diag["dm_mm"] > 0.0)
        & np.isfinite(diag["d0_mm"]) & (diag["d0_mm"] >= d0_min_mm)
        & np.isfinite(diag["sigma_m_mm"])
        & np.isfinite(diag["log_nw"])
        & np.isfinite(diag["lwc_g_m3"]) & (diag["lwc_g_m3"] > 0.0)
        & np.isfinite(diag["nt_m3"]) & (diag["nt_m3"] > 0.0)
    )
    if not keep.any():
        return _empty_samples()
    keys = (*OBS_PARAMETERS, "mu")
    return {key: diag[key][keep].astype(np.float32) for key in keys}


def _process_task(
    task: tuple[str, dict[str, str], float, str, float, float],
) -> tuple[dict[str, np.ndarray], list[str]]:
    experiment, paths, min_qr, onemom_number_source, n0_fixed, d0_min_mm = task
    warnings: list[str] = []
    try:
        qr = _read_masked_field(Path(paths["RAIN"]), "RAIN")
        temp = _read_masked_field(Path(paths["TEMPERATURE"]), "TEMPERATURE")
        pres = _read_masked_field(Path(paths["PRESSURE"]), "PRESSURE")
        pnr = _read_masked_field(Path(paths["PNR"]), "PNR") if experiment == "2mom" else None
    except Exception as exc:
        warnings.append(f"WARNING {experiment} {paths['RAIN']}: {exc}")
        return _empty_samples(), warnings

    rainy = np.isfinite(qr) & np.isfinite(temp) & np.isfinite(pres) & (qr > min_qr)
    rho = pres / (RD * temp)
    rainy &= np.isfinite(rho) & (rho > 0.0)
    if pnr is not None:
        rainy &= np.isfinite(pnr) & (pnr > 0.0)
    if not rainy.any():
        return _empty_samples(), warnings

    qr_v = qr[rainy]
    rho_v = rho[rainy]
    lwc_g_m3 = 1000.0 * rho_v * qr_v

    if experiment == "2mom":
        nt_m3 = rho_v * pnr[rainy]
    elif onemom_number_source == "abel_boutle":
        native = mp_from_q_abel_boutle(qr_v, rho_v)
        nt_m3 = native["nt_m3"]
    else:
        native = mp_from_q_fixed_n0(qr_v, rho_v, n0_per_m3_mm=n0_fixed)
        nt_m3 = native["nt_m3"]

    return _normalised_gamma_from_lwc_nt(lwc_g_m3, nt_m3, d0_min_mm=d0_min_mm), warnings


def gather_experiment(
    experiment: str,
    records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]],
    domain_mask,
    *,
    min_qr: float,
    onemom_number_source: str,
    n0_fixed: float,
    d0_min_mm: float,
    workers: int,
    progress_every: int,
    tasks_per_child: int,
) -> dict[str, np.ndarray]:
    if not records:
        return _empty_samples()
    tasks = [
        (
            experiment,
            {var: str(path) for var, path in rec[3].items()},
            min_qr,
            onemom_number_source,
            n0_fixed,
            d0_min_mm,
        )
        for rec in records
    ]
    print(f"  [{experiment}] processing {len(tasks):,} timesteps -> raw per-cell mu=1", flush=True)
    keys = (*OBS_PARAMETERS, "mu")
    accumulators: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    if workers <= 1:
        _init_worker(domain_mask.mask)
        iterator = (_process_task(task) for task in tasks)
    else:
        pool = get_context("fork").Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(domain_mask.mask,),
            maxtasksperchild=tasks_per_child,
        )
        iterator = pool.imap_unordered(_process_task, tasks)
    try:
        for idx, (samples, warnings) in enumerate(iterator, 1):
            for key in keys:
                accumulators[key].append(samples[key])
            for warning in warnings:
                print(warning, flush=True)
            if idx % progress_every == 0 or idx == len(tasks):
                running = sum(a.size for a in accumulators["dm_mm"])
                print(f"  [{experiment}] processed {idx}/{len(tasks)} (samples so far: {running:,})", flush=True)
    finally:
        if workers > 1:
            pool.close()
            pool.join()
    return {
        key: (np.concatenate(parts).astype(np.float32) if parts else np.empty(0, dtype=np.float32))
        for key, parts in accumulators.items()
    }


def _load_obs_mu1(obs_samples_npz: Path, *, d0_min_mm: float) -> dict[str, np.ndarray]:
    with np.load(obs_samples_npz) as npz:
        lwc = np.asarray(npz["obs_pathA__lwc_g_m3"], dtype=float)
        nt = np.asarray(npz["obs_pathA__nt_m3"], dtype=float)
    return _normalised_gamma_from_lwc_nt(lwc, nt, d0_min_mm=d0_min_mm)


def _parse_leads(text: str) -> tuple[int, ...] | None:
    if text.lower() in {"all", "*"}:
        return None
    leads: list[int] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = piece.split("-", 1)
            leads.extend(range(int(a), int(b) + 1))
        else:
            leads.append(int(piece))
    return tuple(sorted(set(leads)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raw per-cell mu=1 normalized-gamma DSD comparison")
    parser.add_argument("--lead", default="all")
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--tasks-per-child", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--min-qr", type=float, default=1.0e-7)
    parser.add_argument("--d0-min-mm", type=float, default=0.2)
    parser.add_argument("--onemom-number-source", choices=("abel_boutle", "fixed_n0"), default="abel_boutle")
    parser.add_argument("--n0-fixed", type=float, default=8000.0)
    parser.add_argument("--netcdf-root", type=Path, default=NETCDF_ROOT)
    parser.add_argument("--mask-file", type=Path, default=MASK_FILE)
    parser.add_argument("--mask-var", default=None)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR / "normalized_gamma_percell_mu1_lwc_nt_raw")
    parser.add_argument(
        "--obs-samples-npz",
        type=Path,
        default=PROCESSED_DIR / "disdrometer_dsd_samples_all_leads.npz",
    )
    parser.add_argument("--output-tag", default="all_leads")
    parser.add_argument("--bins", type=int, default=60)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    leads = _parse_leads(args.lead)

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
        samples = gather_experiment(
            exp,
            records,
            domain_mask,
            min_qr=args.min_qr,
            onemom_number_source=args.onemom_number_source,
            n0_fixed=args.n0_fixed,
            d0_min_mm=args.d0_min_mm,
            workers=max(1, int(args.workers)),
            progress_every=args.progress_every,
            tasks_per_child=args.tasks_per_child,
        )
        print(f"  [{exp}] kept {samples['dm_mm'].size:,} raw per-cell mu=1 samples", flush=True)
        samples_per_exp[exp] = samples

    print(f"loading obs LWC+Nt from {args.obs_samples_npz}", flush=True)
    obs = _load_obs_mu1(args.obs_samples_npz, d0_min_mm=args.d0_min_mm)
    print(f"  [obs] kept {obs['dm_mm'].size:,} mu=1 samples", flush=True)

    out_npz = args.processed_dir / f"disdrometer_dsd_percell_mu1_lwc_nt_raw_{args.output_tag}.npz"
    payload: dict[str, np.ndarray] = {f"obs__{key}": value for key, value in obs.items()}
    for exp, samples in samples_per_exp.items():
        payload.update({f"{exp}__{key}": value for key, value in samples.items()})
    np.savez_compressed(out_npz, **payload)
    print(f"samples NPZ -> {out_npz}", flush=True)

    samples_all = {"obs": obs, **samples_per_exp}
    for x_field, x_label, suffix in (
        ("d0_mm", "D$_0$ (mm)", "logNw_D0"),
        ("dm_mm", "D$_m$ (mm)", "logNw_Dm"),
    ):
        out = args.figure_dir / f"dsd_percell_mu1_lwc_nt_raw_{suffix}_{args.output_tag}.png"
        title = (
            f"log$_{{10}}$ N$_w$ vs {'D$_0$' if x_field == 'd0_mm' else 'D$_m$'}, "
            "mu=1 normalized gamma from raw per-cell fields"
        )
        plot_2x2_with_marginals(
            out_path=out,
            samples=samples_all,
            x_field=x_field,
            x_label=x_label,
            title=title,
            bins=args.bins,
        )
        print(f"rendered {out}", flush=True)


if __name__ == "__main__":
    main()
