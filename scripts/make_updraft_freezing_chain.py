#!/usr/bin/env python3
"""In-updraft freezing-chain diagnostic around the 0 C level.

Tests, per experiment, the mechanism "updrafts loft many tiny rain drops into
the freezing layer, the drops freeze (latent heat of fusion), and the updraft
warms":

  (a) rain drop number concentration Nt carried by updrafts vs height
      (G2M: prognostic PNR; C1M/G1M: Abel-Boutle closure Nt = M0(qr,rho)),
  (b) mean (volume) drop diameter dmeanr = (6 qr rho /(1000 pi Nt))**(1/3),
  (c) in-updraft rain qr vs snow+graupel qs+qg (the mass hand-off at 0 C),
  (d) updraft temperature excess dT = <T|updraft> - <T|all cells> per level.

All statistics are per-level means inside strong updrafts (-UD_OMEGA >=
threshold, UD_MESH_FRAC > min) over the radar mask, all hourly files;
(a)/(b) additionally require rainy cells (qr > min_qr). Accumulators are
cached per experiment.
"""
from __future__ import annotations

import argparse
import math
from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

from alaro_analysis.common.constants import (
    EXPERIMENT_COLORS,
    EXPERIMENT_LABELS,
    EXPERIMENTS,
    RD,
)
from alaro_analysis.workflows.disdrometer_dsd import (
    MASK_FILE,
    NETCDF_ROOT,
    build_domain_mask_from_netcdf,
)
from run_dsd_per_cell_moments_truncated import _parse_leads, discover_records

RUNS_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS")
OUTPUT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/12. updraft freezing chain")
CACHE_DIR = RUNS_ROOT / "processed-data" / "updraft_freezing_chain"
HEIGHT_CACHE = RUNS_ROOT / "processed-data" / "paper5_from_netcdf"
TEMP_CACHE = RUNS_ROOT / "processed-data" / "temperature" / "2years"
NLEV = 87
AB_PREFACTOR = math.pi * 220.0
FREEZING_K = 273.15

_WORKER_MASK: np.ndarray | None = None


def _init_worker(mask: np.ndarray) -> None:
    global _WORKER_MASK
    _WORKER_MASK = np.asarray(mask, dtype=bool)


def _read_levels(path: Path, var: str) -> np.ndarray:
    with Dataset(path) as ds:
        data = ds.variables[var]
        if data.ndim == 4:
            field = np.asarray(data[0], dtype=np.float64)
        elif data.ndim == 3:
            field = np.asarray(data[:], dtype=np.float64)
        else:
            raise ValueError(f"Unexpected ndim for {var} in {path}: {data.ndim}")
    return field[:, _WORKER_MASK]


KEYS = (
    "n_all", "sum_T_all",          # every radar-mask cell
    "n_up", "sum_T_up",            # updraft cells
    "sum_qr_up", "sum_qsg_up",     # mean hydrometeor contents over updraft cells
    "n_rain", "sum_nt", "sum_dmean",  # rainy updraft cells (DSD quantities)
)


def _empty() -> dict[str, np.ndarray]:
    return {k: np.zeros(NLEV) for k in KEYS}


def _process_task(
    task: tuple[str, dict[str, str], float, float, float],
) -> tuple[dict[str, np.ndarray], list[str]]:
    experiment, paths, min_qr, min_pa_s, min_mesh = task
    acc = _empty()
    warnings: list[str] = []
    try:
        qr = _read_levels(Path(paths["RAIN"]), "RAIN")
        temp = _read_levels(Path(paths["TEMPERATURE"]), "TEMPERATURE")
        pres = _read_levels(Path(paths["PRESSURE"]), "PRESSURE")
        omega = _read_levels(Path(paths["UD_OMEGA"]), "UD_OMEGA")
        mesh = _read_levels(Path(paths["UD_MESH_FRAC"]), "UD_MESH_FRAC")
        snow = _read_levels(Path(paths["SNOW"]), "SNOW")
        graup = (_read_levels(Path(paths["GRAUPEL"]), "GRAUPEL")
                 if "GRAUPEL" in paths else None)
        pnr = _read_levels(Path(paths["PNR"]), "PNR") if experiment == "2mom" else None
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"WARNING {experiment} {paths['RAIN']}: {exc}")
        return acc, warnings

    nlev = min(qr.shape[0], NLEV)
    rho = pres / (RD * temp)
    updraft = (
        np.isfinite(omega) & np.isfinite(mesh)
        & (mesh > min_mesh) & ((-omega) >= min_pa_s)
    )
    rainy = updraft & np.isfinite(qr) & (qr > min_qr) & np.isfinite(rho) & (rho > 0.0)
    if pnr is not None:
        rainy &= np.isfinite(pnr) & (pnr > 0.0)

    qsg = snow if graup is None else snow + graup
    t_ok = np.isfinite(temp)
    for lev in range(nlev):
        tl = temp[lev]
        ok = t_ok[lev]
        acc["n_all"][lev] += ok.sum()
        acc["sum_T_all"][lev] += tl[ok].sum()
        up = updraft[lev] & ok
        acc["n_up"][lev] += up.sum()
        acc["sum_T_up"][lev] += tl[up].sum()
        if up.any():
            q_up = np.where(np.isfinite(qr[lev]), qr[lev], 0.0)
            g_up = np.where(np.isfinite(qsg[lev]), qsg[lev], 0.0)
            acc["sum_qr_up"][lev] += q_up[up].sum()
            acc["sum_qsg_up"][lev] += g_up[up].sum()
        rn = rainy[lev]
        if rn.any():
            q = qr[lev][rn]
            r = rho[lev][rn]
            if experiment == "2mom":
                nt = pnr[lev][rn] * r
            else:
                lamb = (r * q / AB_PREFACTOR) ** (-1.0 / 1.8)   # 1/m
                nt = 0.22 * lamb**1.2                            # AB closure M0
            dmean_mm = 1.0e3 * (6.0 * q * r / (1000.0 * math.pi * nt)) ** (1.0 / 3.0)
            # cap at the disdrometer QC maximum: near-zero PNR cells otherwise
            # produce unphysical dmean that poisons the level mean.
            good = (np.isfinite(nt) & (nt > 0.0)
                    & np.isfinite(dmean_mm) & (dmean_mm <= 8.0))
            acc["n_rain"][lev] += good.sum()
            acc["sum_nt"][lev] += nt[good].sum()
            acc["sum_dmean"][lev] += dmean_mm[good].sum()
    return acc, warnings


def gather(experiment, records, domain_mask, *, min_qr, min_pa_s, min_mesh,
           workers, progress_every) -> dict[str, np.ndarray]:
    tasks = [
        (experiment, {v: str(p) for v, p in rec[3].items()}, min_qr, min_pa_s, min_mesh)
        for rec in records
    ]
    print(f"  [{experiment}] {len(tasks):,} timesteps -> freezing-chain profiles", flush=True)
    total = _empty()
    pool = get_context("fork").Pool(
        processes=workers, initializer=_init_worker, initargs=(domain_mask.mask,),
        maxtasksperchild=128,
    )
    try:
        for idx, (acc, warns) in enumerate(pool.imap_unordered(_process_task, tasks), 1):
            for k in KEYS:
                total[k] += acc[k]
            for w in warns:
                print(w, flush=True)
            if idx % progress_every == 0 or idx == len(tasks):
                print(f"  [{experiment}] {idx}/{len(tasks)}", flush=True)
    finally:
        pool.close()
        pool.join()
    return total


def load_height_km(exp: str) -> np.ndarray:
    with np.load(HEIGHT_CACHE / f"{exp}_height.npz") as d:
        return np.asarray(d["height_m"], dtype=np.float64) / 1000.0


def mean_freezing_km(exp: str) -> float:
    with np.load(TEMP_CACHE / f"{exp}_full-domain_diurnal_profile.npz") as d:
        temp = np.asarray(d["mean"], dtype=np.float64)
    z = load_height_km(exp)
    n = min(z.size, temp.shape[0])
    zz, tt = z[:n], temp[:n]
    levels = []
    for h in range(tt.shape[1]):
        col = tt[:, h]
        order = np.argsort(zz)
        zs, cs = zz[order], col[order]
        for i in range(zs.size - 1):
            if (cs[i] - FREEZING_K) * (cs[i + 1] - FREEZING_K) < 0.0:
                f = (FREEZING_K - cs[i]) / (cs[i + 1] - cs[i])
                levels.append(zs[i] + f * (zs[i + 1] - zs[i]))
                break
    return float(np.nanmean(levels)) if levels else np.nan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--lead", default="all")
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=2000)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--min-qr", type=float, default=1.0e-7)
    parser.add_argument("--min-updraft-pa-s", type=float, default=10.0)
    parser.add_argument("--min-mesh-frac", type=float, default=0.0)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--max-height-km", type=float, default=10.0)
    parser.add_argument("--netcdf-root", type=Path, default=NETCDF_ROOT)
    parser.add_argument("--mask-file", type=Path, default=MASK_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--recompute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    leads = _parse_leads(args.lead)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"omega{args.min_updraft_pa_s:g}_mesh{args.min_mesh_frac:g}_dcap8"

    data: dict[str, dict[str, np.ndarray]] = {}
    domain_mask = None
    for exp in args.experiments:
        cache = args.cache_dir / f"{exp}_chain_{tag}.npz"
        if cache.exists() and not args.recompute and args.max_days is None:
            with np.load(cache) as d:
                data[exp] = {k: d[k] for k in KEYS}
            print(f"  [{exp}] using cache {cache}", flush=True)
            continue
        records = discover_records(exp, leads, args.netcdf_root, args.max_days)
        base = args.netcdf_root / exp / "masked-netcdf"
        extra_vars = ["UD_OMEGA", "UD_MESH_FRAC", "SNOW"]
        if exp != "control":
            extra_vars.append("GRAUPEL")  # control is the 2-ice scheme: no graupel
        recs = []
        for valid, init, lead, paths in records:
            extra = {}
            ok = True
            for var in extra_vars:
                cand = base / var / Path(paths["RAIN"]).parent.name / Path(paths["RAIN"]).name
                if not cand.exists():
                    ok = False
                    break
                extra[var] = cand
            if ok:
                recs.append((valid, init, lead, {**paths, **extra}))
        if domain_mask is None:
            domain_mask = build_domain_mask_from_netcdf(recs[0][3]["RAIN"], args.mask_file)
            print(f"  mask cells: {domain_mask.n_cells}", flush=True)
        data[exp] = gather(
            exp, recs, domain_mask,
            min_qr=args.min_qr, min_pa_s=args.min_updraft_pa_s, min_mesh=args.min_mesh_frac,
            workers=max(1, int(args.workers)), progress_every=args.progress_every,
        )
        if args.max_days is None:
            np.savez(cache, **data[exp])
            print(f"  [{exp}] cached -> {cache}", flush=True)

    # ----- profiles -----
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 6.4), sharey=True)
    letters = ["(a)", "(b)", "(c)", "(d)"]
    txt_blocks: dict[str, dict[str, np.ndarray]] = {}
    for exp in args.experiments:
        d = data[exp]
        z = load_height_km(exp)[:NLEV]
        order = np.argsort(z)

        def prof(sum_key: str, cnt_key: str) -> np.ndarray:
            cnt = d[cnt_key]
            out = np.full(NLEV, np.nan)
            ok = cnt >= args.min_samples
            out[ok] = d[sum_key][ok] / cnt[ok]
            return out[order]

        zs = z[order]
        nt = prof("sum_nt", "n_rain")
        dmean = prof("sum_dmean", "n_rain")
        qr_up = prof("sum_qr_up", "n_up") * 1.0e3      # g/kg
        qsg_up = prof("sum_qsg_up", "n_up") * 1.0e3    # g/kg
        dT = prof("sum_T_up", "n_up") - prof("sum_T_all", "n_all")
        keep = (zs >= 0.0) & (zs <= args.max_height_km)
        color = EXPERIMENT_COLORS[exp]
        label = EXPERIMENT_LABELS[exp]

        axes[0].plot(nt[keep], zs[keep], color=color, lw=2.2, label=label)
        axes[1].plot(dmean[keep], zs[keep], color=color, lw=2.2, label=label)
        axes[2].plot(qr_up[keep], zs[keep], color=color, lw=2.2, label=f"{label} q$_r$")
        axes[2].plot(qsg_up[keep], zs[keep], color=color, lw=2.0, ls="--",
                     label=f"{label} q$_s$+q$_g$")
        axes[3].plot(dT[keep], zs[keep], color=color, lw=2.2, label=label)
        txt_blocks[exp] = {
            "z": zs, "nt": nt, "dmean": dmean, "qr": qr_up, "qsg": qsg_up, "dT": dT,
            "n_rain": d["n_rain"][order], "n_up": d["n_up"][order],
        }

    fl = np.nanmean([mean_freezing_km(exp) for exp in args.experiments])
    for i, ax in enumerate(axes):
        ax.axhline(fl, color="black", ls=":", lw=1.8,
                   label="0 °C level" if i == 0 else None)
        ax.grid(True, color="0.9", linewidth=0.8)
        ax.text(0.04, 0.97, letters[i], transform=ax.transAxes, ha="left", va="top",
                fontsize=14, fontweight="bold")
        ax.tick_params(labelsize=11)
    axes[3].axvline(0.0, color="0.4", lw=1.0)

    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"Rain drop number N$_t$ (m$^{-3}$)", fontsize=12)
    axes[0].set_ylabel("Height (km)", fontsize=12)
    axes[1].set_xlabel(r"Mean drop diameter d$_{mean,r}$ (mm)", fontsize=12)
    axes[2].set_xlabel(r"In-updraft content (g kg$^{-1}$)", fontsize=12)
    axes[3].set_xlabel(r"Updraft T excess  $\langle T\rangle_{up}-\langle T\rangle_{all}$ (K)", fontsize=12)
    axes[0].set_ylim(0.0, args.max_height_km)
    axes[0].legend(loc="upper right", fontsize=9, frameon=False)
    axes[2].legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()

    out = args.output_dir / "12. updraft freezing chain_450dpi.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")

    txt = args.output_dir / "12. updraft freezing chain_data.txt"
    with txt.open("w", encoding="utf-8") as fh:
        fh.write("In-updraft freezing-chain profiles (radar mask, all hourly files)\n")
        fh.write(f"updraft: -UD_OMEGA >= {args.min_updraft_pa_s:g} Pa/s, UD_MESH_FRAC > "
                 f"{args.min_mesh_frac:g}; DSD panels need qr > {args.min_qr:g}\n")
        fh.write("Nt: G2M = PNR*rho (prognostic); C1M/G1M = Abel-Boutle closure 0.22*lambda^1.2.\n")
        fh.write("dmean = (6 qr rho/(1000 pi Nt))^(1/3). dT = mean T in updrafts minus mean T all cells.\n")
        fh.write(f"levels with < {args.min_samples} samples masked. mean 0C = {fl:.3f} km\n")
        for exp in args.experiments:
            b = txt_blocks[exp]
            fh.write(f"\n=== {EXPERIMENT_LABELS[exp]} ===\n")
            fh.write("height_km,nt_m3,dmean_mm,qr_up_gkg,qsg_up_gkg,dT_K,n_rain,n_updraft\n")
            for i in range(NLEV):
                fh.write(f"{b['z'][i]:.4f},{b['nt'][i]:.6g},{b['dmean'][i]:.6g},"
                         f"{b['qr'][i]:.6g},{b['qsg'][i]:.6g},{b['dT'][i]:.6g},"
                         f"{int(b['n_rain'][i])},{int(b['n_up'][i])}\n")
    print(f"[saved] {txt}")


if __name__ == "__main__":
    main()
