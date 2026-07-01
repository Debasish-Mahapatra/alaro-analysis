#!/usr/bin/env python3
"""CFAD (frequency-by-altitude) of the rain mass-weighted diameter Dm inside updrafts.

Per grid cell (radar mask, all 87 levels) in a strong updraft
(-UD_OMEGA >= threshold, UD_MESH_FRAC > min) and rainy (qr > min_qr), the
per-cell Dm is computed with the truncated-moment formulation (paper plot 10):

* C1M / G1M (Abel-Boutle, mu = 0):  lambda = (rho*qr/(pi*220))**(-1/1.8)
* G2M (2-moment, ALARO variable mu): mu from the tanh(dmeanr) law,
  lambda = (Nt*gamma(mu+4)/(M3*gamma(mu+1)))**(1/3)
* Dm = (M4/M3) * cut(4)/cut(3) = (mu+4)/lambda * cut(4)/cut(3),
  cut_gamma truncation at the disdrometer minimum diameter (0.312 mm).

Per level the Dm values are histogrammed; each level row is normalised to
100 % ("frequency per level"), matching the dsd_cfad family style. One panel
per experiment, white/black dashed 0 C line. Histogram accumulators are cached
per experiment so replotting is instant.
"""
from __future__ import annotations

import argparse
import math
from multiprocessing import get_context
from pathlib import Path

import cmaps
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.special import gammaincc, gammaln

from alaro_analysis.common.constants import (
    EXPERIMENT_LABELS,
    EXPERIMENTS,
    RD,
)
from alaro_analysis.common.dsd import DEFAULT_QC_DIAMETER_MIN_MM
from alaro_analysis.common.figio import strip_cbar_zeros
from alaro_analysis.workflows.disdrometer_dsd import (
    MASK_FILE,
    NETCDF_ROOT,
    build_domain_mask_from_netcdf,
)
from run_dsd_per_cell_moments_truncated import _parse_leads, discover_records

RUNS_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS")
OUTPUT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/11. dm profile in updrafts")
CACHE_DIR = RUNS_ROOT / "processed-data" / "dsd_updraft_dm_profile"
HEIGHT_CACHE = RUNS_ROOT / "processed-data" / "paper5_from_netcdf"
TEMP_CACHE = RUNS_ROOT / "processed-data" / "temperature" / "2years"
NLEV = 87
AB_PREFACTOR = math.pi * 220.0
MU_MIN, MU_MAX = 0.1, 50.0
FREEZING_K = 273.15

_WORKER_MASK: np.ndarray | None = None


def _init_worker(mask: np.ndarray) -> None:
    global _WORKER_MASK
    _WORKER_MASK = np.asarray(mask, dtype=bool)


def _read_levels(path: Path, var: str) -> np.ndarray:
    """(NLEV, n_mask_cells) float64 from a masked-netcdf file."""
    with Dataset(path) as ds:
        data = ds.variables[var]
        if data.ndim == 4:
            field = np.asarray(data[0], dtype=np.float64)
        elif data.ndim == 3:
            field = np.asarray(data[:], dtype=np.float64)
        else:
            raise ValueError(f"Unexpected ndim for {var} in {path}: {data.ndim}")
    return field[:, _WORKER_MASK]


def _cut(k: float, lamb_per_m: np.ndarray, dmin_m: float) -> np.ndarray:
    return gammaincc(k + 1.0, lamb_per_m * dmin_m)


def _process_task(
    task: tuple[str, dict[str, str], float, float, float, float, float, float, int],
) -> tuple[np.ndarray, list[str]]:
    (experiment, paths, min_qr, dmin_m, min_pa_s, min_mesh,
     dm_lo, dm_hi, nbins) = task
    hist = np.zeros((NLEV, nbins), dtype=np.int64)
    upd_counts = np.zeros(NLEV, dtype=np.int64)
    warnings: list[str] = []
    try:
        qr = _read_levels(Path(paths["RAIN"]), "RAIN")
        temp = _read_levels(Path(paths["TEMPERATURE"]), "TEMPERATURE")
        pres = _read_levels(Path(paths["PRESSURE"]), "PRESSURE")
        omega = _read_levels(Path(paths["UD_OMEGA"]), "UD_OMEGA")
        mesh = _read_levels(Path(paths["UD_MESH_FRAC"]), "UD_MESH_FRAC")
        pnr = _read_levels(Path(paths["PNR"]), "PNR") if experiment == "2mom" else None
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"WARNING {experiment} {paths['RAIN']}: {exc}")
        return hist, upd_counts, warnings

    nlev = min(qr.shape[0], NLEV)
    rho = pres / (RD * temp)
    updraft = (
        np.isfinite(omega) & np.isfinite(mesh)
        & (mesh > min_mesh) & ((-omega) >= min_pa_s)
    )
    upd_counts[:nlev] = updraft[:nlev].sum(axis=1)
    keep = (
        updraft
        & np.isfinite(qr) & (qr > min_qr)
        & np.isfinite(rho) & (rho > 0.0)
    )
    if pnr is not None:
        keep &= np.isfinite(pnr) & (pnr > 0.0)

    dbin = (dm_hi - dm_lo) / nbins
    for lev in range(nlev):
        sel = keep[lev]
        if not sel.any():
            continue
        q = qr[lev][sel]
        r = rho[lev][sel]
        if experiment == "2mom":
            nt = pnr[lev][sel] * r
            m3 = 6.0 * q * r / (1000.0 * math.pi)
            dmean_mm = 1.0e3 * (m3 / nt) ** (1.0 / 3.0)
            mu = np.clip(19.0 * np.tanh(0.6 * (dmean_mm - 1.8)) + 17.0, MU_MIN, MU_MAX)
            lamb = (nt * np.exp(gammaln(mu + 4.0) - gammaln(mu + 1.0)) / m3) ** (1.0 / 3.0)
        else:
            mu = 0.0
            lamb = (r * q / AB_PREFACTOR) ** (-1.0 / 1.8)
        dm_mm = 1000.0 * (np.asarray(mu) + 4.0) / lamb * (
            _cut(4.0, lamb, dmin_m) / _cut(3.0, lamb, dmin_m)
        )
        bi = np.floor((dm_mm - dm_lo) / dbin)
        good = np.isfinite(bi) & (bi >= 0) & (bi < nbins)
        if good.any():
            hist[lev] += np.bincount(bi[good].astype(np.int64), minlength=nbins)
    return hist, upd_counts, warnings


def gather(
    experiment: str,
    records,
    domain_mask,
    *,
    min_qr: float,
    dmin_m: float,
    min_pa_s: float,
    min_mesh: float,
    dm_lo: float,
    dm_hi: float,
    nbins: int,
    workers: int,
    progress_every: int,
) -> np.ndarray:
    tasks = [
        (experiment, {v: str(p) for v, p in rec[3].items()}, min_qr, dmin_m,
         min_pa_s, min_mesh, dm_lo, dm_hi, nbins)
        for rec in records
    ]
    print(f"  [{experiment}] {len(tasks):,} timesteps -> updraft Dm CFAD", flush=True)
    hist = np.zeros((NLEV, nbins), dtype=np.int64)
    upd_counts = np.zeros(NLEV, dtype=np.int64)
    pool = get_context("fork").Pool(
        processes=workers, initializer=_init_worker, initargs=(domain_mask.mask,),
        maxtasksperchild=128,
    )
    try:
        for idx, (h, u, warns) in enumerate(pool.imap_unordered(_process_task, tasks), 1):
            hist += h
            upd_counts += u
            for w in warns:
                print(w, flush=True)
            if idx % progress_every == 0 or idx == len(tasks):
                print(f"  [{experiment}] {idx}/{len(tasks)} (cells kept: {hist.sum():,})", flush=True)
    finally:
        pool.close()
        pool.join()
    return hist, upd_counts


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


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    mid = 0.5 * (v[1:] + v[:-1])
    return np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--lead", default="all")
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=2000)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--min-qr", type=float, default=1.0e-7)
    parser.add_argument("--dmin-mm", type=float, default=DEFAULT_QC_DIAMETER_MIN_MM)
    parser.add_argument("--min-updraft-pa-s", type=float, default=10.0)
    parser.add_argument("--min-mesh-frac", type=float, default=0.0)
    parser.add_argument("--dm-range", type=float, nargs=2, default=(0.0, 6.0),
                        help="Dm histogram range (mm); cells outside are dropped.")
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--min-samples", type=int, default=50,
                        help="Mask levels with fewer kept cells than this.")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Cap of the frequency colour scale (%%); default = data max.")
    parser.add_argument("--max-height-km", type=float, default=9.0)
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
    dmin_m = args.dmin_mm * 1.0e-3
    dm_lo, dm_hi = args.dm_range
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"cfad2_omega{args.min_updraft_pa_s:g}_mesh{args.min_mesh_frac:g}"
           f"_dmin{args.dmin_mm:g}_dm{dm_lo:g}-{dm_hi:g}_b{args.bins}")
    x_edges = np.linspace(dm_lo, dm_hi, args.bins + 1)

    hists: dict[str, np.ndarray] = {}
    upd_counts: dict[str, np.ndarray] = {}
    domain_mask = None
    for exp in args.experiments:
        cache = args.cache_dir / f"{exp}_dm_updraft_{tag}.npz"
        if cache.exists() and not args.recompute and args.max_days is None:
            with np.load(cache) as d:
                hists[exp] = d["hist"]
                upd_counts[exp] = d["upd_counts"]
            print(f"  [{exp}] using cache {cache}", flush=True)
            continue
        records = discover_records(exp, leads, args.netcdf_root, args.max_days)
        base = args.netcdf_root / exp / "masked-netcdf"
        recs = []
        for valid, init, lead, paths in records:
            extra = {}
            ok = True
            for var in ("UD_OMEGA", "UD_MESH_FRAC"):
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
        hist, upd = gather(
            exp, recs, domain_mask,
            min_qr=args.min_qr, dmin_m=dmin_m,
            min_pa_s=args.min_updraft_pa_s, min_mesh=args.min_mesh_frac,
            dm_lo=dm_lo, dm_hi=dm_hi, nbins=args.bins,
            workers=max(1, int(args.workers)), progress_every=args.progress_every,
        )
        hists[exp] = hist
        upd_counts[exp] = upd
        if args.max_days is None:
            np.savez(cache, hist=hist, upd_counts=upd, x_edges=x_edges)
            print(f"  [{exp}] cached -> {cache}", flush=True)

    # ----- two normalisations -----
    # v1: each level row normalised over rainy in-updraft cells (classic CFAD).
    # v2: normalised over ALL updraft cells, so rain-free cells dilute the
    #     colours and scarcity aloft shows up as fading.
    versions = []
    for suffix, norm in (("", "rainy"), ("_v2", "updraft")):
        freqs: dict[str, np.ndarray] = {}
        vmax = 0.0
        for exp in args.experiments:
            hist = hists[exp].astype(np.float64)
            if norm == "rainy":
                denom = hist.sum(axis=1, keepdims=True)
            else:
                denom = upd_counts[exp].astype(np.float64)[:, None]
            freq = np.where(denom >= args.min_samples, 100.0 * hist / np.maximum(denom, 1.0), np.nan)
            freqs[exp] = freq
            if np.isfinite(freq).any():
                vmax = max(vmax, float(np.nanmax(freq)))
        if suffix == "" and args.vmax is not None:
            vmax = float(args.vmax)
        versions.append((suffix, norm, freqs, vmax))

    letters = "abcdefg"
    for suffix, norm, freqs, vmax in versions:
        fig, axes = plt.subplots(
            1, len(args.experiments), figsize=(5.4 * len(args.experiments), 4.6),
            sharey=True, constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        last_im = None
        for col, exp in enumerate(args.experiments):
            ax = axes[col]
            z = load_height_km(exp)[:NLEV]
            order = np.argsort(z)
            y_edges = centers_to_edges(z[order])
            freq = np.ma.masked_invalid(freqs[exp][order, :])
            last_im = ax.pcolormesh(
                x_edges, y_edges, freq, cmap=cmaps.WhViBlGrYeOrRe,
                shading="auto", vmin=0.0, vmax=vmax,
            )
            freeze = mean_freezing_km(exp)
            if np.isfinite(freeze):
                ax.axhline(freeze, color="white", linestyle="--", linewidth=1.4, alpha=0.95)
                ax.axhline(freeze, color="black", linestyle="--", linewidth=0.7, alpha=0.75)
            ax.set_title(EXPERIMENT_LABELS.get(exp, exp), fontsize=13, pad=10)
            ax.text(0.03, 0.96, f"({letters[col]})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=12, fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                          "edgecolor": "none", "alpha": 0.78})
            if col == 0:
                ax.set_ylabel("Height (km)", fontsize=12)
            ax.set_xlabel(r"D$_m$ = M$_4$/M$_3$ (mm)", fontsize=12)
            ax.set_ylim(0.0, args.max_height_km)
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(False)

        cbar = fig.colorbar(last_im, ax=axes, shrink=0.84, pad=0.012)
        strip_cbar_zeros(cbar)
        if norm == "rainy":
            cbar.set_label("Frequency per level (%)", fontsize=11)
        else:
            cbar.set_label("Frequency per level (% of updraft cells)", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

        out = args.output_dir / f"11. dm profile in updrafts{suffix}_450dpi.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[saved] {out}")

        txt = args.output_dir / f"11. dm profile in updrafts{suffix}_data.txt"
        with txt.open("w", encoding="utf-8") as fh:
            fh.write("CFAD of rain Dm inside updrafts (radar mask, all hourly files)\n")
            fh.write(f"updraft: -UD_OMEGA >= {args.min_updraft_pa_s:g} Pa/s, "
                     f"UD_MESH_FRAC > {args.min_mesh_frac:g}; rainy: qr > {args.min_qr:g}\n")
            fh.write("Dm = (mu+4)/lambda * cut(4)/cut(3), truncated-moment formulation "
                     f"(dmin = {args.dmin_mm:g} mm); histogram range {dm_lo:g}-{dm_hi:g} mm, "
                     f"{args.bins} bins; rows with denominator < {args.min_samples} masked.\n")
            if norm == "rainy":
                fh.write("Normalisation: each level row sums to 100% over RAINY updraft cells.\n")
            else:
                fh.write("Normalisation: percentages of ALL updraft cells at that level "
                         "(rain-free cells included in the denominator).\n")
            fh.write("# x_edges_mm\n" + ",".join(f"{v:.6g}" for v in x_edges) + "\n")
            for exp in args.experiments:
                z = load_height_km(exp)[:NLEV]
                fh.write(f"\n=== {EXPERIMENT_LABELS[exp]} (0C mean {mean_freezing_km(exp):.3f} km) ===\n")
                fh.write("height_km,n_rainy,n_updraft," + ",".join(
                    f"f{0.5 * (x_edges[i] + x_edges[i + 1]):.3f}" for i in range(args.bins)) + "\n")
                hist = hists[exp]
                freq = freqs[exp]
                for lev in range(NLEV):
                    row = ",".join("nan" if not np.isfinite(v) else f"{v:.5g}" for v in freq[lev])
                    fh.write(f"{z[lev]:.4f},{int(hist[lev].sum())},{int(upd_counts[exp][lev])},{row}\n")
        print(f"[saved] {txt}")


if __name__ == "__main__":
    main()
