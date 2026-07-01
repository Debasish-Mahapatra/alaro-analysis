"""Surface Dm vs logNw remake from the moment formulation with dmin truncation.

Model rain DSDs are built from the scheme-native moments and truncated at the
disdrometer's minimum detectable diameter via upper-incomplete-gamma factors,
so the model only "sees" the part of the spectrum the instrument sees:

* C1M / G1M (1-moment, Abel & Boutle 2012, mu = 0)::

      lambda = (rho*qr / (pi*220))**(-1/1.8)            # 1/m
      Mk     = 0.22 * gamma(k+1) * lambda**(2.2-1-k)

* G2M (2-moment, ALARO variable shape)::

      Nt     = nr * rho
      M3     = 6*qr*rho / (1000*pi)
      dmeanr = (M3/Nt)**(1/3)
      mu     = max(0.1, min(50, 19*tanh(0.6*(dmeanr*1e3 - 1.8)) + 17))
      lambda = (Nt*gamma(mu+4) / (M3*gamma(mu+1)))**(1/3)
      Mk     = Nt * gamma(mu+1+k)/gamma(mu+1) / lambda**k

* truncation (verbatim from the provided post-processing)::

      cut_gamma(k, mu, dmin, lambda) = exp(-x)            if k+mu == 0
                                     = gammaincc(k+1, x)  otherwise,  x = lambda*dmin

Plotted per rainy cell (bottom model level, radar mask):

      Dm     = M4/M3 * cut(4)/cut(3)              (mass-weighted diameter)
      LWC    = rho*qr * cut(3)
      Nt     = M0 * cut(0)
      log_nw = log10( (256e3/pi) * LWC[g/m3] / Dm[mm]^4 )

The disdrometer panel is unchanged (empirical path-A samples; the instrument
spectrum is already truncated at its own dmin). Default dmin = 0.312 mm, the
Parsivel QC minimum used for the observations in this repo.
"""
from __future__ import annotations

import argparse
import math
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.special import gammaincc, gammaln

from alaro_analysis.common.constants import EXPERIMENTS, RD
from alaro_analysis.common.dsd import (
    DEFAULT_QC_DIAMETER_MIN_MM,
    DEFAULT_QC_MIN_RUN_MINUTES,
    DEFAULT_QC_RAIN_RATE_MIN,
    DEFAULT_QC_REFLECTIVITY_MAX_DBZ,
    NW_PREFACTOR,
    normalized_gamma_from_empirical_samples,
)
from alaro_analysis.workflows.disdrometer_comparison import CACHE_DIR, OBS_ZIP
from alaro_analysis.workflows.disdrometer_dsd import (
    FIGURE_DIR,
    MASK_FILE,
    NETCDF_ROOT,
    OBS_PARAMETERS,
    PROCESSED_DIR,
    build_domain_mask_from_netcdf,
    plot_2x2_with_marginals,
    read_observation_samples,
)
from run_dsd_per_cell_normalized_gamma_raw_sets import (  # reuse, identical I/O
    _init_worker,
    _read_masked_field,
    _parse_leads,
    discover_records,
)

SAMPLE_KEYS = (*OBS_PARAMETERS, "mu")
AB_LAMBDA_PREFACTOR = math.pi * 220.0  # = 691.15; lambda = (rho*qr/691.15)**(-1/1.8)
MU_MIN, MU_MAX = 0.1, 50.0
FALL_B = 0.7706                 # w(D) = CF * D**FALL_B
FALL_CF = 654.5
CM = 1000.0 * math.pi / 6.0


def cut_gamma(k: float, mu: np.ndarray | float, dmin_m: float, lamb_per_m: np.ndarray) -> np.ndarray:
    """Truncated-moment factor, verbatim from the provided post-processing."""
    x = np.asarray(lamb_per_m, dtype=float) * dmin_m
    mu_b = np.broadcast_to(np.asarray(mu, dtype=float), x.shape)
    out = gammaincc(k + 1.0, x)
    zero = (k + mu_b) == 0.0
    if np.any(zero):
        out = np.where(zero, np.exp(-x), out)
    return out


def _empty_samples() -> dict[str, np.ndarray]:
    return {key: np.empty(0, dtype=np.float32) for key in SAMPLE_KEYS}


def _diag_from_moments(
    qr: np.ndarray,
    rho: np.ndarray,
    m0: np.ndarray,
    m3: np.ndarray,
    m3b: np.ndarray,
    m4: np.ndarray,
    m6: np.ndarray,
    mu: np.ndarray,
    lamb_per_m: np.ndarray,
    dmin_m: float,
    min_rr_mm_h: float,
    max_z_dbz: float,
) -> dict[str, np.ndarray]:
    """Truncated Dm/LWC/Nt/logNw samples from absolute SI moments.

    Cells are kept only if the truncated spectrum would pass the same QC the
    disdrometer applies (rain rate >= min_rr, reflectivity <= max dBZ), using
    pfplsl and zrefl from the provided post-processing.
    """
    q0 = cut_gamma(0.0, mu, dmin_m, lamb_per_m)
    q3 = cut_gamma(3.0, mu, dmin_m, lamb_per_m)
    q4 = cut_gamma(4.0, mu, dmin_m, lamb_per_m)
    q3b = cut_gamma(3.0 + FALL_B, mu, dmin_m, lamb_per_m)
    q6 = cut_gamma(6.0, mu, dmin_m, lamb_per_m)

    with np.errstate(divide="ignore", invalid="ignore"):
        dm_mm = 1000.0 * (m4 / m3) * (q4 / q3)          # zdmassw
        lwc_g_m3 = 1000.0 * rho * qr * q3               # zqrmp in g/m^3
        nt_m3 = m0 * q0                                 # pnrmp
        nw = NW_PREFACTOR * lwc_g_m3 / dm_mm**4
        log_nw = np.log10(nw)
        d0_mm = 1000.0 * (3.67 + mu) / lamb_per_m       # untruncated analytic (filter only)
        sigma_m_mm = dm_mm / np.sqrt(mu + 4.0)          # filter only
        rain_rate_mm_h = 3600.0 * m3b * CM * FALL_CF * q3b   # pfplsl
        refl_dbz = 10.0 * np.log10(m6 * q6 * 1.0e18)         # zrefl

    keep = (
        np.isfinite(dm_mm) & (dm_mm > 0.0)
        & np.isfinite(log_nw)
        & np.isfinite(lwc_g_m3) & (lwc_g_m3 > 0.0)
        & np.isfinite(nt_m3) & (nt_m3 > 0.0)
        & np.isfinite(d0_mm) & (d0_mm > 0.0)
        & np.isfinite(sigma_m_mm)
        & np.isfinite(rain_rate_mm_h) & (rain_rate_mm_h >= min_rr_mm_h)
        & np.isfinite(refl_dbz) & (refl_dbz <= max_z_dbz)
    )
    if not keep.any():
        return _empty_samples()
    fields = {
        "dm_mm": dm_mm, "d0_mm": d0_mm, "sigma_m_mm": sigma_m_mm,
        "log_nw": log_nw, "lwc_g_m3": lwc_g_m3, "nt_m3": nt_m3,
        "mu": np.broadcast_to(np.asarray(mu, dtype=float), dm_mm.shape),
    }
    return {key: np.asarray(val)[keep].astype(np.float32) for key, val in fields.items()}


def moments_abel_boutle(
    qr: np.ndarray, rho: np.ndarray, dmin_m: float, min_rr: float, max_dbz: float
) -> dict[str, np.ndarray]:
    """1-moment Abel-Boutle moments, mu = 0 (Mk = 0.22*gamma(k+1)*lambda^(2.2-1-k))."""
    rho_qr = rho * qr
    lamb = (rho_qr / AB_LAMBDA_PREFACTOR) ** (-1.0 / 1.8)   # 1/m
    mu = np.zeros_like(lamb)
    m0 = 0.22 * lamb**1.2
    m3 = 6.0 * rho_qr / (1000.0 * math.pi)
    m3b = 0.22 * math.gamma(4.0 + FALL_B) * lamb ** (2.2 - 1.0 - 3.0 - FALL_B)
    m4 = 0.22 * math.gamma(5.0) * lamb ** (2.2 - 1.0 - 4.0)
    m6 = 0.22 * math.gamma(7.0) * lamb ** (2.2 - 1.0 - 6.0)
    return _diag_from_moments(
        qr, rho, m0, m3, m3b, m4, m6, mu, lamb, dmin_m, min_rr, max_dbz
    )


def moments_two_moment(
    qr: np.ndarray,
    nr_per_kg: np.ndarray,
    rho: np.ndarray,
    dmin_m: float,
    min_rr: float,
    max_dbz: float,
) -> dict[str, np.ndarray]:
    """2-moment gamma moments with the ALARO variable mu(dmeanr) law."""
    nt = nr_per_kg * rho                                    # zmom0
    m3 = 6.0 * qr * rho / (1000.0 * math.pi)                # zmom3
    with np.errstate(divide="ignore", invalid="ignore"):
        dmean_m = (m3 / nt) ** (1.0 / 3.0)
        mu = np.clip(19.0 * np.tanh(0.6 * (dmean_m * 1.0e3 - 1.8)) + 17.0, MU_MIN, MU_MAX)
        lamb = (nt * np.exp(gammaln(mu + 4.0) - gammaln(mu + 1.0)) / m3) ** (1.0 / 3.0)
        m3b = nt * np.exp(gammaln(mu + 4.0 + FALL_B) - gammaln(mu + 1.0)) / lamb ** (3.0 + FALL_B)
        m4 = nt * np.exp(gammaln(mu + 5.0) - gammaln(mu + 1.0)) / lamb**4
        m6 = nt * np.exp(gammaln(mu + 7.0) - gammaln(mu + 1.0)) / lamb**6
    return _diag_from_moments(
        qr, rho, nt, m3, m3b, m4, m6, mu, lamb, dmin_m, min_rr, max_dbz
    )


def _process_task(
    task: tuple[str, dict[str, str], float, float, float, float],
) -> tuple[dict[str, np.ndarray], list[str]]:
    experiment, paths, min_qr, dmin_m, min_rr, max_dbz = task
    warnings: list[str] = []
    try:
        qr = _read_masked_field(Path(paths["RAIN"]), "RAIN")
        temp = _read_masked_field(Path(paths["TEMPERATURE"]), "TEMPERATURE")
        pres = _read_masked_field(Path(paths["PRESSURE"]), "PRESSURE")
        pnr = _read_masked_field(Path(paths["PNR"]), "PNR") if experiment == "2mom" else None
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"WARNING {experiment} {paths['RAIN']}: {exc}")
        return _empty_samples(), warnings

    rho = pres / (RD * temp)
    rainy = (
        np.isfinite(qr) & np.isfinite(rho) & (rho > 0.0) & (qr > min_qr)
    )
    if pnr is not None:
        rainy &= np.isfinite(pnr) & (pnr > 0.0)
    if not rainy.any():
        return _empty_samples(), warnings

    if experiment == "2mom":
        diag = moments_two_moment(qr[rainy], pnr[rainy], rho[rainy], dmin_m, min_rr, max_dbz)
    else:
        diag = moments_abel_boutle(qr[rainy], rho[rainy], dmin_m, min_rr, max_dbz)
    return diag, warnings


def gather_experiment(
    experiment: str,
    records,
    domain_mask,
    *,
    min_qr: float,
    dmin_m: float,
    min_rr: float,
    max_dbz: float,
    workers: int,
    progress_every: int,
    tasks_per_child: int,
) -> dict[str, np.ndarray]:
    if not records:
        return _empty_samples()
    tasks = [
        (experiment, {var: str(path) for var, path in rec[3].items()}, min_qr, dmin_m, min_rr, max_dbz)
        for rec in records
    ]
    print(f"  [{experiment}] processing {len(tasks):,} timesteps -> truncated-moment DSD", flush=True)
    acc: dict[str, list[np.ndarray]] = {key: [] for key in SAMPLE_KEYS}
    if workers <= 1:
        _init_worker(domain_mask.mask)
        iterator = (_process_task(task) for task in tasks)
        pool = None
    else:
        pool = get_context("fork").Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(domain_mask.mask,),
            maxtasksperchild=tasks_per_child,
        )
        iterator = pool.imap_unordered(_process_task, tasks)
    try:
        for idx, (diag, warnings) in enumerate(iterator, 1):
            for key in SAMPLE_KEYS:
                acc[key].append(diag[key])
            for warning in warnings:
                print(warning, flush=True)
            if idx % progress_every == 0 or idx == len(tasks):
                n = sum(a.size for a in acc["dm_mm"])
                print(f"  [{experiment}] processed {idx}/{len(tasks)} (samples: {n:,})", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return {
        key: (np.concatenate(parts).astype(np.float32) if parts else np.empty(0, dtype=np.float32))
        for key, parts in acc.items()
    }


def _save_npz(path: Path, samples: dict[str, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        f"{name}__{key}": values
        for name, sample in samples.items()
        for key, values in sample.items()
    }
    np.savez_compressed(path, **payload)
    print(f"samples NPZ -> {path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--lead", default="all")
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--tasks-per-child", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--min-qr", type=float, default=1.0e-7)
    parser.add_argument(
        "--dmin-mm",
        type=float,
        default=DEFAULT_QC_DIAMETER_MIN_MM,
        help="Disdrometer minimum detectable diameter for the moment truncation (mm).",
    )
    parser.add_argument("--netcdf-root", type=Path, default=NETCDF_ROOT)
    parser.add_argument("--mask-file", type=Path, default=MASK_FILE)
    parser.add_argument("--mask-var", default=None)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--obs-zip", type=Path, default=OBS_ZIP)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--obs-min-precip", type=float, default=DEFAULT_QC_RAIN_RATE_MIN)
    parser.add_argument("--obs-max-dbz", type=float, default=DEFAULT_QC_REFLECTIVITY_MAX_DBZ)
    parser.add_argument("--obs-min-run-minutes", type=int, default=DEFAULT_QC_MIN_RUN_MINUTES)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--plot-root",
        type=Path,
        default=Path("/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper"),
        help="Paper figure root; the figure goes to '<root>/10. raindrop size distribution/'.",
    )
    parser.add_argument("--output-tag", default="all_leads")
    parser.add_argument("--bins", type=int, default=60)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    leads = _parse_leads(args.lead)
    dmin_m = args.dmin_mm * 1.0e-3

    print(f"reading observations from {args.obs_zip}", flush=True)
    obs = read_observation_samples(
        args.obs_zip,
        args.cache_dir,
        min_rr_mm_h=args.obs_min_precip,
        max_z_dbz=args.obs_max_dbz,
        min_run_minutes=args.obs_min_run_minutes,
    )
    obs_samples = normalized_gamma_from_empirical_samples(obs.path_a)
    print(
        f"obs QC kept {obs.qc_kept:,}/{obs.qc_total:,}; samples {obs_samples['dm_mm'].size:,}; "
        f"model dmin truncation at {args.dmin_mm:g} mm",
        flush=True,
    )

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

    samples: dict[str, dict[str, np.ndarray]] = {"obs": obs_samples}
    for exp in args.experiments:
        records = discover_records(exp, leads, args.netcdf_root, args.max_days)
        samples[exp] = gather_experiment(
            exp,
            records,
            domain_mask,
            min_qr=args.min_qr,
            dmin_m=dmin_m,
            min_rr=args.obs_min_precip,
            max_dbz=args.obs_max_dbz,
            workers=max(1, int(args.workers)),
            progress_every=args.progress_every,
            tasks_per_child=args.tasks_per_child,
        )
        print(f"  [{exp}] kept {samples[exp]['dm_mm'].size:,} samples", flush=True)

    _save_npz(
        args.processed_dir / f"disdrometer_dsd_new_plots_moments_dmin_{args.output_tag}.npz",
        samples,
    )

    out_dir = args.plot_root / "10. raindrop size distribution"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "10. raindrop size distribution_450dpi.png"
    plot_2x2_with_marginals(
        out_path=out,
        samples=samples,
        x_field="dm_mm",
        x_label="D$_m$ = M$_4$/M$_3$ (mm)",
        title=(
            "log$_{10}$ N$_w$ vs D$_m$, scheme moments, "
            f"truncated at D$_{{min}}$ = {args.dmin_mm:g} mm"
        ),
        bins=args.bins,
    )
    print(f"rendered {out}", flush=True)


if __name__ == "__main__":
    main()
