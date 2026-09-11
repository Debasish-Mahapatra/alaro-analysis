"""Raw per-cell normalized-gamma DSD plot sets.

Creates two comparison sets from the raw masked model netCDF fields:

* ``mu = 1``: all datasets are fitted as normalized gamma with fixed ``mu=1``
  using LWC + Nt.
* ``old way of doing mu``: observations use fitted gamma ``mu`` from the
  empirical mass-spectrum width; C1M/G1M use the native exponential
  Abel-Boutle rain shape (``mu=0``); G2M uses the fixed gamma rain shape
  (``mu=1``).
* ``variable mu``: same as above, except G2M uses the ALARO variable rain
  shape law ``mu(dmeanr)`` and only the mass-weighted diameter ``D_m=M4/M3``
  plot is rendered.
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENTS, RD
from alaro_analysis.common.dsd import (
    DEFAULT_QC_MIN_RUN_MINUTES,
    DEFAULT_QC_RAIN_RATE_MIN,
    DEFAULT_QC_REFLECTIVITY_MAX_DBZ,
    MP_FIXED_N0_PER_M3_MM,
    gamma_dsd_from_q_n_per_kg,
    gamma_dsd_from_q_n_variable_mu_per_kg,
    mp_from_q_abel_boutle,
    mp_from_q_fixed_n0,
    normalized_gamma_diagnostics_from_lwc_nt_mu,
    normalized_gamma_from_empirical_samples,
)
from alaro_analysis.common.observation_settings import CACHE_DIR, OBS_ZIP
from alaro_analysis.data.cache import save_npz_compressed
from alaro_analysis.data.dsd_percell import (
    _init_worker,
    _parse_leads,
    _read_masked_field,
    discover_records,
)
from alaro_analysis.data.masks import build_domain_mask_from_netcdf
from alaro_analysis.dsd.surface import (
    FIGURE_DIR,
    MASK_FILE,
    NETCDF_ROOT,
    OBS_PARAMETERS,
    PROCESSED_DIR,
    read_observation_samples,
)
from alaro_analysis.plotting.dsd_surface import plot_2x2_with_marginals

SAMPLE_KEYS = (*OBS_PARAMETERS, "mu")


def _empty_samples() -> dict[str, np.ndarray]:
    return {key: np.empty(0, dtype=np.float32) for key in SAMPLE_KEYS}


def _filter_diag(
    diag: dict[str, np.ndarray],
    *,
    d0_min_mm: float,
    fallback_mu: float | None = None,
    require_d0_min: bool = True,
) -> dict[str, np.ndarray]:
    d0_ok = np.isfinite(diag["d0_mm"]) & (diag["d0_mm"] > 0.0)
    if require_d0_min:
        d0_ok &= diag["d0_mm"] >= d0_min_mm
    keep = (
        np.isfinite(diag["dm_mm"])
        & (diag["dm_mm"] > 0.0)
        & d0_ok
        & np.isfinite(diag["sigma_m_mm"])
        & np.isfinite(diag["log_nw"])
        & np.isfinite(diag["lwc_g_m3"])
        & (diag["lwc_g_m3"] > 0.0)
        & np.isfinite(diag["nt_m3"])
        & (diag["nt_m3"] > 0.0)
    )
    if not keep.any():
        return _empty_samples()
    out = {key: diag[key][keep].astype(np.float32) for key in OBS_PARAMETERS}
    if "mu" in diag:
        out["mu"] = diag["mu"][keep].astype(np.float32)
    elif fallback_mu is not None:
        out["mu"] = np.full(out["dm_mm"].shape, float(fallback_mu), dtype=np.float32)
    else:
        raise KeyError("diag has no mu and no fallback_mu was provided")
    return out


def _fixed_mu1_from_lwc_nt(
    lwc_g_m3: np.ndarray,
    nt_m3: np.ndarray,
    *,
    d0_min_mm: float,
) -> dict[str, np.ndarray]:
    diag = normalized_gamma_diagnostics_from_lwc_nt_mu(lwc_g_m3, nt_m3, 1.0)
    return _filter_diag(diag, d0_min_mm=d0_min_mm)


def _process_task(
    task: tuple[str, dict[str, str], float, str, float, float, bool],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    experiment, paths, min_qr, onemom_closure, n0_fixed, d0_min_mm, only_variable_mu = task
    warnings: list[str] = []
    try:
        qr = _read_masked_field(Path(paths["RAIN"]), "RAIN")
        temp = _read_masked_field(Path(paths["TEMPERATURE"]), "TEMPERATURE")
        pres = _read_masked_field(Path(paths["PRESSURE"]), "PRESSURE")
        pnr = _read_masked_field(Path(paths["PNR"]), "PNR") if experiment == "2mom" else None
    except Exception as exc:
        warnings.append(f"WARNING {experiment} {paths['RAIN']}: {exc}")
        return _empty_samples(), _empty_samples(), _empty_samples(), warnings

    rho = pres / (RD * temp)
    rainy = (
        np.isfinite(qr)
        & np.isfinite(temp)
        & np.isfinite(pres)
        & np.isfinite(rho)
        & (rho > 0.0)
        & (qr > min_qr)
    )
    if pnr is not None:
        rainy &= np.isfinite(pnr) & (pnr > 0.0)
    if not rainy.any():
        return _empty_samples(), _empty_samples(), _empty_samples(), warnings

    qr_v = qr[rainy]
    rho_v = rho[rainy]
    lwc_g_m3 = 1000.0 * rho_v * qr_v

    if experiment == "2mom":
        nt_m3 = rho_v * pnr[rainy]
        variable_mu_diag = gamma_dsd_from_q_n_variable_mu_per_kg(qr_v, pnr[rainy], rho_v)
        variable_mu = _filter_diag(variable_mu_diag, d0_min_mm=d0_min_mm, require_d0_min=False)
        if only_variable_mu:
            return _empty_samples(), _empty_samples(), variable_mu, warnings
        native_diag = gamma_dsd_from_q_n_per_kg(qr_v, pnr[rainy], rho_v, mu=1.0)
        native = _filter_diag(native_diag, d0_min_mm=d0_min_mm)
    elif onemom_closure == "abel_boutle":
        native_diag = mp_from_q_abel_boutle(qr_v, rho_v)
        nt_m3 = native_diag["nt_m3"]
        variable_mu = _filter_diag(
            native_diag,
            d0_min_mm=d0_min_mm,
            fallback_mu=0.0,
            require_d0_min=False,
        )
        if only_variable_mu:
            return _empty_samples(), _empty_samples(), variable_mu, warnings
        native = _filter_diag(native_diag, d0_min_mm=d0_min_mm, fallback_mu=0.0)
    else:
        native_diag = mp_from_q_fixed_n0(qr_v, rho_v, n0_per_m3_mm=n0_fixed)
        nt_m3 = native_diag["nt_m3"]
        variable_mu = _filter_diag(
            native_diag,
            d0_min_mm=d0_min_mm,
            fallback_mu=0.0,
            require_d0_min=False,
        )
        if only_variable_mu:
            return _empty_samples(), _empty_samples(), variable_mu, warnings
        native = _filter_diag(native_diag, d0_min_mm=d0_min_mm, fallback_mu=0.0)

    mu1 = _fixed_mu1_from_lwc_nt(lwc_g_m3, nt_m3, d0_min_mm=d0_min_mm)
    return native, mu1, variable_mu, warnings


def gather_experiment_sets(
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
    only_variable_mu: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not records:
        return _empty_samples(), _empty_samples(), _empty_samples()
    tasks = [
        (
            experiment,
            {var: str(path) for var, path in rec[3].items()},
            min_qr,
            onemom_closure,
            n0_fixed,
            d0_min_mm,
            only_variable_mu,
        )
        for rec in records
    ]
    if only_variable_mu:
        print(
            f"  [{experiment}] processing {len(tasks):,} timesteps -> Dm-only comparison",
            flush=True,
        )
    else:
        print(
            f"  [{experiment}] processing {len(tasks):,} timesteps -> raw per-cell normalized gamma",
            flush=True,
        )
    native_acc: dict[str, list[np.ndarray]] = {key: [] for key in SAMPLE_KEYS}
    mu1_acc: dict[str, list[np.ndarray]] = {key: [] for key in SAMPLE_KEYS}
    variable_mu_acc: dict[str, list[np.ndarray]] = {key: [] for key in SAMPLE_KEYS}
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
        for idx, (native, mu1, variable_mu, warnings) in enumerate(iterator, 1):
            for key in SAMPLE_KEYS:
                native_acc[key].append(native[key])
                mu1_acc[key].append(mu1[key])
                variable_mu_acc[key].append(variable_mu[key])
            for warning in warnings:
                print(warning, flush=True)
            if idx % progress_every == 0 or idx == len(tasks):
                native_n = sum(a.size for a in native_acc["dm_mm"])
                mu1_n = sum(a.size for a in mu1_acc["dm_mm"])
                variable_mu_n = sum(a.size for a in variable_mu_acc["dm_mm"])
                if only_variable_mu:
                    print(
                        f"  [{experiment}] processed {idx}/{len(tasks)} "
                        f"(Dm comparison samples: {variable_mu_n:,})",
                        flush=True,
                    )
                else:
                    print(
                        f"  [{experiment}] processed {idx}/{len(tasks)} "
                        f"(native samples: {native_n:,}; mu=1 samples: {mu1_n:,}; "
                        f"Dm comparison samples: {variable_mu_n:,})",
                        flush=True,
                    )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    def finish(acc: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
        return {
            key: (
                np.concatenate(parts).astype(np.float32) if parts else np.empty(0, dtype=np.float32)
            )
            for key, parts in acc.items()
        }

    return finish(native_acc), finish(mu1_acc), finish(variable_mu_acc)


def _save_npz(path: Path, samples: dict[str, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        f"{name}__{key}": values
        for name, sample in samples.items()
        for key, values in sample.items()
    }
    save_npz_compressed(path, **payload)
    print(f"samples NPZ -> {path}", flush=True)


def _render_set(
    out_dir: Path,
    samples: dict[str, dict[str, np.ndarray]],
    *,
    prefix: str,
    title_note: str,
    output_tag: str,
    bins: int,
    fields: tuple[tuple[str, str, str], ...] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = (
            ("d0_mm", "D$_0$ (mm)", "logNw_D0"),
            ("dm_mm", "D$_m$ (mm)", "logNw_Dm"),
        )
    for x_field, x_label, suffix in fields:
        out = out_dir / f"{prefix}_{suffix}_{output_tag}.png"
        title = f"log$_{{10}}$ N$_w$ vs {'D$_0$' if x_field == 'd0_mm' else 'D$_m$'}, {title_note}"
        plot_2x2_with_marginals(
            out_path=out,
            samples=samples,
            x_field=x_field,
            x_label=x_label,
            title=title,
            bins=bins,
        )
        print(f"rendered {out}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raw per-cell normalized-gamma DSD plot sets")
    parser.add_argument("--lead", default="all")
    parser.add_argument(
        "--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS)
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--tasks-per-child", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--min-qr", type=float, default=1.0e-7)
    parser.add_argument("--d0-min-mm", type=float, default=0.2)
    parser.add_argument(
        "--onemom-closure", choices=("abel_boutle", "fixed_n0"), default="abel_boutle"
    )
    parser.add_argument("--n0-fixed", type=float, default=MP_FIXED_N0_PER_M3_MM)
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
    parser.add_argument("--plot-root", type=Path, default=FIGURE_DIR / "new plots")
    parser.add_argument("--output-tag", default="all_leads")
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument(
        "--only-variable-mu",
        action="store_true",
        help="Only save/render the G2M variable-mu Dm=M4/M3 comparison product.",
    )
    return parser


def main() -> None:
    args = parse_configured_args(
        build_parser(), "run_dsd_per_cell_normalized_gamma_raw_sets", argv=None
    )
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.plot_root.mkdir(parents=True, exist_ok=True)
    leads = _parse_leads(args.lead)

    print(f"reading observations from {args.obs_zip}", flush=True)
    obs = read_observation_samples(
        args.obs_zip,
        args.cache_dir,
        min_rr_mm_h=args.obs_min_precip,
        max_z_dbz=args.obs_max_dbz,
        min_run_minutes=args.obs_min_run_minutes,
    )
    obs_native = normalized_gamma_from_empirical_samples(obs.path_a)
    if args.only_variable_mu:
        obs_mu1 = _empty_samples()
        print(
            f"obs QC kept {obs.qc_kept:,}/{obs.qc_total:,}; "
            f"Dm comparison samples {obs_native['dm_mm'].size:,}",
            flush=True,
        )
    else:
        obs_mu1 = _fixed_mu1_from_lwc_nt(
            obs.path_a["lwc_g_m3"],
            obs.path_a["nt_m3"],
            d0_min_mm=args.d0_min_mm,
        )
        print(
            f"obs QC kept {obs.qc_kept:,}/{obs.qc_total:,}; "
            f"native/fitted mu samples {obs_native['dm_mm'].size:,}; mu=1 samples {obs_mu1['dm_mm'].size:,}",
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

    native_samples: dict[str, dict[str, np.ndarray]] = {"obs": obs_native}
    mu1_samples: dict[str, dict[str, np.ndarray]] = {"obs": obs_mu1}
    variable_mu_samples: dict[str, dict[str, np.ndarray]] = {"obs": obs_native}
    for exp in args.experiments:
        records = discover_records(exp, leads, args.netcdf_root, args.max_days)
        native, mu1, variable_mu = gather_experiment_sets(
            exp,
            records,
            domain_mask,
            min_qr=args.min_qr,
            onemom_closure=args.onemom_closure,
            n0_fixed=args.n0_fixed,
            d0_min_mm=args.d0_min_mm,
            workers=max(1, int(args.workers)),
            progress_every=args.progress_every,
            tasks_per_child=args.tasks_per_child,
            only_variable_mu=args.only_variable_mu,
        )
        native_samples[exp] = native
        mu1_samples[exp] = mu1
        variable_mu_samples[exp] = variable_mu
        if args.only_variable_mu:
            print(f"  [{exp}] kept Dm comparison {variable_mu['dm_mm'].size:,}", flush=True)
        else:
            print(
                f"  [{exp}] kept native/fitted {native['dm_mm'].size:,}; "
                f"mu=1 {mu1['dm_mm'].size:,}; Dm comparison {variable_mu['dm_mm'].size:,}",
                flush=True,
            )

    if not args.only_variable_mu:
        _save_npz(
            args.processed_dir
            / f"disdrometer_dsd_new_plots_native_fitted_mu_raw_{args.output_tag}.npz",
            native_samples,
        )
        _save_npz(
            args.processed_dir / f"disdrometer_dsd_new_plots_mu1_raw_{args.output_tag}.npz",
            mu1_samples,
        )
    _save_npz(
        args.processed_dir / f"disdrometer_dsd_new_plots_variable_mu_raw_{args.output_tag}.npz",
        variable_mu_samples,
    )

    if not args.only_variable_mu:
        _render_set(
            args.plot_root / "old way of doing mu",
            native_samples,
            prefix="dsd_percell_native_fitted_mu_raw",
            title_note="normalized gamma, native/fitted mu, raw per-cell fields",
            output_tag=args.output_tag,
            bins=args.bins,
        )
        _render_set(
            args.plot_root / "mu = 1",
            mu1_samples,
            prefix="dsd_percell_mu1_raw",
            title_note="normalized gamma, mu=1, raw per-cell fields",
            output_tag=args.output_tag,
            bins=args.bins,
        )
    _render_set(
        args.plot_root / "variable mu",
        variable_mu_samples,
        prefix="dsd_percell_variable_mu_raw",
        title_note="normalized gamma, G2M variable mu(dmeanr), raw per-cell fields",
        output_tag=args.output_tag,
        bins=args.bins,
        fields=(("dm_mm", "D$_m$ = M$_4$/M$_3$ (mm)", "logNw_Dm"),),
    )


if __name__ == "__main__":
    main()
