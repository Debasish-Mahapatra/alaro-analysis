"""Disdrometer vs ALARO domain-mean DSD comparison (Marshall-Palmer framework).

Both observations and the model runs are reduced to the same Marshall-Palmer
DSD parameter set (D_m, D_0, log_10 N_w, sigma_m), with two parallel obs
reductions:

* **Path A (empirical obs)**: D_m = M_4/M_3, D_0 from the cumulative
  mass-50% crossing, log_10 N_w from LWC and the empirical D_m.
  No DSD shape forced on the observations.
* **Path B (MP-projected obs)**: fit Marshall-Palmer (N_0, lambda) per
  rainy QC'd minute from observed (LWC, N_t).  All parameters come from
  the analytic MP relations.

For the model side both paths use the same MP closure:

* **2-moment (G2M)** uses prognostic q_r and PNR -> (N_0, lambda).
* **1-moment (C1M, G1M)** uses fixed N_0 = 8e6 m^-4 (Marshall & Palmer
  1948), with lambda solved from q_r alone.

QC of the observations follows ``disdro.tex`` Section 1: R > 0.1 mm/h,
Z_e < 55 dBZ, and a minimum of five consecutive rainy minutes.

Outputs:

* Path A figures ``figures/disdrometer_dsd/dsd_pathA_*_<tag>.png``
* Path B figures ``figures/disdrometer_dsd/dsd_pathB_*_<tag>.png``
* Per-experiment domain-mean time series under ``processed-data/``
* Sample arrays per dataset/path in NPZ for reuse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENTS
from alaro_analysis.common.dsd import (
    DEFAULT_QC_MIN_RUN_MINUTES,
    DEFAULT_QC_RAIN_RATE_MIN,
    DEFAULT_QC_REFLECTIVITY_MAX_DBZ,
    MP_FIXED_N0_PER_M3_MM,
)
from alaro_analysis.common.observation_settings import CACHE_DIR, OBS_ZIP
from alaro_analysis.data.dsd_exports import (
    save_samples_npz,
    write_experiment_time_series_netcdf,
)
from alaro_analysis.data.masks import build_domain_mask_from_netcdf
from alaro_analysis.data.model_records import lead_label, parse_lead_selection
from alaro_analysis.dsd.surface import (
    ANALYTICS_DIR,
    FIGURE_DIR,
    MASK_FILE,
    NETCDF_ROOT,
    PROCESSED_DIR,
    discover_netcdf_records,
    gather_experiment_time_series,
    read_observation_samples,
    time_series_to_samples,
)

# ---------------------------------------------------------------------------
# Observation reduction
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Plotting helpers (2x2 with obs-contour overlay)
# ---------------------------------------------------------------------------


def run_dsd(args: argparse.Namespace) -> dict[str, Path]:
    from alaro_analysis.plotting.dsd_surface import plot_2x2_joint_density

    experiments = tuple(args.experiments)
    leads = parse_lead_selection(args.lead)
    output_tag = f"{lead_label(leads)}"

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.analytics_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading observations from {args.obs_zip}", flush=True)
    obs = read_observation_samples(
        args.obs_zip,
        args.cache_dir,
        min_rr_mm_h=args.obs_min_precip,
        max_z_dbz=args.obs_max_dbz,
        min_run_minutes=args.obs_min_run_minutes,
    )
    print(
        f"obs QC: kept {obs.qc_kept:,} / {obs.qc_total:,} minutes "
        f"(R>{args.obs_min_precip} mm/h, Ze<{args.obs_max_dbz} dBZ, "
        f">={args.obs_min_run_minutes}-min runs)",
        flush=True,
    )
    print(
        f"  Path A (empirical) samples: {obs.path_a['dm_mm'].size:,}; "
        f"Path B (MP-projected) samples: {obs.path_b['dm_mm'].size:,}",
        flush=True,
    )

    sample_records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]] | None = None
    for exp in experiments:
        recs = discover_netcdf_records(exp, leads, args.netcdf_root, args.max_days)
        if recs:
            sample_records = recs
            break
    if not sample_records:
        raise RuntimeError("No netCDF records found for any experiment")

    sample_path = sample_records[0][3]["RAIN"]
    print(f"building radar mask from {args.mask_file} (sample {sample_path})", flush=True)
    domain_mask = build_domain_mask_from_netcdf(
        sample_path,
        args.mask_file,
        mask_var=args.mask_var,
        mask_threshold=args.mask_threshold,
    )
    print(f"  mask cells kept: {domain_mask.n_cells} / {domain_mask.mask.size}", flush=True)

    model_samples: dict[str, dict[str, np.ndarray]] = {}
    for exp in experiments:
        print(f"gathering domain-mean MP DSD for {exp}", flush=True)
        records = discover_netcdf_records(exp, leads, args.netcdf_root, args.max_days)
        rows = gather_experiment_time_series(
            exp,
            records,
            domain_mask,
            min_qr=args.min_qr,
            onemom_closure=args.onemom_closure,
            n0_fixed=args.n0_fixed,
            workers=max(1, int(args.workers)),
            progress_every=args.progress_every,
            tasks_per_child=args.tasks_per_child,
        )
        nc_path = args.processed_dir / f"disdrometer_dsd_mp_{exp}_{output_tag}.nc"
        if rows:
            write_experiment_time_series_netcdf(
                nc_path,
                rows,
                exp,
                domain_mask,
                onemom_closure=args.onemom_closure,
                n0_fixed=args.n0_fixed,
            )
            print(f"  [{exp}] wrote {nc_path}", flush=True)
        samples = time_series_to_samples(rows)
        model_samples[exp] = samples
        print(f"  [{exp}] kept {samples['dm_mm'].size:,} rainy timesteps", flush=True)

    samples_a = {"obs": obs.path_a, **model_samples}
    samples_b = {"obs": obs.path_b, **model_samples}
    npz_path = args.processed_dir / f"disdrometer_dsd_samples_{output_tag}.npz"
    save_samples_npz(
        npz_path,
        {
            "obs_pathA": obs.path_a,
            "obs_pathB": obs.path_b,
            **{exp: model_samples[exp] for exp in experiments},
        },
    )
    print(f"sample arrays saved to {npz_path}", flush=True)

    fig_paths: dict[str, Path] = {}
    if not args.no_plots:
        for path_label, samples in (("pathA", samples_a), ("pathB", samples_b)):
            obs_kind = "empirical" if path_label == "pathA" else "MP-projected"
            for x_field, x_label, suffix in (
                ("d0_mm", "D$_0$ (mm)", "logNw_D0"),
                ("dm_mm", "D$_m$ (mm)", "logNw_Dm"),
            ):
                title = (
                    f"log$_{{10}}$ N$_w$ vs {'D$_0$' if x_field == 'd0_mm' else 'D$_m$'}, "
                    f"obs {obs_kind}"
                )
                out = args.figure_dir / f"dsd_{path_label}_{suffix}_{output_tag}.png"
                plot_2x2_joint_density(
                    out_path=out,
                    samples=samples,
                    x_field=x_field,
                    x_label=x_label,
                    title=title,
                )
                fig_paths[f"{path_label}_{suffix}"] = out
                print(f"  rendered {out}", flush=True)

    outputs = {"samples_npz": npz_path, **fig_paths}
    for label, path in outputs.items():
        if path.exists():
            print(f"{label}: {path}", flush=True)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Disdrometer vs ALARO domain-mean DSD comparison under a "
            "Marshall-Palmer closure (model side) and two parallel obs "
            "reductions (empirical and MP-projected)."
        )
    )
    parser.add_argument("--lead", default="all")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS),
        choices=list(EXPERIMENTS),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--tasks-per-child", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--min-qr",
        type=float,
        default=1.0e-7,
        help="Minimum domain-mean rain mixing ratio (kg/kg) to keep a timestep.",
    )
    parser.add_argument(
        "--onemom-closure",
        choices=("abel_boutle", "fixed_n0"),
        default="abel_boutle",
        help="DSD closure for the 1-moment runs: 'abel_boutle' (Abel & Boutle 2012, "
        "the formula in arpifs/adiab/gpprs0d.F90) or 'fixed_n0' (classical Marshall-Palmer).",
    )
    parser.add_argument(
        "--n0-fixed",
        type=float,
        default=MP_FIXED_N0_PER_M3_MM,
        help="Marshall-Palmer N0 (1/(m^3 mm)) used when --onemom-closure=fixed_n0 "
        "(default 8000, ie 8e6 m^-4).",
    )
    parser.add_argument("--obs-min-precip", type=float, default=DEFAULT_QC_RAIN_RATE_MIN)
    parser.add_argument("--obs-max-dbz", type=float, default=DEFAULT_QC_REFLECTIVITY_MAX_DBZ)
    parser.add_argument("--obs-min-run-minutes", type=int, default=DEFAULT_QC_MIN_RUN_MINUTES)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--netcdf-root", type=Path, default=NETCDF_ROOT)
    parser.add_argument("--obs-zip", type=Path, default=OBS_ZIP)
    parser.add_argument("--mask-file", type=Path, default=MASK_FILE)
    parser.add_argument("--mask-var", default=None)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--analytics-dir", type=Path, default=ANALYTICS_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parse_configured_args(parser, "disdrometer_dsd", argv=argv)
    run_dsd(args)


def __getattr__(name):
    """Compatibility for pre-refactor imports; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    main()
