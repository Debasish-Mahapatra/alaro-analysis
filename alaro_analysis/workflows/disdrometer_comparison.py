from __future__ import annotations

import argparse
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from alaro_analysis.analysis.disdrometer import summarize_experiment
from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENTS
from alaro_analysis.common.observation_settings import (
    ANALYTICS_DIR,
    BOTTOM_LEVEL,
    CACHE_DIR,
    DEFAULT_ALT_M,
    DEFAULT_LAT,
    DEFAULT_LON,
    FIGURE_DIR,
    MODEL_ROOT,
    OBS_ZIP,
    PROCESSED_DIR,
)
from alaro_analysis.data.cache import save_npz
from alaro_analysis.data.disdrometer import (
    average_observation_window,
    read_observations,
)
from alaro_analysis.data.disdrometer_exports import (
    save_summary_csv,
    save_timeseries_csv,
)
from alaro_analysis.data.model_points import (
    _init_epygram_worker,
    _read_model_points_task,
)
from alaro_analysis.data.model_records import (
    discover_model_records,
    lead_label,
    parse_lead_selection,
    title_lead_text,
)
from alaro_analysis.data.observation_models import ObservationWindow


def run_comparison(args: argparse.Namespace) -> dict[str, Path]:
    from alaro_analysis.plotting.disdrometer_comparison import (
        plot_distribution,
        plot_scatter,
        plot_timeseries,
    )

    experiments = tuple(args.experiments)
    workers = max(1, min(int(args.workers), 32))
    leads = parse_lead_selection(args.lead)
    output_tag = lead_label(leads)
    lead_text = title_lead_text(leads)
    print(f"reading observations from {args.obs_zip}", flush=True)
    obs = read_observations(
        args.obs_zip,
        args.cache_dir,
        min_precip_rate_mm_h=args.obs_min_precip,
    )
    print(
        f"obs samples: {len(obs.times):,}; rainy/QC samples: "
        f"{int(np.isfinite(obs.rain_number_m3).sum()):,}",
        flush=True,
    )

    import epygram

    epygram.init_env()
    model_records = {
        exp: discover_model_records(exp, leads, args.model_root, args.max_days)
        for exp in experiments
    }
    common_records = sorted(set.intersection(*(set(v) for v in model_records.values())))
    if not common_records:
        raise RuntimeError("No common model files found for requested experiments/leads")
    unique_valid_times = sorted({record[0] for record in common_records})
    print(f"lead selection: {lead_text}", flush=True)
    print(f"common model records: {len(common_records):,}", flush=True)
    print(f"unique valid times: {len(unique_valid_times):,}", flush=True)
    print(f"workers: {workers}", flush=True)

    obs_windows: dict[str, ObservationWindow] = {}
    for valid_time_key in unique_valid_times:
        window = average_observation_window(
            obs,
            np.datetime64(valid_time_key),
            args.obs_window_minutes,
        )
        obs_windows[valid_time_key] = window

    tasks = [
        (
            f"{init_time_key}+{lead:04d}->{valid_time_key}",
            valid_time_key,
            lead,
            {exp: str(model_records[exp][record]) for exp in experiments},
            experiments,
            args.station_lat,
            args.station_lon,
            args.level,
        )
        for record in common_records
        for valid_time_key, init_time_key, lead in [record]
    ]

    model_rows: dict[str, dict[str, Any]] = {}
    if workers == 1:
        import epygram

        epygram.init_env()
        for idx, task in enumerate(tasks, 1):
            valid_time_key, model_row, warnings = _read_model_points_task(task)
            for warning in warnings:
                print(warning, flush=True)
            model_rows[valid_time_key] = model_row
            if idx % args.progress_every == 0 or idx == len(tasks):
                print(f"processed {idx}/{len(tasks)} valid times", flush=True)
    else:
        with get_context("fork").Pool(
            processes=workers,
            initializer=_init_epygram_worker,
            maxtasksperchild=args.tasks_per_child,
        ) as pool:
            for idx, (valid_time_key, model_row, warnings) in enumerate(
                pool.imap_unordered(_read_model_points_task, tasks),
                1,
            ):
                for warning in warnings:
                    print(warning, flush=True)
                model_rows[valid_time_key] = model_row
                if idx % args.progress_every == 0 or idx == len(tasks):
                    print(f"processed {idx}/{len(tasks)} valid times", flush=True)

    rows: list[dict[str, Any]] = []
    model_arrays: dict[str, list[float]] = {exp: [] for exp in experiments}
    model_qr_arrays: dict[str, list[float]] = {exp: [] for exp in experiments}
    obs_values: list[float] = []
    valid_times_out: list[np.datetime64] = []
    init_times_out: list[np.datetime64] = []
    leads_out: list[int] = []

    for record in common_records:
        valid_time_key, init_time_key, lead = record
        record_key = f"{init_time_key}+{lead:04d}->{valid_time_key}"
        window = obs_windows[valid_time_key]
        row: dict[str, Any] = {
            "valid_time_utc": valid_time_key.replace("T", " "),
            "init_time_utc": init_time_key.replace("T", " "),
            "lead_hours": lead,
            "obs_rain_number_m3": window.rain_number_m3,
            "obs_precip_rate_mm_h": window.precip_rate_mm_h,
            "obs_median_volume_diameter_mm": window.median_volume_diameter_mm,
            "obs_sample_count": window.sample_count,
        }
        obs_values.append(window.rain_number_m3)
        valid_times_out.append(np.datetime64(valid_time_key))
        init_times_out.append(np.datetime64(init_time_key))
        leads_out.append(lead)

        row.update(model_rows[record_key])
        for exp in experiments:
            model_arrays[exp].append(row[f"{exp}_rain_number_m3"])
            model_qr_arrays[exp].append(row[f"{exp}_rain_mixing_ratio_kgkg"])

        rows.append(row)

    valid_times_arr = np.asarray(valid_times_out, dtype="datetime64[s]")
    init_times_arr = np.asarray(init_times_out, dtype="datetime64[s]")
    leads_arr = np.asarray(leads_out, dtype=int)
    obs_arr = np.asarray(obs_values, dtype=float)
    model_arrs = {exp: np.asarray(values, dtype=float) for exp, values in model_arrays.items()}
    model_qr_arrs = {
        exp: np.asarray(values, dtype=float) for exp, values in model_qr_arrays.items()
    }

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.analytics_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.processed_dir / f"disdrometer_rain_number_comparison_{output_tag}.npz"
    save_npz(
        npz_path,
        valid_time=valid_times_arr.astype("datetime64[s]").astype(str),
        init_time=init_times_arr.astype("datetime64[s]").astype(str),
        lead_hours=leads_arr,
        obs_rain_number_m3=obs_arr,
        obs_precip_rate_mm_h=np.asarray([row["obs_precip_rate_mm_h"] for row in rows], dtype=float),
        obs_median_volume_diameter_mm=np.asarray(
            [row["obs_median_volume_diameter_mm"] for row in rows], dtype=float
        ),
        obs_sample_count=np.asarray([row["obs_sample_count"] for row in rows], dtype=int),
        **{f"{exp}_rain_number_m3": values for exp, values in model_arrs.items()},
        **{f"{exp}_rain_mixing_ratio_kgkg": values for exp, values in model_qr_arrs.items()},
    )

    timeseries_csv = (
        args.analytics_dir / f"disdrometer_rain_number_comparison_{output_tag}_timeseries.csv"
    )
    save_timeseries_csv(timeseries_csv, rows, experiments)

    summaries = [
        summarize_experiment(exp, obs_arr, model_arrs[exp], model_qr_arrs[exp])
        for exp in experiments
    ]
    summary_csv = (
        args.analytics_dir / f"disdrometer_rain_number_comparison_{output_tag}_summary.csv"
    )
    save_summary_csv(summary_csv, summaries)

    timeseries_png = args.figure_dir / f"disdrometer_rain_number_timeseries_{output_tag}.png"
    scatter_png = args.figure_dir / f"disdrometer_rain_number_scatter_{output_tag}.png"
    distribution_png = args.figure_dir / f"disdrometer_rain_number_distribution_{output_tag}.png"
    if not args.no_plots:
        plot_timeseries(timeseries_png, valid_times_arr, obs_arr, model_arrs, lead_text)
        plot_scatter(scatter_png, obs_arr, model_arrs, lead_text)
        plot_distribution(distribution_png, obs_arr, model_arrs, lead_text)

    outputs = {
        "npz": npz_path,
        "timeseries_csv": timeseries_csv,
        "summary_csv": summary_csv,
        "timeseries_png": timeseries_png,
        "scatter_png": scatter_png,
        "distribution_png": distribution_png,
    }
    for key, path in outputs.items():
        if path.exists():
            print(f"{key}: {path}", flush=True)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ARM GOAmazon Manacapuru disdrometer drop number "
            "concentration with lowest-level ALARO rain number."
        )
    )
    parser.add_argument(
        "--lead",
        default="all",
        help="Forecast lead selection: all, a single hour like 0024, or a range like 0000-0024.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS),
        choices=list(EXPERIMENTS),
        help="Experiments to compare.",
    )
    parser.add_argument("--level", type=int, default=BOTTOM_LEVEL, help="Model level to read.")
    parser.add_argument("--station-lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--station-lon", type=float, default=DEFAULT_LON)
    parser.add_argument("--station-alt-m", type=float, default=DEFAULT_ALT_M)
    parser.add_argument("--obs-window-minutes", type=int, default=30)
    parser.add_argument("--obs-min-precip", type=float, default=0.1)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--tasks-per-child", type=int, default=32)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--obs-zip", type=Path, default=OBS_ZIP)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--analytics-dir", type=Path, default=ANALYTICS_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parse_configured_args(parser, "disdrometer_comparison", argv=argv)
    run_comparison(args)


def __getattr__(name):
    """Compatibility for pre-refactor imports; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    main()
