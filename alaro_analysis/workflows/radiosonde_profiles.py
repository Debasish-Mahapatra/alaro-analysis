"""Extract ALARO model profiles matched to ARM radiosonde launch times."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.radiosonde_settings import (
    DEFAULT_ALARO_ROOT,
    DEFAULT_OUTPUT_SUBDIR,
    DEFAULT_SONDE_ROOT,
    EXPERIMENT_ALIASES,
)
from alaro_analysis.data.radiosondes import (
    _build_output_dataset,
    _parse_yyyymmdd,
    _write_manifest,
    _write_missing_log,
    discover_sonde_launches,
    extract_experiment_profiles,
    median_launch_site,
)


def _resolve_experiments(tokens: Sequence[str]) -> list[str]:
    experiments: list[str] = []
    for token in tokens:
        key = token.strip().lower()
        if key not in EXPERIMENT_ALIASES:
            raise ValueError(
                f"Unknown experiment {token!r}; expected one of "
                f"{', '.join(sorted(EXPERIMENT_ALIASES))}"
            )
        experiment = EXPERIMENT_ALIASES[key]
        if experiment not in experiments:
            experiments.append(experiment)
    return experiments


def run(args: argparse.Namespace) -> dict[str, object]:
    alaro_root = Path(args.alaro_root)
    sonde_root = Path(args.sonde_root)
    start_date = _parse_yyyymmdd(args.start_date, "start-date")
    end_date = _parse_yyyymmdd(args.end_date, "end-date")
    experiments = _resolve_experiments(args.experiments)

    launches = discover_sonde_launches(
        sonde_root,
        start_date=start_date,
        end_date=end_date,
        limit=args.max_launches,
    )
    if not launches:
        raise ValueError(f"No radiosonde launches found in {sonde_root}")

    if args.site_lat is None or args.site_lon is None:
        site_lat, site_lon = median_launch_site(launches)
    else:
        site_lat = float(args.site_lat)
        site_lon = float(args.site_lon)

    print(
        f"Using ARM radiosonde launches in UTC: {len(launches)} launches, "
        f"site=({site_lat:.6f}, {site_lon:.6f}), source={args.source}, "
        f"workers={args.workers}",
        flush=True,
    )

    summary: dict[str, object] = {
        "alaro_root": str(alaro_root),
        "sonde_root": str(sonde_root),
        "source": args.source,
        "site_lat": site_lat,
        "site_lon": site_lon,
        "launches": len(launches),
        "workers": int(args.workers),
        "experiments": {},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    for experiment in experiments:
        out_dir = alaro_root / experiment / args.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{experiment}_radiosonde_matched_profiles.nc"
        if out_file.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists and --no-overwrite was set: {out_file}")

        print(f"[{experiment}] extracting to {out_file}", flush=True)
        results, missing = extract_experiment_profiles(
            alaro_root=alaro_root,
            experiment=experiment,
            launches=launches,
            site_lat=site_lat,
            site_lon=site_lon,
            source=args.source,
            progress_every=args.progress_every,
            workers=args.workers,
        )

        manifest_file = out_dir / f"{experiment}_radiosonde_matched_manifest.csv"
        missing_file = out_dir / f"{experiment}_radiosonde_matched_missing.csv"
        summary_file = out_dir / f"{experiment}_radiosonde_matched_summary.json"

        if results:
            ds = _build_output_dataset(
                experiment,
                results,
                site_lat=site_lat,
                site_lon=site_lon,
                source=args.source,
            )
            encoding = {
                name: {"zlib": True, "complevel": int(args.compression_level)}
                for name in (
                    "tdry",
                    "dewpoint",
                    "rh",
                    "pressure",
                    "specific_humidity",
                    "temperature_k",
                    "model_height",
                )
            }
            ds.to_netcdf(out_file, mode="w", encoding=encoding)
            ds.close()
        _write_manifest(manifest_file, results)
        _write_missing_log(missing_file, missing)

        exp_summary = {
            "output_file": str(out_file) if results else None,
            "manifest_file": str(manifest_file),
            "missing_file": str(missing_file),
            "profiles_written": len(results),
            "missing_profiles": len(missing),
            "output_subdir": str(out_dir),
            "source": args.source,
            "workers": int(args.workers),
        }
        summary["experiments"][experiment] = exp_summary
        summary_file.write_text(json.dumps(exp_summary, indent=2, sort_keys=True) + "\n")
        print(
            f"[{experiment}] done: profiles={len(results)} missing={len(missing)}",
            flush=True,
        )

    if args.plot_output_dir:
        from alaro_analysis.plotting.radiosondes import plot_profiles

        plot_args = argparse.Namespace(
            alaro_root=str(alaro_root),
            input_subdir=args.output_subdir,
            experiments=args.experiments,
            stat=args.plot_stat,
            max_height_m=args.plot_max_height_m,
            height_step_m=args.plot_height_step_m,
            min_samples=args.plot_min_samples,
            dpi=args.plot_dpi,
            output_dir=args.plot_output_dir,
        )
        plot_paths = plot_profiles(plot_args)
        summary["plot_outputs"] = [str(path) for path in plot_paths]
        for plot_path in plot_paths:
            print(f"Wrote plot: {plot_path}", flush=True)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract dry-bulb temperature, dewpoint, and RH model profiles at "
            "the ARM MAO radiosonde launch site and launch UTC hours."
        )
    )
    parser.add_argument("--sonde-root", default=str(DEFAULT_SONDE_ROOT))
    parser.add_argument("--alaro-root", default=str(DEFAULT_ALARO_ROOT))
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["control", "graupel", "2mom"],
        help="Experiments or aliases to process: c1m/control g1m/graupel g2m/2mom.",
    )
    parser.add_argument(
        "--source",
        choices=["fa", "netcdf"],
        default="fa",
        help="Read raw FA untar-output files or the existing masked-netcdf cache.",
    )
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--site-lat", type=float, default=None)
    parser.add_argument("--site-lon", type=float, default=None)
    parser.add_argument("--start-date", metavar="YYYYMMDD", default=None)
    parser.add_argument("--end-date", metavar="YYYYMMDD", default=None)
    parser.add_argument(
        "--max-launches",
        type=int,
        default=None,
        help="Limit launches for smoke tests.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes per experiment. Use with care for raw FA reads.",
    )
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument(
        "--plot-output-dir",
        default=None,
        help="Optional directory for separate tdry/dewpoint/RH plots after extraction.",
    )
    parser.add_argument("--plot-stat", choices=["mean", "median"], default="mean")
    parser.add_argument("--plot-max-height-m", type=float, default=20000.0)
    parser.add_argument("--plot-height-step-m", type=float, default=250.0)
    parser.add_argument("--plot-min-samples", type=int, default=10)
    parser.add_argument("--plot-dpi", type=int, default=180)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parse_configured_args(parser, "radiosonde_profiles", argv=argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def __getattr__(name):
    """Compatibility for pre-refactor imports; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    raise SystemExit(main())
