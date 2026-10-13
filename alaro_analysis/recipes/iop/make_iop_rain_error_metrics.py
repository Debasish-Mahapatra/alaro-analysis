#!/usr/bin/env python3
"""Error metrics of hourly rain against the SIPAM radar for the GoAmazon IOPs.

Companion to the bias maps (figure 1): the same common-valid hourly rain written by
``build_iop_rainfall_common_valid`` (0.1 deg grid, radar valid-time mask and radar
footprint applied to every dataset), scored against the radar with the bias, the
RMSE and the correlation at four levels of aggregation:

    point hourly      every cell and every hour, the strictest score
    domain hourly     area mean of the footprint each hour, so placement errors
                      inside the domain drop out and only the hourly amount is scored
    domain daily      area mean per day, mm/day
    diurnal cycle     the 24 local-hour means of figure 3
    time-mean map     the per-cell time means of figure 1, a pure pattern error

Point hourly RMSE in convective rain is dominated by the double penalty: a shower
in the wrong cell costs twice. The domain and diurnal scores are the interpretable
ones. The centred RMSE is the part of the RMSE that is left after the mean bias is
removed, so RMSE^2 = bias^2 + centred RMSE^2.

Writes ``rain_error_metrics_<period>.txt`` beside the bias-map data.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.iop_settings import FIGURES, GOAMAZON_IOPS, IOP_PERIODS, figure_files
from alaro_analysis.recipes.iop.build_iop_rainfall_common_valid import (
    COMMON_SUBDIR,
    DEFAULT_WORK_DIR,
)

# Period -> common-valid subdirectory
PERIODS = {
    "combined": COMMON_SUBDIR,
    "iop1": f"{COMMON_SUBDIR}-iop1",
    "iop2": f"{COMMON_SUBDIR}-iop2",
}
REFERENCE = "Radar"
CANDIDATES = ("IMERG", "C1M", "G1M", "G2M", "G2M-XCU")
FIGURE_NUMBER = 1
UTC_OFFSET_HOURS = -4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--figure-root", type=Path, default=FIGURES)
    p.add_argument("--periods", nargs="+", default=tuple(IOP_PERIODS), choices=tuple(IOP_PERIODS))
    return parse_configured_args(p, "make_iop_rain_error_metrics", argv=argv)


def read_rain(data_dir: Path, label: str) -> tuple[np.ndarray, np.ndarray]:
    """(time, lat, lon) hourly rain in mm/h and the UTC hours."""
    with xr.open_dataset(data_dir / f"{label}_common_valid.nc") as ds:
        values = np.asarray(ds["rainfall"].values, dtype=float)
        times = np.asarray(ds["time"].values, dtype="datetime64[h]")
    return values, times


def scores(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Bias, RMSE, centred RMSE, MAE and correlation over the finite pairs."""
    ok = np.isfinite(reference) & np.isfinite(candidate)
    ref, cand = reference[ok], candidate[ok]
    if ref.size < 2:
        return {k: float("nan") for k in ("n", "ref", "cand", "bias", "rmse", "crmse", "mae", "r")}
    error = cand - ref
    bias = float(error.mean())
    rmse = float(np.sqrt(np.mean(error**2)))
    r = float(np.corrcoef(ref, cand)[0, 1]) if ref.std() > 0 and cand.std() > 0 else float("nan")
    return {
        "n": float(ref.size),
        "ref": float(ref.mean()),
        "cand": float(cand.mean()),
        "bias": bias,
        "rmse": rmse,
        "crmse": float(np.sqrt(max(rmse**2 - bias**2, 0.0))),
        "mae": float(np.abs(error).mean()),
        "r": r,
    }


def nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    """``np.nanmean`` without the warning for the all-NaN cells outside the footprint."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


def domain_mean(values: np.ndarray) -> np.ndarray:
    """Area mean over the radar footprint, one value per hour."""
    return nanmean(values.reshape(values.shape[0], -1), axis=1)


def daily_mean(series: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Day mean of an hourly domain series, in mm/day, on local days."""
    local = times + np.timedelta64(UTC_OFFSET_HOURS, "h")
    days = local.astype("datetime64[D]")
    keys, index = np.unique(days, return_inverse=True)
    totals = np.zeros(keys.size)
    counts = np.zeros(keys.size)
    np.add.at(totals, index, np.where(np.isfinite(series), series, 0.0))
    np.add.at(counts, index, np.isfinite(series).astype(float))
    full = counts == 24  # whole local days only, so a day is not biased by its sampled hours
    return np.where(full, totals, np.nan), keys


def diurnal_cycle(series: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Mean of an hourly domain series by local hour of day."""
    hours = ((times.astype("datetime64[h]").astype(int) + UTC_OFFSET_HOURS) % 24).astype(int)
    out = np.full(24, np.nan)
    for hour in range(24):
        sel = hours == hour
        if sel.any():
            out[hour] = np.nanmean(series[sel])
    return out


def metric_rows(data_dir: Path) -> tuple[list[dict], list[str]]:
    """One row per candidate and aggregation level, plus a few notes."""
    ref, times = read_rain(data_dir, REFERENCE)
    ref_domain = domain_mean(ref)
    ref_daily, _ = daily_mean(ref_domain, times)
    ref_diurnal = diurnal_cycle(ref_domain, times)
    ref_map = nanmean(ref, axis=0)
    rows: list[dict] = []
    for label in CANDIDATES:
        cand, cand_times = read_rain(data_dir, label)
        if not np.array_equal(times, cand_times):
            raise ValueError(f"{label} is not on the radar's common-valid hours")
        cand_domain = domain_mean(cand)
        cand_daily, _ = daily_mean(cand_domain, times)
        levels = {
            "point hourly (mm/h)": scores(ref, cand),
            "domain hourly (mm/h)": scores(ref_domain, cand_domain),
            "domain daily (mm/day)": scores(ref_daily, cand_daily),
            "diurnal cycle (mm/h)": scores(ref_diurnal, diurnal_cycle(cand_domain, times)),
            "time-mean map (mm/h)": scores(ref_map, nanmean(cand, axis=0)),
        }
        for level, s in levels.items():
            rows.append({"candidate": label, "level": level, **s})
    notes = [
        f"Grid cells: {ref.shape[1]} x {ref.shape[2]}, of which "
        f"{int(np.isfinite(ref_map).sum())} are inside the radar footprint.",
        f"Common-valid hours: {times.size}; whole local days: "
        f"{int(np.isfinite(ref_daily).sum())} (local time = UTC{UTC_OFFSET_HOURS}).",
    ]
    return rows, notes


def table(rows: list[dict]) -> list[str]:
    """The metrics as a fixed-width table, one block per aggregation level."""
    head = (f"{'candidate':<9} {'n':>8} {'radar':>8} {'model':>8} {'bias':>8} {'bias %':>8} "
            f"{'RMSE':>8} {'RMSE %':>8} {'cRMSE':>8} {'MAE':>8} {'r':>6}")
    lines: list[str] = []
    for level in dict.fromkeys(r["level"] for r in rows):
        lines += [level, "-" * len(head), head]
        for r in (r for r in rows if r["level"] == level):
            rel = 100.0 * r["bias"] / r["ref"] if r["ref"] else float("nan")
            nrmse = 100.0 * r["rmse"] / r["ref"] if r["ref"] else float("nan")
            lines.append(
                f"{r['candidate']:<9} {int(r['n']):>8d} {r['ref']:>8.3f} {r['cand']:>8.3f} "
                f"{r['bias']:>+8.3f} {rel:>+8.1f} {r['rmse']:>8.3f} {nrmse:>8.1f} "
                f"{r['crmse']:>8.3f} {r['mae']:>8.3f} {r['r']:>6.3f}"
            )
        lines.append("")
    return lines


def write_metrics(rows: list[dict], notes: list[str], out: Path, period: str) -> Path:
    """Write the table, the method and a machine-readable block."""
    iops = IOP_PERIODS[period]
    names = " and ".join(f"IOP{iop}" for iop in iops)
    title = f"Rain error metrics against the SIPAM radar, {period}"
    lines = [
        title,
        "=" * len(title),
        f"Period: GoAmazon2014/5 {names}{' combined' if len(iops) > 1 else ''} "
        "(Martin et al. 2016, ACP 16, 4785).",
    ]
    for iop in iops:
        start, end = GOAMAZON_IOPS[iop]
        lines.append(f"IOP{iop}: {start} to {end} inclusive.")
    lines += notes
    lines += [
        "Reference: SIPAM radar. Candidates: GPM IMERG and the four runs, all on the",
        "0.1 deg IMERG grid with the radar valid-time mask and the radar footprint.",
        "bias = mean(candidate - radar); RMSE = root mean square of that difference;",
        "cRMSE = centred RMSE, the RMSE after the mean bias is removed, so",
        "RMSE^2 = bias^2 + cRMSE^2; MAE = mean absolute difference; r = Pearson",
        "correlation of the pairs. 'bias %' and 'RMSE %' are relative to the radar mean.",
        "Point hourly scores every cell and hour, so a shower in the wrong cell is",
        "penalised twice; the domain and diurnal rows remove that double penalty.",
        "The bias here is a domain aggregate, mean(candidate) - mean(radar) over all the",
        "pairs. The bias map of figure 1 instead takes the relative bias of each cell and",
        "then averages the cells, which weights the dry edge of the footprint more, so its",
        "percentages are a little larger. Both are correct; they answer different questions.",
        "",
    ]
    lines += table(rows)
    lines += ["csv", "---", "candidate,level,n_pairs,radar_mean,candidate_mean,bias,rmse,"
              "centred_rmse,mae,pearson_r"]
    for r in rows:
        lines.append(
            f"{r['candidate']},{r['level']},{int(r['n'])},{r['ref']:.6g},{r['cand']:.6g},"
            f"{r['bias']:.6g},{r['rmse']:.6g},{r['crmse']:.6g},{r['mae']:.6g},{r['r']:.6g}"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for period in args.periods:
        data_dir = args.work_dir / PERIODS[period]
        rows, notes = metric_rows(data_dir)
        _, _, txt = figure_files(args.figure_root, FIGURE_NUMBER, "spatial_bias_vs_radar", period)
        out = write_metrics(rows, notes, txt.with_name(f"rain_error_metrics_{period}.txt"), period)
        print(f"[saved] {out}", flush=True)
        for r in rows:
            if r["level"].startswith("domain hourly"):
                print(f"  {r['candidate']:<8} domain hourly  bias {r['bias']:+.3f}  "
                      f"RMSE {r['rmse']:.3f} mm/h  r {r['r']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
