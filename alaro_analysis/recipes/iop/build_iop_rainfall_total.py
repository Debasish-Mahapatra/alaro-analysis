#!/usr/bin/env python3
"""Hourly total surface rainfall for one IOP experiment.

RAINFALL_TOTAL = SURFPREC.EAU.CON + SURFPREC.EAU.GEC, both already deaccumulated
to hourly amounts by ``alaro-deaccumulate-precipitation``. One tree is written
next to the components in every ``*-netcdf-deaccumulated`` root of the experiment:

    <exp_dir>/<domain>-netcdf-deaccumulated/RAINFALL_TOTAL/pfYYYYMMDD/pfABOFABOF+HHHH.nc

Values are kg m-2 over the hour ending at the file's lead (i.e. mm h-1). Lead
+0000 is 0 because accumulation restarts at each day's forecast start, so a
day's rain is leads +0001..+0024. Existing outputs are skipped; files are
written atomically.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import xarray as xr

from alaro_analysis.common.cli_config import parse_configured_args

COMPONENTS = ("SURFPREC.EAU.CON", "SURFPREC.EAU.GEC")
OUT_VAR = "RAINFALL_TOTAL"


def total_day(task):
    root, day = task
    con_dir, gec_dir = (root / name / day for name in COMPONENTS)
    out_dir = root / OUT_VAR / day
    written = 0
    try:
        con_files = sorted(con_dir.glob("*.nc"))
        gec_names = sorted(p.name for p in gec_dir.glob("*.nc"))
        if [p.name for p in con_files] != gec_names:
            raise ValueError(f"lead files differ between {COMPONENTS[0]} and {COMPONENTS[1]}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for con_file in con_files:
            out_file = out_dir / con_file.name
            if out_file.exists():
                continue
            gec_file = gec_dir / con_file.name
            with (
                xr.open_dataset(con_file, decode_times=False) as con,
                xr.open_dataset(gec_file, decode_times=False) as gec,
            ):
                # NaN outside the radar mask stays NaN in the sum.
                total = con[COMPONENTS[0]] + gec[COMPONENTS[1]]
                total.attrs = {
                    "long_name": "Hourly total surface rainfall (convective + stratiform)",
                    "units": "kg m-2",
                    "comment": "Amount over the hour ending at this lead (mm h-1); "
                    f"sum of deaccumulated {COMPONENTS[0]} and {COMPONENTS[1]}.",
                }
                out = total.to_dataset(name=OUT_VAR)
                out.attrs = dict(con.attrs)
                out.attrs["source_components"] = f"{con_file} + {gec_file}"
                tmp_file = out_file.with_name(out_file.name + ".tmp")
                out.to_netcdf(tmp_file, encoding={OUT_VAR: {"zlib": True, "complevel": 1}})
            tmp_file.replace(out_file)
            written += 1
        return root.name, day, written, None
    except Exception as exc:  # noqa: BLE001
        return root.name, day, written, repr(exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "exp_dir", type=Path, help="<iop>/<exp> directory holding *-netcdf-deaccumulated trees"
    )
    ap.add_argument("--workers", type=int, default=32)
    args = parse_configured_args(ap, "build_iop_rainfall_total", argv=None)

    tasks = [
        (root, day.name)
        for root in sorted(args.exp_dir.glob("*-netcdf-deaccumulated"))
        for day in sorted((root / COMPONENTS[0]).glob("pf2*"))
        if day.is_dir()
    ]
    if not tasks:
        raise SystemExit(f"no deaccumulated precipitation found under {args.exp_dir}")

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(total_day, tasks))
    failed = [r for r in results if r[3] is not None]
    for root_name, day, _, err in failed:
        print(f"FAILED {root_name}/{day}: {err}", flush=True)
    print(
        f"{args.exp_dir}: {len(tasks)} domain-days, {sum(r[2] for r in results)} "
        f"{OUT_VAR} files written, {len(failed)} failed",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
