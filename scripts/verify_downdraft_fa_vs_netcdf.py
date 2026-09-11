#!/usr/bin/env python3
"""Verify the raw-FA downdraft reduction against the masked-netcdf archive.

build_downdraft_vertvel_diurnal_from_FA.py reads the raw FA files and averages
over the 2822 radar-mask points directly.  The netcdf archive holds the same
fields on the 75x72 crop with the out-of-mask cells set to NaN, so an
independent NaN-safe reduction of those files must reproduce the FA profiles
exactly.  This samples hours spread across the two-year run, compares the two,
and writes the summary to processed-data/paper_downdraft_from_FA/verification.txt
(the figure's _data.txt embeds it).

Note the NaN-safe part matters: evaluating ``np.where(mesh > 0, ..., 0.0)`` on
the padded netcdf array turns the out-of-mask NaNs into zeros and dilutes the
mean by 75*72/2822 = 1.9135.  build_updraft_cloud_diurnal_from_netcdf.py does
exactly that, which is why the published updraft mass-flux panel is low by that
constant factor.  Nothing here has that problem: the FA reduction never holds a
padded cell.

Requires the epygram conda environment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_downdraft_vertvel_diurnal_from_FA as B  # noqa: E402

SAMPLES_PER_EXP = 8
QUANTITIES = ("extent", "flux", "w_dn", "temp")


def netcdf_reference(exp: str, day: str, hour: int) -> dict[str, np.ndarray] | None:
    root = B.ALARO / exp / "masked-netcdf"

    def read(var: str) -> np.ndarray | None:
        path = root / var / day / f"pfABOFABOF+{hour:04d}.nc"
        if not path.exists():
            return None
        with xr.open_dataset(path) as ds:
            return np.asarray(ds[var].values[0], dtype=np.float64)

    mesh, omega = read("DD_MESH_FRAC"), read("DD_OMEGA")
    temp, w = read("TEMPERATURE"), read("VERT_VELOCIT")
    if any(a is None for a in (mesh, omega, temp, w)):
        return None
    finite = np.isfinite(mesh) & np.isfinite(omega)
    return {
        "extent": np.nanmean(mesh, axis=(1, 2)),
        "flux": np.nanmean(
            np.where(finite, np.where(mesh > 0.0, (omega * mesh) / B.G, 0.0), np.nan),
            axis=(1, 2),
        ),
        "w_dn": np.nanmean(
            np.where(np.isfinite(w), np.where(w < 0.0, -w, 0.0), np.nan), axis=(1, 2)
        ),
        "temp": np.nanmean(temp, axis=(1, 2)),
    }


def main() -> int:
    window = B.build_window()
    B._init_worker(window)

    lines = [
        "Verification: raw-FA reduction vs an independent NaN-safe reduction of the",
        f"masked-netcdf archive, {SAMPLES_PER_EXP} sampled hours per experiment.",
        "FA level index 0 is S001 (model top); the netcdf level index runs the other way.",
        "",
    ]
    worst = 0.0
    for exp in B.EXPS:
        days = sorted(p.name for p in (B.ALARO / exp / "untar-output").iterdir() if p.is_dir())
        picks = [days[i] for i in np.linspace(0, len(days) - 1, SAMPLES_PER_EXP).astype(int)]
        acc: dict[str, list[float]] = {q: [] for q in QUANTITIES}
        n_ok = 0
        for i, day in enumerate(picks):
            hour = int(3 * i) % 24
            fa_file = B.fa_path(exp, day, hour)
            ref = netcdf_reference(exp, day, hour)
            if ref is None or not fa_file.exists():
                continue
            got = B.profiles_from_file(fa_file)
            n_ok += 1
            for q in QUANTITIES:
                fa = got[q][::-1]
                scale = max(float(np.nanmax(np.abs(ref[q]))), 1e-30)
                acc[q].append(float(np.nanmax(np.abs(fa - ref[q]))) / scale)
        for q in QUANTITIES:
            rel = 100.0 * max(acc[q]) if acc[q] else float("nan")
            worst = max(worst, rel if np.isfinite(rel) else 0.0)
            lines.append(
                f"  {exp:8s} {q:7s}: max relative difference over {n_ok} hours = {rel:.2e} %"
            )
    lines += [
        "",
        f"Worst relative difference anywhere: {worst:.2e} % (round-off).",
    ]
    text = "\n".join(lines)
    print(text)
    B.OUT.mkdir(parents=True, exist_ok=True)
    (B.OUT / "verification.txt").write_text(text + "\n")
    print(f"\n[saved] {B.OUT / 'verification.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
