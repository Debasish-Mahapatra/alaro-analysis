#!/usr/bin/env python3
"""
Standalone CFAD plot of hydrometeor mass fraction (full 2 years, from raw masked NetCDF).

Contoured Frequency by Altitude Diagrams (CFADs) of hydrometeor mass fraction
for a set of model runs. The figure is a grid with one row per hydrometeor
species and one column per model run; each panel shows the per-level frequency
(%) distribution of the mass fraction against height, with the per-level median
mass fraction overplotted.

ALARO outputs MASS FRACTION (kg/kg), not mixing ratio, so axes are labelled
accordingly.  Row order: graupel, snow, ice, liquid water, rain
("ice" = cloud ice = the model's SOLID_WATER field).

This is the standalone notebook script, adapted ONLY in its data layer to read
the project's masked-netcdf, which is stored as per-species directories of
per-hour files (ALARO/<run>/masked-netcdf/<SPECIES>/pf<YYYYMMDD>/*+NNNN.nc)
rather than one merged file per run.  The scientific logic -- log-spaced
mass-fraction bins, per-level frequency normalisation to 100 %, log-norm colour
scale, median overlay, and the whole plot_cfads layout -- is preserved verbatim.

Two adaptations forced by the per-hour 2-year layout (everything else verbatim):
  * compute_model_cfad accumulates the per-level np.histogram across every
    hourly file (histograms are additive, so the counts are exact); forecast
    leads are restricted to +0000..+0023 to avoid double-counting hour 0.
  * the per-level median is read from the accumulated histogram (the exact
    np.median over all positive values cannot be held in memory at 2-year scale;
    binned over 71 log bins it is visually identical on the log x-axis).
  * heights come from the masked GEOPOTENTIEL field, which is ALREADY in metres
    (units "m"; converted from geopotential during masking), so -- unlike the
    notebook's WRF path -- it must NOT be divided by g again.
"""

import argparse
import os
import re
import glob

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")  # headless-safe; remove if you want an interactive window
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from concurrent.futures import ProcessPoolExecutor


# --------------------------------------------------------------------------- #
# Colormap
# --------------------------------------------------------------------------- #
def load_colormap():
    """Return the White-Blue-Green-Yellow-Red colormap used in the notebook.

    Falls back to a perceptually-ordered matplotlib colormap if the ``cmaps``
    package is unavailable.
    """
    try:
        import cmaps
        return cmaps.WhBlGrYeRe
    except Exception:
        # 'turbo' is a reasonable, perceptually-ordered stand-in.
        return plt.get_cmap("turbo")


# --------------------------------------------------------------------------- #
# Default data layout (3 model runs, short codes as in the LaTeX table)
#   C1M = control (1-moment)   G1M = graupel (1-moment)   G2M = 2-moment
#
# Adapted: each run is a masked-netcdf base dir holding per-species sub-dirs of
# per-hour files, plus a GEOPOTENTIEL sub-dir for heights.
# --------------------------------------------------------------------------- #
def build_data_paths(data_root):
    """Build the {model: {base, geopo}} mapping rooted at ``data_root`` (=ALARO)."""
    def run(name):
        base = os.path.join(data_root, name, "masked-netcdf")
        return {"base": base, "geopo": os.path.join(base, "GEOPOTENTIEL")}
    return {"C1M": run("control"), "G1M": run("graupel"), "G2M": run("2mom")}


# Candidate variable names per species — ROW ORDER: graupel, snow, ice, liquid water, rain
# ("ice" = cloud ice = the model's SOLID_WATER field)
MIXING_CANDIDATES = {
    "graupel": ["GRAUPEL", "graupel", "QGRAUPEL", "qg"],
    "snow": ["SNOW", "snow", "QSNOW", "qs"],
    "ice": ["SOLID_WATER", "solid_water", "SOLID", "SOLIDWATER"],
    "liquid water": ["LIQUID_WATER", "liquid_water", "QCLOUD", "qc", "cloud", "CLOUD_WATER"],
    "rain": ["RAIN", "rain", "QRAIN", "RAIN_MMR", "rain_mmr", "qr", "q_r"],
}

# masked-netcdf sub-directory name for each species row.
SPECIES_DIRNAME = {
    "graupel": "GRAUPEL",
    "snow": "SNOW",
    "ice": "SOLID_WATER",
    "liquid water": "LIQUID_WATER",
    "rain": "RAIN",
}

# Candidate variable names for the geopotential / height field
GEO_CANDIDATES = ["GEOPOTENTIEL", "PH", "PHB", "Z"]

# Mass-fraction histogram bin edges (kg/kg), log-spaced 1e-10 .. 1e-1
VAL_BINS = np.logspace(-10, -1, 72)

# Forecast-lead window (hours). +0024 duplicates the next day's +0000, so 0..23.
MIN_LEAD, MAX_LEAD = 0, 23
_LEAD_RE = re.compile(r"\+(\d{4})\.nc$")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def find_var(ds, candidates):
    """Return the first variable in ``ds`` matching any candidate (case-insensitive)."""
    keys = {k.lower(): k for k in ds.variables}
    for c in candidates:
        if c in ds.variables:
            return c
        if c.lower() in keys:
            return keys[c.lower()]
    return None


def get_vdim(var):
    """Return the name of the vertical dimension of ``var``."""
    for d in ["level", "lev", "z", "height", "zg", "bottom_top"]:
        if d in var.dims:
            return d
    if len(var.dims) >= 3:
        return var.dims[1]
    return None


def list_hour_files(species_dir):
    """All per-hour files for one (run, species), leads +0000..+0023, sorted by day+lead."""
    out = []
    for day in sorted(glob.glob(os.path.join(species_dir, "pf*"))):
        for f in sorted(glob.glob(os.path.join(day, "*.nc"))):
            m = _LEAD_RE.search(os.path.basename(f))
            if m and MIN_LEAD <= int(m.group(1)) <= MAX_LEAD:
                out.append(f)
    return out


def heights_metres_from_geopotential(geopo_dir, geo_cands, vdim, nlev):
    """Per-level height profile (m) from a representative masked GEOPOTENTIEL file.

    The masked GEOPOTENTIEL field is ALREADY geopotential height in metres
    (NetCDF units 'm', converted by /g during masking), so -- unlike the
    notebook's raw-geopotential / WRF PH+PHB path -- it is used directly.
    """
    files = list_hour_files(geopo_dir)
    if not files:
        return None
    with xr.open_dataset(files[0], decode_times=False) as ds:
        geon = find_var(ds, geo_cands)
        if geon is None:
            return None
        g = ds[geon]
        other = [d for d in g.dims if d != vdim]
        prof = g.mean(dim=other).values if other else g.values
    prof = np.asarray(prof, dtype=np.float64).ravel()
    return prof if prof.size == nlev else None


# --------------------------------------------------------------------------- #
# CFAD computation for a single (model, species) -- accumulated over 2 years
# --------------------------------------------------------------------------- #
def compute_model_cfad(species_dir, geopo_dir, hy_cands, geo_cands, bins):
    """Per-level histogram counts, totals and median for one field, over all files.

    Returns ``(counts, bins, heights, var_name, total_per_level, median_mr)``
    or ``None`` if the field/files could not be read.
    """
    files = list_hour_files(species_dir)
    if not files:
        print("No files under", species_dir)
        return None

    # Variable name, vertical dim and level count from the first file.
    try:
        with xr.open_dataset(files[0], decode_times=False) as ds0:
            var_name = find_var(ds0, hy_cands)
            if var_name is None:
                return None
            vdim = get_vdim(ds0[var_name])
            if vdim is None:
                return None
            nlev = int(ds0[var_name].sizes[vdim])
            axis = ds0[var_name].dims.index(vdim)
    except Exception as e:
        print("Could not open", files[0], e)
        return None

    # Heights (metres) from the geopotential dir, then fall back to index.
    heights = None
    if geopo_dir:
        try:
            heights = heights_metres_from_geopotential(geopo_dir, geo_cands, vdim, nlev)
        except Exception:
            heights = None
    if heights is None:
        heights = np.arange(nlev, dtype=np.float64)

    nb = len(bins) - 1
    counts = np.zeros((nlev, nb), dtype=np.int64)
    total_per_level = np.zeros(nlev, dtype=np.int64)

    # Tag for progress lines, e.g. "control/RAIN".
    run_tag = os.path.basename(os.path.dirname(os.path.dirname(species_dir)))
    tag = f"{run_tag}/{os.path.basename(species_dir)}"
    n_files = len(files)
    print(f"[{tag}] start: {n_files} files", flush=True)

    # Accumulate the per-level histogram across every hourly file (additive).
    for k, path in enumerate(files, 1):
        try:
            with xr.open_dataset(path, decode_times=False) as ds:
                data = np.asarray(ds[var_name].values, dtype=np.float64)
        except Exception as e:
            print("skip", path, e)
            continue
        if k % 2000 == 0 or k == n_files:
            print(f"[{tag}] {k}/{n_files} files", flush=True)
        # Move the vertical axis to the front, flatten the rest -> (nlev, npts).
        flat = np.moveaxis(data, axis, 0).reshape(nlev, -1)
        for iz in range(nlev):
            arr = flat[iz]
            arr = arr[np.isfinite(arr)]
            arr = np.maximum(arr, 0)
            arr_pos = arr[arr > 0]
            if arr_pos.size:
                counts[iz, :] += np.histogram(arr_pos, bins=bins)[0]
                total_per_level[iz] += arr_pos.size

    # Per-level median read off the accumulated cumulative histogram (binned;
    # exact np.median over all positive values is infeasible at 2-year scale).
    centers = 0.5 * (bins[:-1] + bins[1:])
    median_mr = np.full(nlev, np.nan)
    cum = np.cumsum(counts, axis=1)
    for iz in range(nlev):
        if total_per_level[iz] > 0:
            half = 0.5 * cum[iz, -1]
            median_mr[iz] = centers[min(int(np.searchsorted(cum[iz], half)), centers.size - 1)]

    return counts, bins, heights, var_name, total_per_level, median_mr


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def compute_all(data_paths, workers):
    """Run ``compute_model_cfad`` for every (species, model) pair in parallel."""
    jobs = []
    for s, cands in MIXING_CANDIDATES.items():
        for m, p in data_paths.items():
            species_dir = os.path.join(p["base"], SPECIES_DIRNAME[s])
            jobs.append((m, s, species_dir, p.get("geopo", None), cands))

    results = {}
    n_workers = max(1, min(workers, len(jobs)))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(
                compute_model_cfad,
                j[2],
                j[3],
                j[4],
                GEO_CANDIDATES if j[3] else [],
                VAL_BINS,
            ): (j[0], j[1])
            for j in jobs
        }
        for fut, key in futures.items():
            m, s = key
            results.setdefault(s, {})[m] = fut.result()
            print(f"[done] {s} / {m}", flush=True)
    return results


def plot_cfads(results, data_paths, cmap, save_path):
    """Build and save the species x model CFAD grid figure."""
    # Determine global vmax for a shared frequency (%) colour scale.
    global_max = 0
    for s in results:
        for m in results[s]:
            ent = results[s][m]
            if ent is None:
                continue
            counts = ent[0]
            total_per_level = ent[4]
            if counts is not None and counts.size > 0:
                frequency = 100.0 * counts / np.maximum(total_per_level[:, np.newaxis], 1)
                global_max = max(global_max, frequency.max())
    global_max = max(global_max, 0.1)

    models = list(data_paths.keys())
    species = list(MIXING_CANDIDATES.keys())
    fig, axes = plt.subplots(
        len(species),
        len(models),
        figsize=(4 * len(models), 3 * len(species)),
        squeeze=False,
    )

    # Column titles = model names
    for j, m in enumerate(models):
        axes[0][j].set_title(m, fontsize=12, fontweight="bold", pad=10)

    for i, s in enumerate(species):
        for j, m in enumerate(models):
            ax = axes[i][j]
            ent = results.get(s, {}).get(m)
            if ent is None:
                ax.axis("off")
                continue

            counts, bins, heights, varname, total_per_level, median_mr = ent

            frequency = 100.0 * counts / np.maximum(total_per_level[:, np.newaxis], 1)

            if np.all(frequency == 0):
                ax.set_xlim(VAL_BINS[0], VAL_BINS[-1])
                ax.set_ylim(0, 20)
                ax.set_xscale("log")
                ax.text(
                    0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="gray",
                )
                ax.set_xlabel("Mass fraction (kg kg⁻¹)")
                ax.set_ylabel("Height (km)")
                continue

            heights_km = (heights - np.nanmin(heights)) / 1000.0
            centers = 0.5 * (bins[:-1] + bins[1:])
            X, Y = np.meshgrid(centers, heights_km)

            # 1. CFAD
            mesh = ax.pcolormesh(
                X, Y, frequency,
                norm=LogNorm(vmin=0.01, vmax=global_max),
                shading="auto", cmap=cmap,
            )

            # 2. Median mass-fraction line
            valid_median_mr = median_mr[~np.isnan(median_mr)]
            valid_heights_km = heights_km[~np.isnan(median_mr)]
            if valid_median_mr.size > 0 and np.any(valid_median_mr > 0):
                ax.plot(
                    valid_median_mr, valid_heights_km,
                    color="black", linewidth=2, linestyle="-", label="Median",
                )
                # 3. Per-subplot legend
                ax.legend(loc="upper right", frameon=True, fontsize=8)

            ax.set_xscale("log")
            ax.set_xlabel("Mass fraction (kg kg⁻¹)")
            ax.set_ylabel("Height (km)")
            ax.set_ylim(0, 20)
            cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.04, pad=0.03)
            cbar.set_label("Frequency (%)")

        # Row label on the left
        axes[i][0].set_ylabel(s.capitalize(), fontsize=12, fontweight="bold")

    plt.tight_layout(pad=2.5, w_pad=1.8, h_pad=1.8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=450, bbox_inches="tight")
    print(f"Saved to {save_path}")
    return fig


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot CFADs of hydrometeor mass fraction (species x model grid), "
                    "full 2 years from the raw masked-netcdf.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-root", default="/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO",
        help="Root containing <run>/masked-netcdf/<SPECIES>/pf*/*.nc",
    )
    p.add_argument(
        "--output-dir",
        default="/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/6. hydrometeor mixing-ratio cfad",
        help="Directory to write the figure into",
    )
    p.add_argument(
        "--output-name",
        default="cfad-mass-fraction-frequency_with-median_2yr.png",
        help="Output figure filename",
    )
    p.add_argument(
        "--workers", type=int, default=15,
        help="Max parallel worker processes (one per (species, model) job; 15 jobs)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    data_paths = build_data_paths(args.data_root)
    if not any(os.path.isdir(p["base"]) for p in data_paths.values()):
        raise SystemExit(f"No masked-netcdf runs found under {args.data_root}")

    cmap = load_colormap()
    results = compute_all(data_paths, args.workers)
    save_path = os.path.join(args.output_dir, args.output_name)
    plot_cfads(results, data_paths, cmap, save_path)


if __name__ == "__main__":
    main()
