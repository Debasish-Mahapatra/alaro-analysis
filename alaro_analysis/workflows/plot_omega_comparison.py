"""
One-off comparison plot: UD_OMEGA, UD_MESH_FRAC, their product (effective
grid-scale updraft omega contribution), and VERT_VELOCIT, all at a mid-
tropospheric level from a single 2mom FA file.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import faxarray as fx

FA_FILE = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO/2mom/untar-output"
    "/pf20140101/pfABOFABOF+0015"
)
OUT_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/convection_omega_comparison"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEVEL_IDX = 51  # ~500 hPa

def _load_level(var: str, level: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    ds = fx.open_dataset(str(FA_FILE), variables=[var], stack_levels=True)
    try:
        da = ds[var].isel(time=0, level=level)
        values = np.asarray(da.values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        attrs = dict(da.attrs)
    finally:
        ds.close()
    return values, lon, lat, attrs


def _sym_limit(values: np.ndarray, pct: float = 99.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return float(np.nanpercentile(np.abs(finite), pct))


def _strip_zeros(cbar) -> None:
    cbar.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))


def main() -> None:
    ud_omega, lon, lat, _ = _load_level("UD_OMEGA", LEVEL_IDX)
    ud_frac, _, _, _ = _load_level("UD_MESH_FRAC", LEVEL_IDX)
    vert_w, _, _, _ = _load_level("VERT_VELOCIT", LEVEL_IDX)
    # pressure for the title
    pres, _, _, _ = _load_level("PRESSURE", LEVEL_IDX)
    p_hpa = float(np.nanmean(pres) / 100.0)

    effective = ud_omega * ud_frac

    # limits
    lim_om = _sym_limit(ud_omega)
    lim_eff = _sym_limit(effective)
    lim_w = _sym_limit(vert_w)

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 10),
        sharex=True, sharey=True,
        constrained_layout=True,
    )

    panels = [
        ("UD_OMEGA (within-updraft)", ud_omega, "RdBu_r", -lim_om, lim_om, "Pa s$^{-1}$ (negative = upward)"),
        ("UD_MESH_FRAC (updraft area fraction)", ud_frac, "viridis", 0.0, max(1e-6, float(np.nanmax(ud_frac))), "fraction"),
        ("UD_OMEGA × UD_MESH_FRAC (effective updraft)", effective, "RdBu_r", -lim_eff, lim_eff, "Pa s$^{-1}$ (negative = upward)"),
        ("VERT_VELOCIT (resolved grid-scale)", vert_w, "RdBu_r", -lim_w, lim_w, "m s$^{-1}$ (positive = upward)"),
    ]

    for ax, (title, data, cmap, vmin, vmax, cbar_label) in zip(axes.flat, panels):
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
        cbar.set_label(cbar_label, fontsize=9)
        _strip_zeros(cbar)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"ALARO 2mom — updraft vs resolved vertical motion at ~{p_hpa:.0f} hPa",
        fontsize=13,
    )

    out_file = OUT_DIR / f"ud_omega_vs_vertvel_2mom_L{LEVEL_IDX}.png"
    fig.savefig(out_file, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
