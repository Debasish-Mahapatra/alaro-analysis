"""Iop precipitation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from alaro_analysis.common.iop_settings import CONT, LABEL, OFFSET, OUT
from alaro_analysis.data.iop_precipitation import fa, fpath


def plot_pair(i, e, dates, other, cont):
    base = f"iop{i}_{e}_smoke"
    OUT.mkdir(parents=True, exist_ok=True)
    m = np.nanmean([x["rain"] for x in other], axis=0)
    c = np.nanmean([x["rain"] for x in cont], axis=0)
    delta = m - c
    # Native FA lon/lat are read once; no crop or radar mask is applied.
    rr = fa(fpath(CONT / e / "untar-output", dates[0], 0))
    try:
        geom = rr.geometry
        lon = np.asarray(geom.get_lonlat_grid()[0])
        lat = np.asarray(geom.get_lonlat_grid()[1])
    finally:
        rr.close()
    absmax = float(np.nanpercentile(np.stack([c, m]), 99.5))
    dmax = float(np.nanpercentile(np.abs(delta), 99.5))
    dmax = max(dmax, 1e-12)
    fig, ax = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    for j, (a, z, t, cm, norm) in enumerate(
        zip(
            ax,
            (c, m, delta),
            (f"{LABEL[e]} continuous", f"{LABEL[e]} IOP", f"{LABEL[e]} IOP − continuous"),
            ("viridis", "viridis", "RdBu_r"),
            (None, None, None),
        )
    ):
        vmin, vmax = (0, absmax) if j < 2 else (-dmax, dmax)
        p = a.pcolormesh(lon, lat, z, cmap=cm, vmin=vmin, vmax=vmax, shading="auto")
        a.set_title(t)
        a.set_xlabel("longitude (deg)")
        a.set_ylabel("latitude (deg)")
        cb = fig.colorbar(p, ax=a)
        cb.set_label("mm day$^{-1}$")
        cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    out = OUT / f"{base}_rainfall_maps.png"
    fig.savefig(out, dpi=450)
    plt.close(fig)
    # Hourly fields are UTC interval starts; reorder to local UTC-4 display.
    order = np.argsort((np.arange(24) + OFFSET) % 24)
    lh = (np.arange(24) + OFFSET) % 24
    cd = np.nanmean([x["hourly"].mean(axis=(1, 2)) for x in cont], axis=0)
    md = np.nanmean([x["hourly"].mean(axis=(1, 2)) for x in other], axis=0)
    dd = md - cd
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(lh[order], cd[order], label=f"{LABEL[e]} continuous")
    ax.plot(lh[order], md[order], label=f"{LABEL[e]} IOP")
    ax.plot(lh[order], dd[order], label="IOP − continuous")
    ax.set_xlabel("local hour (UTC-4)")
    ax.set_ylabel("total precipitation (mm h$^{-1}$)")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.3)
    ax.legend()
    out2 = OUT / f"{base}_rainfall_diurnal.png"
    fig.savefig(out2, dpi=450)
    plt.close(fig)
    # Profiles have columns endpoint hours 1..24; display those endpoints in local time.
    z = np.nanmean([x["height"] for x in cont], axis=0) / 1000
    profout = []
    fig, ax = plt.subplots(3, 3, figsize=(20, 18), constrained_layout=True)
    for r, k, unit in (
        (0, "extent", "fraction"),
        (1, "flux", "kg m$^{-2}$ s$^{-1}$"),
        (2, "cloud", "fraction"),
    ):
        cv = np.nanmean([x[k] for x in cont], axis=0)
        iv = np.nanmean([x[k] for x in other], axis=0)
        dv = iv - cv
        av = float(np.nanpercentile(np.stack([cv, iv]), 99))
        q = max(float(np.nanpercentile(np.abs(dv), 99)), 1e-12)
        for j, (v, cm, vmin, vmax, title) in enumerate(
            (
                (cv, "viridis", 0, av, f"{LABEL[e]} continuous"),
                (iv, "viridis", 0, av, f"{LABEL[e]} IOP"),
                (dv, "RdBu_r", -q, q, f"{LABEL[e]} IOP − continuous"),
            )
        ):
            p = ax[r, j].pcolormesh(
                lh[order], z, v[order, :].T, cmap=cm, vmin=vmin, vmax=vmax, shading="auto"
            )
            ax[r, j].set_ylim(0, 16)
            ax[r, j].set_xlabel("local endpoint hour (UTC-4)")
            ax[r, j].set_ylabel("height (km)")
            ax[r, j].set_title(f"{title}: {k}")
            cb = fig.colorbar(p, ax=ax[r, j])
            cb.set_label(unit)
            cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
            if j < 2:
                run = cont if j == 0 else other
                tt = np.nanmean([x["temp"] for x in run], axis=0)
                zz = np.nanmean([x["height"] for x in run], axis=0) / 1000
                freeze = []
                for h in range(24):
                    good = np.isfinite(zz) & np.isfinite(tt[h])
                    orderz = np.argsort(zz[good])
                    zs = zz[good][orderz]
                    ts = tt[h][good][orderz]
                    cross = np.where(np.diff(np.sign(ts - 273.15)) != 0)[0]
                    freeze.append(
                        float(
                            np.interp(
                                273.15,
                                ts[cross[0] : cross[0] + 2][::-1],
                                zs[cross[0] : cross[0] + 2][::-1],
                            )
                        )
                        if cross.size
                        else np.nan
                    )
                ax[r, j].plot(
                    lh[order],
                    np.asarray(freeze)[order],
                    color="black",
                    ls="--",
                    lw=1.2,
                    label="0 °C",
                )
                if r == 0 and j == 0:
                    ax[r, j].legend(loc="upper right", fontsize=8)
        profout.append((k, cv, iv, dv))
    out3 = OUT / f"{base}_updraft.png"
    fig.savefig(out3, dpi=450)
    plt.close(fig)
    header = f"IOP {i} {e} ({LABEL[e]}); dates={','.join(x.isoformat() for x in dates)}; n_dates={len(dates)}; full native grid={m.shape}; sources=raw FA continuous and IOP; primary delta=IOP-continuous\n"
    (OUT / f"{base}_rainfall_maps.txt").write_text(
        header
        + "lon\n"
        + np.array2string(lon, threshold=np.inf)
        + "\nlat\n"
        + np.array2string(lat, threshold=np.inf)
        + "\ncontinuous_mm_day\n"
        + np.array2string(c, threshold=np.inf)
        + "\nIOP_mm_day\n"
        + np.array2string(m, threshold=np.inf)
        + "\ndelta_mm_day\n"
        + np.array2string(delta, threshold=np.inf)
    )
    (OUT / f"{base}_rainfall_diurnal.txt").write_text(
        header
        + f"local_hours={lh[order].tolist()}\ncontinuous_mm_h={cd[order].tolist()}\nIOP_mm_h={md[order].tolist()}\ndelta_mm_h={dd[order].tolist()}"
    )
    with (OUT / f"{base}_updraft.txt").open("w") as f:
        f.write(header + f"height_km={z.tolist()}\nlocal_endpoint_hours={lh[order].tolist()}\n")
        for k, cv, iv, dv in profout:
            f.write(
                f"## {k}\ncontinuous_{k}\n{np.array2string(cv[order, :], threshold=np.inf)}\nIOP_{k}\n{np.array2string(iv[order, :], threshold=np.inf)}\ndelta_{k}\n{np.array2string(dv[order, :], threshold=np.inf)}\n"
            )
    return out, out2, out3
