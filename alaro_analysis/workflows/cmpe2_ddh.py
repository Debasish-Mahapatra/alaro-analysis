"""Sui (2007) CMPE2 per lead hour, fully from DDH.

CMPE2 = P_s / ([SI_qv] + sgn(Q_CM)·Q_CM)          (Sui 2007 Eq. 2)

With every term drawn from the SAME DDH column so the budget closes:

    [SI_qv] = − ∫ ρ·(condcv + condrs) dz      column condensation + deposition
    [SO_qv] =   ∫ ρ·(evapcv + evaprs) dz      column evaporation + melting-as-evap
    P_s     = − ∫ ρ·(prec-cv + prec-rs)_QR dz + same for QS + QG
              (negative of the column sedimentation tendency = flux exiting
              the column through the surface)

The hydrometeor budget identity Q_CM = P_s − [SI_qv] + [SO_qv] lets us keep
everything in terms of these three column integrals, so sgn(Q_CM)·Q_CM is
trivially computable.  Because all three are from the same DDH column, the
Sui-bounded relation CMPE2 ≤ 100% holds by construction.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DDH_AGG   = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/_aggregated")
OUT_NPZ   = Path("/gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/cmpe2_ddh/cmpe2_ddh.npz")
OUT_FIG   = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/cmpe2_ddh/cmpe2_ddh_diurnal.png")
OUT_DIAG  = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/cmpe2_ddh/cmpe2_ddh_diagnosis.png")

EXPERIMENTS = ("control", "graupel", "2mom")
LEADS       = [f"{h:04d}" for h in range(1, 25)]
UTC_OFFSET  = -4
RD          = 287.05
G           = 9.80665
GKGDAY_TO_KGKGS = 1.0 / (1000.0 * 86400.0)


def density_profile(alt_km: np.ndarray, temp_k: np.ndarray,
                    p_surf_pa: float = 101_325.0):
    """Hydrostatic density on DDH altitude grid; returns (order, dz_m, rho)."""
    order = np.argsort(alt_km)
    z_s = alt_km[order] * 1000.0
    T_s = temp_k[order]
    dz  = np.empty_like(z_s); dz[0] = z_s[0]; dz[1:] = np.diff(z_s)
    ln_p = np.log(p_surf_pa) - np.cumsum(G / (RD * T_s) * dz)
    rho  = np.exp(ln_p) / (RD * T_s)
    return order, dz, rho


def column_integrate(rate_gkgday: np.ndarray, order: np.ndarray,
                     rho: np.ndarray, dz_m: np.ndarray) -> float:
    r = rate_gkgday[order] * GKGDAY_TO_KGKGS
    return float(np.sum(rho * r * dz_m))


def get_block(path: Path, name: str, n_lev: int = 87) -> np.ndarray:
    if not path.exists():
        return np.zeros(n_lev)
    with np.load(path, allow_pickle=True) as d:
        return d[name] if name in d.files else np.zeros(n_lev)


def main():
    # Density per experiment
    dens = {}
    for e in EXPERIMENTS:
        t = np.load(DDH_AGG / f"temperature_{e}.npz", allow_pickle=True)
        order, dz_m, rho = density_profile(t["altitude_km"], t["temperature_k"])
        dens[e] = {"order": order, "dz_m": dz_m, "rho": rho}

    # Per lead x experiment: compute SI, SO, P_s column-integrated from DDH.
    SI    = {e: np.full(24, np.nan) for e in EXPERIMENTS}
    SO    = {e: np.full(24, np.nan) for e in EXPERIMENTS}
    Ps    = {e: np.full(24, np.nan) for e in EXPERIMENTS}
    for lead in LEADS:
        local_h = (int(lead) + UTC_OFFSET) % 24
        for e in EXPERIMENTS:
            qv_path = DDH_AGG / f"lead{lead}_VZ" / f"{e}_QV.npz"
            if not qv_path.exists():
                continue
            qv = np.load(qv_path, allow_pickle=True)
            # SI_qv from the QV vapor-loss blocks (negative values = vapor sink)
            sink = -(qv["block__condcv"] + qv["block__condrs"])
            sink = np.where(sink > 0, sink, 0.0)
            SI[e][local_h] = column_integrate(sink, **dens[e])
            # SO_qv from the QV vapor-gain blocks
            src  =  (qv["block__evapcv"] + qv["block__evaprs"])
            src  = np.where(src > 0, src, 0.0)
            SO[e][local_h] = column_integrate(src, **dens[e])
            # P_s = − ∫ ρ·(prec-cv + prec-rs) dz, summed over QR, QS, QG.
            p_surf = 0.0
            for sp in ("QR", "QS", "QG"):
                sp_path = DDH_AGG / f"lead{lead}_VZ" / f"{e}_{sp}.npz"
                if not sp_path.exists():
                    continue
                s = np.load(sp_path, allow_pickle=True)
                rate = get_block(sp_path, "block__prec-cv") + get_block(sp_path, "block__prec-rs")
                p_surf += -column_integrate(rate, **dens[e])   # sedimentation out
            Ps[e][local_h] = p_surf

    # CMPE2: use the identity Q_CM = P_s − SI + SO.
    cmpe2 = {}
    for e in EXPERIMENTS:
        Q_CM   = Ps[e] - SI[e] + SO[e]
        pos_QCM = np.where(Q_CM > 0, Q_CM, 0.0)
        denom  = SI[e] + pos_QCM
        with np.errstate(divide="ignore", invalid="ignore"):
            cmpe2[e] = np.where(denom > 0, Ps[e] / denom, np.nan)
        print(f"  {e:8s}  Ps max={np.nanmax(Ps[e])*3600:.3f} mm/h  "
              f"SI max={np.nanmax(SI[e])*3600:.3f} mm/h  "
              f"CMPE2 range {np.nanmin(cmpe2[e])*100:.1f}..{np.nanmax(cmpe2[e])*100:.1f}%")

    # Save
    save = {"local_hour": np.arange(24)}
    for e in EXPERIMENTS:
        save[f"{e}_cmpe2"] = cmpe2[e]
        save[f"{e}_SI"]    = SI[e]
        save[f"{e}_SO"]    = SO[e]
        save[f"{e}_Ps"]    = Ps[e]
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, **save)
    print(f"saved {OUT_NPZ}")

    # Plot (single panel)
    FS = 15
    plt.rcParams.update({"font.size": FS, "axes.titlesize": FS,
                         "axes.labelsize": FS, "xtick.labelsize": FS,
                         "ytick.labelsize": FS, "legend.fontsize": FS})
    COLOR = {"control": "#d62728", "graupel": "#1f77b4", "2mom": "#2ca02c"}
    LABEL = {"control": "C1M", "graupel": "G1M", "2mom": "G2M"}
    hours = np.arange(24)

    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    for e in EXPERIMENTS:
        ax.plot(hours, cmpe2[e] * 100.0, color=COLOR[e], lw=2.4,
                marker="o", ms=5, label=LABEL[e])
    ax.axhline(100, color="k", lw=0.8, ls=":", alpha=0.6)
    ax.set_xticks(np.arange(0, 24, 3)); ax.set_xlim(-0.3, 23.3)
    ax.set_xlabel("Hour (Amazon UTC-4)")
    ax.set_ylabel("CMPE2 (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_FIG}")

    # Diagnostic: CMPE2, P_s, SI side by side
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    panels = [
        ("(a)", "CMPE2 (DDH-only)",
         lambda e: cmpe2[e] * 100.0, "CMPE2 (%)"),
        ("(b)", "Surface precip flux",
         lambda e: Ps[e] * 3600.0,   r"$P_s$ (mm h$^{-1}$)"),
        ("(c)", "Column condensation + deposition",
         lambda e: SI[e] * 3600.0,   r"$[S_{I,qv}]$ (mm h$^{-1}$)"),
    ]
    for ax, (tag, title, f, ylab) in zip(axes, panels):
        for e in EXPERIMENTS:
            ax.plot(hours, f(e), color=COLOR[e], lw=2.4, marker="o", ms=5, label=LABEL[e])
        ax.set_xticks(np.arange(0, 24, 3)); ax.set_xlim(-0.3, 23.3)
        ax.set_xlabel("Hour (Amazon UTC-4)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        if tag == "(a)":
            ax.set_ylim(0, 105); ax.axhline(100, color="k", lw=0.8, ls=":", alpha=0.6)
        ax.text(0.025, 0.975, tag, transform=ax.transAxes, ha="left", va="top",
                bbox={"facecolor":"white","edgecolor":"none","alpha":0.9,"pad":4.0})
    fig.savefig(OUT_DIAG, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIAG}")


if __name__ == "__main__":
    main()
