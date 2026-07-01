"""Box model of the ALARO WBF cloud-water autoconversion in ACACON, showing how the
RAUTEFR (C1M->snow, 0.8e-3) vs RAUTEFG (G1M->graupel, 1.0e-3) coefficient gap
(+25% nominal) actually propagates through the saturating semi-implicit update.

At the model timestep (TSPHY=120 s) and convective mixed-phase loadings the update is
near-saturated (converts ~80% of cloud water to ice per step), so the realized
ice-conversion difference collapses to ~+4%, not +25%; the full +25% only survives in
the dilute limit or at a much shorter substep.

Faithful to phys_dmn/acacon.F90 (verified line-by-line against the CY46 source and
cross-checked by three independent reviewers). 1-moment liquid (LTWOMOMLIQ=F),
non-simplified (LDSIMP=F), LCCNDIAG=F branch:

  ZHSEFL = RAUTEFR*TSPHY*(1 - exp(-(ql/ZQLCR)^2))                 # warm auto -> rain (RAUTEFR in BOTH configs)
  ZWBF   = RAUTEF *TSPHY*RWBF1*(1 - exp(-(ql*qi)/(ZQLCR*ZQICR*RWBF2^2)))*a*(1-a)  # a = qi/(ql+qi)
  PACO_ice = ZWBF *ql/(1 + ZHSEFL + ZWBF)      # -> snow (C1M, RAUTEFR) or graupel (G1M, RAUTEFG)

Constants are the verified Manaus run values (namelist overrides + source defaults,
confirmed on the tier-1 compile source and name.e001.CY46T2cont.sfx).

Outputs -> microphysics-paper/SUP/wbf_autoconversion_boxmodel/.
"""
import io as _io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/SUP/wbf_autoconversion_boxmodel")
DPI = 450

# ---- constants (namelist overrides + source defaults, verified on tier-1) ----
ZPOISS = 2.0/np.sqrt(np.pi)
RTT    = 273.16
RAUTEFR = 0.8e-3     # namelist  name.e001.CY46T2cont.sfx
RAUTEFG = 1.0e-3     # DEFAULT (absent from namelist) suphy0.F90:1132
RWBF1   = 300.0      # namelist
RWBF2   = 4.0        # default (not in namelist)
RQLCR   = 4.0e-4     # namelist
RQICRMIN= 8.0e-7     # namelist
RQICRMAX= 5.0e-5     # namelist
RCPCOEF1, RCPCOEF2, RCPCOEF3 = 2.0e-5, 0.06, 3.5   # defaults
ZALPHA, ZBETA = -0.1572, -4.9632                   # aplmini.F90 PEXPA
TSPHY   = 120.0      # s  (TSTEP=120)


def zqicr(T, lcp06):
    if lcp06:
        return min(RCPCOEF1, 10.0**(RCPCOEF2*(T-RTT)-RCPCOEF3))*ZPOISS
    pexpa = (1.0 - np.tanh(ZALPHA*(T-RTT)+ZBETA))*0.5
    return (RQICRMIN + (RQICRMAX-RQICRMIN)*pexpa)*ZPOISS


def wbf(ql, qi, T, graupel, lcp06=False, tsphy=TSPHY):
    """ql, qi in kg/kg. Returns dict with the ice-conversion increment (kg/kg/step)."""
    ZQLCR = RQLCR*ZPOISS
    ZQICR = zqicr(T, lcp06)
    a = qi/np.maximum(1e-10, ql+qi)
    ZHSEFL = RAUTEFR*tsphy*(1.0 - np.exp(-(ql/ZQLCR)**2))               # always RAUTEFR
    coef = RAUTEFG if graupel else RAUTEFR
    fsat = 1.0 - np.exp(-(ql*qi)/(ZQLCR*ZQICR*RWBF2**2))
    ZWBF = coef*tsphy*RWBF1*fsat*a*(1.0-a)
    denom = 1.0 + ZHSEFL + ZWBF
    PACO_ice = ZWBF*ql/denom       # -> snow(C1M) or graupel(G1M)
    PACORL   = ZHSEFL*ql/denom     # -> rain
    return dict(ice=PACO_ice, rain=PACORL, ZWBF=ZWBF, ZHSEFL=ZHSEFL,
                frac_ice=PACO_ice/ql, frac_tot=(PACO_ice+PACORL)/ql)


def compare(ql, qi, T, lcp06=False, tsphy=TSPHY):
    c = wbf(ql, qi, T, graupel=False, lcp06=lcp06, tsphy=tsphy)
    g = wbf(ql, qi, T, graupel=True,  lcp06=lcp06, tsphy=tsphy)
    return c, g, g['ice']/c['ice']


gk = 1e-3  # g/kg -> kg/kg
Trep = RTT - 12.0


def make_tables():
    """Return the numeric results as a text block (also echoed to stdout)."""
    b = _io.StringIO()
    p = lambda s="": print(s, file=b)
    p("="*74)
    p("NOMINAL coefficient ratio RAUTEFG/RAUTEFR = %.3f  (+%.0f%%)"
      % (RAUTEFG/RAUTEFR, 100*(RAUTEFG/RAUTEFR-1)))
    p("TSPHY=%.0fs  RWBF1=%.0f  RWBF2=%.0f  RQLCR=%.1e  T=%.0f C" %
      (TSPHY, RWBF1, RWBF2, RQLCR, Trep-RTT))
    p("="*74)
    p("\n--- Representative convective mixed-phase point: ql=1.0, qi=0.5 g/kg, T=-12C ---")
    for lcp06 in (False, True):
        c, g, r = compare(1.0*gk, 0.5*gk, Trep, lcp06=lcp06)
        p(" LCP06=%-5s  ZQICR=%.2e  ZWBF: C1M=%.2f G1M=%.2f  ZHSEFL=%.3f" %
          (lcp06, zqicr(Trep, lcp06), c['ZWBF'], g['ZWBF'], c['ZHSEFL']))
        p("            cloud-water->ice fraction/step:  C1M=%.1f%%  G1M=%.1f%%" %
          (100*c['frac_ice'], 100*g['frac_ice']))
        p("            ACTUAL ice-conversion ratio G1M/C1M = %.3f  (+%.1f%%  vs nominal +25%%)"
          % (r, 100*(r-1)))
    p("\n--- Regime limits (LCP06=F, T=-12C) ---")
    for lab, ql, qi in [("dilute      ql=0.02 qi=0.02", 0.02, 0.02),
                        ("light       ql=0.1  qi=0.1 ", 0.10, 0.10),
                        ("moderate    ql=0.5  qi=0.3 ", 0.50, 0.30),
                        ("convective  ql=1.0  qi=0.5 ", 1.00, 0.50),
                        ("core        ql=3.0  qi=1.0 ", 3.00, 1.00)]:
        c, g, r = compare(ql*gk, qi*gk, Trep)
        p("  %-28s  G1M/C1M=%.3f (+%4.1f%%)   convert/step C1M=%4.1f%% G1M=%4.1f%%"
          % (lab, r, 100*(r-1), 100*c['frac_ice'], 100*g['frac_ice']))
    p("\n--- Sensitivity to the physics substep (ql=1.0 qi=0.5 g/kg, T=-12C, LCP06=F) ---")
    for ts in (120, 60, 30, 10, 3, 1):
        c, g, r = compare(1.0*gk, 0.5*gk, Trep, tsphy=ts)
        p("  TSPHY=%4ds   G1M/C1M=%.3f (+%4.1f%%)   ZWBF(C1M)=%6.2f  convert/step C1M=%4.1f%%"
          % (ts, r, 100*(r-1), c['ZWBF'], 100*c['frac_ice']))
    return b.getvalue()


# ---------- plotting ----------
QL_AX = np.logspace(np.log10(0.01), np.log10(5.0), 240)*gk
QIS   = [0.1, 0.3, 0.5, 1.0]


def panel_excess(ax):
    for qi in QIS:
        r = np.array([compare(q, qi*gk, Trep)[2] for q in QL_AX])
        ax.plot(QL_AX/gk, 100*(r-1), lw=2, label="q$_i$=%g g/kg" % qi)
    ax.axhline(25, ls="--", c="0.35", lw=1.3)
    ax.text(0.011, 25.6, "nominal coefficient gap (+25%)", fontsize=8.5, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("cloud liquid water q$_l$ [g/kg]")
    ax.set_ylabel("actual WBF ice-conversion excess  G1M vs C1M  [%]")
    ax.set_ylim(0, 27)
    ax.legend(frameon=False, fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))


def panel_fraction(ax, qi=0.5):
    fc = np.array([wbf(q, qi*gk, Trep, graupel=False)['frac_ice'] for q in QL_AX])
    fg = np.array([wbf(q, qi*gk, Trep, graupel=True )['frac_ice'] for q in QL_AX])
    ax.plot(QL_AX/gk, 100*fc, lw=2, label="C1M  (auto $\\to$ snow, RAUTEFR)")
    ax.plot(QL_AX/gk, 100*fg, lw=2, label="G1M  (auto $\\to$ graupel, RAUTEFG)")
    ax.set_xscale("log")
    ax.set_xlabel("cloud liquid water q$_l$ [g/kg]")
    ax.set_ylabel("cloud water converted to ice per step  [%%]  (q$_i$=%g g/kg)" % qi)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data_txt").mkdir(parents=True, exist_ok=True)

    tables = make_tables()
    print(tables)
    (OUT / "data_txt" / "boxmodel_results.txt").write_text(tables)

    # combined 2-panel figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    panel_excess(ax[0])
    panel_fraction(ax[1])
    fig.tight_layout()
    fig.savefig(OUT / "wbf_autoconversion_boxmodel.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # the two panels as standalone figures
    fig, ax = plt.subplots(figsize=(5.8, 4.3)); panel_excess(ax); fig.tight_layout()
    fig.savefig(OUT / "wbf_conversion_excess.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.3)); panel_fraction(ax); fig.tight_layout()
    fig.savefig(OUT / "wbf_converted_fraction.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)

    print("saved figures + data_txt ->", OUT)


if __name__ == "__main__":
    main()
