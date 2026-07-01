"""Box model: the freezing/melting FUSION-HEAT feedback into the convective updraught (the lever
NIMELIT=2 activates), as a parcel/CAPE estimate built from the ACCVUD code relations. Multi-model
verified (constants, dTdq, CAPE integral, sign/accumulation all confirmed). Parcel estimate of a
coupled scheme -- magnitude + mechanism, not a full column solve.

From accvud.F90 / aplmini.F90 (verified extraction):
  heat source  : ZLHS-ZLHV = L_fusion = RLMLT = 3.337e5 J/kg
  temperature  : dT = Lf*dq/cp   (ZLSCP=(PLHS-PLHV)/PCP, aplmini.F90:307)  -> 0.332 K per g/kg
  buoyancy/CAPE: integrand g*(T_parcel-T_env)/T_env  (accvud.F90:1136 uses the equivalent
                 Rd*(Tv_up-Tv_env)*d(ln p); this box uses dry dT + geometric dz, equal to leading order)
  conservation : PMELNET column-normalized -> column-integrated fusion heat conserved (accvud:1667-79).

Mechanism: freezing of condensate in the mixed-phase updraught releases Lf and WARMS the parcel; the
excess accumulates and is retained aloft -> a CAPE gain. The compensating melting COOLS the falling
precip/downdraught; the column normalization exports that cooling downward, so only a FRACTION f_ret
stays in the buoyant core. Net boost = warming(aloft) - f_ret*cooling.

C1M vs G1M: per-unit-freezing heat is species-INDEPENDENT; the difference is inherited from the fall
speed (fallspeed_melting box): graupel falls ~2-3x faster, exporting precip/cooling LOWER -> smaller
f_ret -> larger net boost. CAVEAT: net CAPE is linear in Q by construction (f_ret, depths are Q-independent
constants), so the clean constant G1M/C1M ratio is a MODELING artifact of two hand-picked f_ret values,
NOT a robust prediction. Direction (graupel amplifies) is robust; magnitude is illustrative -- panel (c)
is the f_ret sensitivity. Omits entrainment, condensate loading, and the ACCVUD fixed-point iteration.

Output -> microphysics-paper/SUP/graupel_snow_boxmodels/.
"""
import io as _io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/SUP/graupel_snow_boxmodels")
DPI = 450

RTT, RG = 273.16, 9.80665
LF, CP  = 3.337e5, 1004.709          # RLMLT, RCPD
dTdq    = LF/CP                       # 332.1 K per (kg/kg)
Tsfc, GAMMA = 300.0, 6.5e-3
def Tenv(z): return Tsfc - GAMMA*z
z0C, zm30 = (Tsfc-RTT)/GAMMA, (Tsfc-(RTT-30))/GAMMA
DZ, ZTOP = 20.0, 12000.0
Z = np.arange(0.0, ZTOP, DZ)
FRET_SNOW, DEPTH_SNOW = 0.50, 300.0
FRET_GRAU, DEPTH_GRAU = 0.25, 700.0
CAPE0 = 1500.0

def warm_profile(q_gkg):
    q = q_gkg*1e-3; frz = (Z >= z0C) & (Z <= zm30)
    dq = np.where(frz, q/max(frz.sum(),1), 0.0)
    return np.cumsum(dq)*dTdq
def cool_profile(q_gkg, depth_m):
    q = q_gkg*1e-3; mlt = (Z >= z0C-depth_m) & (Z < z0C)
    dq = np.where(mlt, q/max(mlt.sum(),1), 0.0)
    return np.cumsum(dq)*dTdq
def dCAPE(q_gkg, f_ret, cool_depth):
    dT = warm_profile(q_gkg) - f_ret*cool_profile(q_gkg, cool_depth)
    return float(np.sum(RG*dT/Tenv(Z))*DZ)

def tables():
    b = _io.StringIO(); p = lambda s="": print(s, file=b)
    p(f"0 C = {z0C:.0f} m ; -30 C = {zm30:.0f} m ; dT/dq = {dTdq:.1f} K/(kg/kg) = {dTdq*1e-3:.3f} K per g/kg")
    p("\n"+"="*70); p("Fusion-heat feedback: net updraught CAPE gain (parcel estimate)"); p("="*70)
    for Q in (0.5, 1.0, 2.0, 3.0):
        dcs, dcg = dCAPE(Q, FRET_SNOW, DEPTH_SNOW), dCAPE(Q, FRET_GRAU, DEPTH_GRAU)
        p(f"  Q={Q:.1f} g/kg :  dCAPE(C1M/snow)={dcs:6.1f} J/kg   dCAPE(G1M/graupel)={dcg:6.1f} J/kg"
          f"   G1M/C1M={dcg/max(dcs,1e-9):4.2f}")
    w0 = np.sqrt(2*CAPE0)
    p(f"\n  Updraft w_max increment (baseline CAPE={CAPE0:.0f}, w0={w0:.1f} m/s):")
    for Q in (1.0, 2.0, 3.0):
        dcs, dcg = dCAPE(Q, FRET_SNOW, DEPTH_SNOW), dCAPE(Q, FRET_GRAU, DEPTH_GRAU)
        p(f"   Q={Q:.1f} g/kg :  dW(C1M)={np.sqrt(2*(CAPE0+dcs))-w0:4.2f} m/s   dW(G1M)={np.sqrt(2*(CAPE0+dcg))-w0:4.2f} m/s")
    p("\n  Pure warming (f_ret=0) = ice-latent-heat CAPE ceiling:")
    for Q in (1.0, 2.0, 3.0):
        p(f"   Q={Q:.1f} g/kg :  dCAPE_warm={dCAPE(Q,0.0,DEPTH_SNOW):6.1f} J/kg")
    p("\n  NOTE: net CAPE is linear in Q; the constant G1M/C1M ratio is set by the f_ret constants (modeling")
    p("  choice), not a robust prediction. Direction robust (graupel amplifies), magnitude illustrative.")
    return b.getvalue()

def make_figure():
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for Q, c in ((1.0,'0.6'), (2.0,'k')):
        ax[0].plot(warm_profile(Q)-FRET_SNOW*cool_profile(Q,DEPTH_SNOW), Z/1e3, color=c, lw=2, label=f"C1M Q={Q:g}")
    ax[0].plot(warm_profile(2.0)-FRET_GRAU*cool_profile(2.0,DEPTH_GRAU), Z/1e3, color='tab:orange', lw=2, ls='--', label="G1M Q=2")
    ax[0].axhline(z0C/1e3, color='0.5', ls=':', lw=1); ax[0].text(0.02, z0C/1e3+0.15, "0 C", fontsize=8, color='0.4')
    ax[0].axvline(0, color='0.7', lw=0.8); ax[0].set_xlabel("net parcel temperature excess [K]"); ax[0].set_ylabel("height [km]")
    ax[0].set_title("freezing warms aloft (retained);\nmelting cools low (partly exported)", fontsize=9)
    ax[0].legend(frameon=False, fontsize=8)

    Qs = np.linspace(0, 4, 60)
    ax[1].plot(Qs, [dCAPE(q,0.0,DEPTH_SNOW) for q in Qs],       color='0.6', lw=1.6, ls=':', label='warming ceiling (f_ret=0)')
    ax[1].plot(Qs, [dCAPE(q,FRET_GRAU,DEPTH_GRAU) for q in Qs], color='tab:orange', lw=2, label='G1M (graupel)')
    ax[1].plot(Qs, [dCAPE(q,FRET_SNOW,DEPTH_SNOW) for q in Qs], color='tab:blue', lw=2, label='C1M (snow)')
    ax[1].set_xlabel("column condensate frozen  Q  [g/kg]"); ax[1].set_ylabel("net updraught CAPE gain [J/kg]")
    ax[1].legend(frameon=False, fontsize=8.5)

    fr = np.linspace(0, 1, 80)
    ax[2].plot(fr, [dCAPE(2.0, f, 500.0) for f in fr], color='tab:green', lw=2)
    ax[2].axvline(FRET_GRAU, color='tab:orange', ls='--', lw=1.2); ax[2].text(FRET_GRAU+0.02, ax[2].get_ylim()[0]+3, "G1M", color='tab:orange', fontsize=8)
    ax[2].axvline(FRET_SNOW, color='tab:blue', ls='--', lw=1.2);   ax[2].text(FRET_SNOW+0.02, ax[2].get_ylim()[0]+3, "C1M", color='tab:blue', fontsize=8)
    ax[2].set_xlabel("fraction of melting-cooling retained in updraught  f_ret"); ax[2].set_ylabel("net CAPE gain [J/kg]  (Q=2 g/kg)")
    ax[2].set_title("faster graupel exports cooling -> lower f_ret\n-> larger net boost", fontsize=9)
    for a in ax: a.tick_params(labelsize=9)
    fig.tight_layout()
    return fig

def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT/"data_txt").mkdir(parents=True, exist_ok=True)
    txt = tables(); print(txt)
    (OUT/"data_txt"/"fusion_heat_feedback.txt").write_text(txt)
    fig = make_figure()
    fig.savefig(OUT/"fusion_heat_feedback.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print("saved ->", OUT/"fusion_heat_feedback.png")

if __name__ == "__main__":
    main()
