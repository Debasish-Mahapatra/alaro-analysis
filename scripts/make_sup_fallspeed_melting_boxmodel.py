"""Box model: SNOW vs GRAUPEL terminal fall speed and melting, from the ALARO CY46 code.
Multi-model verified (fall speed & content EXACT vs aplmphys.F90; melt kernel species-independence
vs acevmel.F90). Supports the microphysics-paper SUP "graupel vs snow" mechanism discussion.

FALL SPEED -- EXACT (aplmphys.F90, single-moment, LFSVAR=.T., LFVGICE3=.T.):
  ZEXPN     = min(1, exp(0.0231*(T-RTT)))
  V_snow    = FSPRAIN*(F/rho^4)^(1/6) * ZEXPN/ZEVGSL0            [FSPRAIN=13.4, ZEVGSL0=3.959]
  V_graupel = ZMULFALG*FSPRAIN*(F/rho^ZEXPRHOG)^(1/7)           [ZMULFALG=0.46, ZEXPRHOG=7/3]
  content   q = F/(rho*V)   (from F = rho*q*V)                  [F kg/m2/s, rho kg/m3, V m/s]

MELTING -- ACEVMEL (acevmel.F90:578-600, the LA0MPS.AND.LEVAPLOP / LGRAPRO branch; guard opens L427,
closes L709) melts snow and graupel with ONE species-INDEPENDENT kernel; they differ only by the
precip-flux partition ZRMN/ZRMG:
  ZMELT = ((T-RTT)/PLSCP) * ZARG/(1+ZARG)         [semi-implicit; (T-RTT)/PLSCP is the ZARG->inf asymptote]
  FONT=2.4e4, ZMULMF=0.4925, ZGAMMA=0.5479 (LAB12); rain-freezing (ZMFREE) routes entirely to graupel.
There is NO distinct graupel-melt law: the melting difference is entirely inherited from the fall speed
(content) and the flux share. Everything plotted is EXACT (no fitted parameters).

Output -> microphysics-paper/SUP/graupel_snow_boxmodels/.
"""
import io as _io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/SUP/graupel_snow_boxmodels")
DPI = 450

# ---- constants (verified against source; suphy0.F90 defaults, latent heats yomcst/sucst) ----
RTT   = 273.16
LF    = 3.337e5           # RLMLT
CP    = 1004.709          # RCPD
PLSCP = LF/CP             # Lf/cp  [K per (kg/kg)]
FSPRAIN, ZEXFALL, ZEXFALLG = 13.4, 1.0/6.0, 1.0/7.0
ZMULFALG, ZEXPRHOG, ZEVGSL0, ZEXTMP = 0.46, 7.0/3.0, 3.959, 0.0231
FONT, ZMULMF, ZGAMMA, PRPCEFF = 2.4e4, 0.5*0.985, 0.5479, 0.9694   # melt (shared by both species)
MMH = 1000.0/3600.0/1000.0        # kg m-2 s-1 per mm/h liquid-equiv (=2.778e-4)

def zexpn(T):        return np.minimum(1.0, np.exp(ZEXTMP*(T-RTT)))
def v_snow(F, rho, T):     return FSPRAIN*(F/rho**4)**ZEXFALL * zexpn(T)/ZEVGSL0
def v_graupel(F, rho, T=None): return ZMULFALG*FSPRAIN*(F/rho**ZEXPRHOG)**ZEXFALLG

rho0 = 0.6

def tables():
    b = _io.StringIO(); p = lambda s="": print(s, file=b)
    p("="*74); p("PART 1  Terminal fall speed and content  (EXACT from aplmphys.F90)"); p("="*74)
    for T in (RTT-2, RTT-10, RTT-25):
        p(f"\n  rho={rho0} kg/m3,  T={T-RTT:+.0f} C   ZEXPN={float(zexpn(T)):.3f}")
        for Rmmh in (1.0, 5.0, 20.0):
            F = Rmmh*MMH; vs, vg = v_snow(F, rho0, T), v_graupel(F, rho0, T)
            qs, qg = F/(rho0*vs)*1e3, F/(rho0*vg)*1e3
            p(f"    R={Rmmh:5.1f} mm/h  V_snow={vs:5.2f}  V_graupel={vg:5.2f}  V ratio={vg/vs:4.2f}"
              f"   q_snow={qs:5.3f}  q_graupel={qg:5.3f} g/kg  (q ratio {qs/qg:4.2f})")
    p("\n"+"="*74); p("PART 2  Melt kernel is species-INDEPENDENT (ACEVMEL); asymptotic heat-limit bound"); p("="*74)
    for dT in (0.5, 2.0, 5.0):
        ceil = dT/PLSCP*1e3
        p(f"  T-0C = +{dT:.1f} K : asymptotic melt bound (T-RTT)/PLSCP = {ceil:6.3f} g/kg "
          f"(actual melt = bound * ZARG/(1+ZARG) < bound; same kernel for snow & graupel)")
    p("  => snow and graupel share the identical ZMELT kernel; the only species difference is the flux")
    p("     partition ZRMN=Fs/(Fs+Fg) (snow) vs ZRMG=1-ZRMN (graupel), plus rain-freezing (ZMFREE) to")
    p("     graupel only. No distinct graupel-melt law exists -> the melt difference is fall-speed-inherited.")
    return b.getvalue()


def make_figure():
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    R = np.logspace(np.log10(0.2), np.log10(50), 240); F = R*MMH
    STYLE = {RTT-2:'-', RTT-10:'--', RTT-25:':'}
    for T in (RTT-2, RTT-10, RTT-25):
        ax[0].plot(R, v_graupel(F, rho0, T), lw=2, color='tab:orange', ls=STYLE[T])
        ax[0].plot(R, v_snow(F, rho0, T),    lw=2, color='tab:blue',   ls=STYLE[T])
    ax[0].plot([], [], color='tab:orange', lw=2, label='graupel'); ax[0].plot([], [], color='tab:blue', lw=2, label='snow')
    for T in (RTT-2, RTT-10, RTT-25): ax[0].plot([], [], color='0.4', ls=STYLE[T], label=f"T = {T-RTT:+.0f} C")
    ax[0].set_xscale('log'); ax[0].set_xlabel("precipitation rate [mm/h]"); ax[0].set_ylabel("terminal fall speed [m/s]")
    ax[0].legend(frameon=False, fontsize=8); ax[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:g}"))

    Trange = np.linspace(RTT-40, RTT-0.5, 240)
    for Rmmh in (1.0, 5.0, 20.0):
        ax[1].plot(Trange-RTT, v_graupel(Rmmh*MMH, rho0, Trange)/v_snow(Rmmh*MMH, rho0, Trange), lw=2, label=f"R = {Rmmh:g} mm/h")
    ax[1].axhline(1, color='0.6', lw=0.8); ax[1].set_xlabel("temperature [C]")
    ax[1].set_ylabel("fall-speed ratio  V_graupel / V_snow"); ax[1].legend(frameon=False, fontsize=9)

    for T in (RTT-2, RTT-10, RTT-25):
        ax[2].plot(R, F/(rho0*v_snow(F, rho0, T))*1e3,    lw=2, color='tab:blue',   ls=STYLE[T])
        ax[2].plot(R, F/(rho0*v_graupel(F, rho0, T))*1e3, lw=2, color='tab:orange', ls=STYLE[T])
    ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlabel("precipitation rate [mm/h]")
    ax[2].set_ylabel("precip content  q = F/(rho V)  [g/kg]")
    ax[2].plot([], [], color='tab:blue', lw=2, label='snow'); ax[2].plot([], [], color='tab:orange', lw=2, label='graupel')
    ax[2].legend(frameon=False, fontsize=9)
    ax[2].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:g}"))
    ax[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:g}"))
    fig.tight_layout()
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT/"data_txt").mkdir(parents=True, exist_ok=True)
    txt = tables(); print(txt)
    (OUT/"data_txt"/"fallspeed_melting.txt").write_text(txt)
    fig = make_figure()
    fig.savefig(OUT/"fallspeed_melting.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print("saved ->", OUT/"fallspeed_melting.png")


if __name__ == "__main__":
    main()
