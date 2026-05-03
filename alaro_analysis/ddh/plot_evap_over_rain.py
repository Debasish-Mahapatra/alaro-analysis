"""Vertical profile of rain evaporation rate divided by surface rain rate.

For each experiment the column DDH aggregates expose the QR tendency split
into the budget terms ``auto-{cv,rs}``, ``evap-{cv,rs}``, ``prec-{cv,rs}``
on a common altitude axis.  Sign convention: ``evap-*`` and ``prec-*`` are
negative on QR (sinks), ``auto-*`` is positive (source).

Definitions used here:

    E(z)       = -(evap-cv + evap-rs)   rain mass evaporated per unit time
    R_surface  = -∫(prec-cv + prec-rs) dz   surface precipitation rate
    E/R(z)     = E(z) / R_surface       fraction of surface rain mass
                                        evaporated per unit altitude (1/km)

Integrating E/R(z) over the full column gives the dimensionless fraction
``E_column / R_surface``, the share of rain mass produced in the column
that ends up re-evaporated rather than reaching the ground.

Outputs ``figures/DDH-figures/evap_over_rain_profile_lead<HHHH>.png``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .io import (
    EXP_COLORS,
    EXPERIMENTS,
    FIG_DIR,
    Z_MAX_KM,
    draw_freeze_lines,
    freezing_level_km,
    load_budget,
    load_temperature,
    set_altitude_axis,
    tick_formatter,
)
from .plot_budgets import column_integrate, evap_sublim_profile, precip_profile


FREEZING_BAND_HALF_WIDTH_KM = 0.5


def evap_over_rain_profile(
    exp_cache: dict,
    z_freezing_km: float | None = None,
    *,
    band_half_km: float = FREEZING_BAND_HALF_WIDTH_KM,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    float | None,
    float | None,
    float | None,
]:
    """Return (altitude, E/R_surface, mask_keep, R_surface, evap_frac_kept, evap_frac_full).

    ``E(z)`` and ``R_surface`` are taken with positive sign (sinks flipped).
    A band of ``+/- band_half_km`` around the 0 deg C isotherm is masked
    because the QR ``evap-rs/cv`` blocks pick up snow-to-rain transition
    artefacts there (the ALARO QR fbl does not separate melting; the band
    contains numerical noise rather than real rain evaporation).
    """
    qr = exp_cache.get("QR")
    if qr is None:
        return None, None, None, None, None, None
    z = np.asarray(qr["altitude_km"], dtype=np.float64)
    blocks = qr["blocks"]
    if not all(b in blocks for b in ("evap-cv", "evap-rs", "prec-cv", "prec-rs")):
        return None, None, None, None, None, None
    evap = -(blocks["evap-cv"] + blocks["evap-rs"])     # positive => rain mass loss
    prec = -(blocks["prec-cv"] + blocks["prec-rs"])     # positive => surface flux
    r_surface = column_integrate(prec, z)
    if not np.isfinite(r_surface) or r_surface <= 0.0:
        return z, np.full_like(evap, np.nan), np.zeros_like(evap, dtype=bool), float(r_surface), None, None

    # Clip negatives to zero: any negative value in the QR phase-change
    # tendency is a snow-to-rain transition / melting artefact bleeding
    # through, not actual rain evaporation.  Only the positive part is
    # interpretable as evaporation.
    if z_freezing_km is not None and np.isfinite(z_freezing_km):
        below_freeze = z < (z_freezing_km - band_half_km)
    else:
        below_freeze = np.ones_like(z, dtype=bool)
    keep = below_freeze & (evap > 0.0)
    evap_kept = np.where(keep, evap, 0.0)
    evap_total_kept = column_integrate(evap_kept, z)
    evap_total_full = column_integrate(np.maximum(evap, 0.0), z)
    return (
        z,
        evap / r_surface,
        keep,
        float(r_surface),
        float(evap_total_kept / r_surface),
        float(evap_total_full / r_surface),
    )


def plot_evap_over_rain(args: argparse.Namespace) -> Path:
    all_data: dict[str, dict[str, dict]] = {}
    for exp in EXPERIMENTS:
        cache: dict[str, dict] = {}
        for var in ("QR",):
            data = load_budget(exp, var, lead=args.lead)
            if data is not None:
                cache[var] = data
        if cache:
            all_data[exp] = cache
    if not all_data:
        raise RuntimeError("No QR DDH aggregates found for any experiment")

    temps = {exp: load_temperature(exp) for exp in EXPERIMENTS}

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    summary_lines: list[str] = []
    z_freezings: list[float] = []
    for exp, label in EXPERIMENTS.items():
        if exp not in all_data:
            continue
        z_freeze = freezing_level_km(temps.get(exp, {}))
        if np.isfinite(z_freeze):
            z_freezings.append(z_freeze)
        z, ratio, keep, r_surf, frac_kept, frac_full = evap_over_rain_profile(
            all_data[exp], z_freezing_km=z_freeze, band_half_km=args.band_half_km
        )
        if z is None or ratio is None:
            continue
        # Plot kept (positive sub-cloud) values solid; everything else as a
        # thin dotted reference so the melting contamination remains visible.
        plot_ratio = np.where(keep, ratio, np.nan)
        ax.plot(plot_ratio, z, color=EXP_COLORS[exp], lw=2.2, label=label)
        ax.plot(ratio, z, color=EXP_COLORS[exp], lw=0.6, alpha=0.35, ls=":")
        summary_lines.append(
            f"{label}: R_surface={r_surf:.3g}, "
            f"E_evap_only/R={frac_kept:.3f}  (z<z_0C-{args.band_half_km:g}km, evap>0)"
        )
    if z_freezings:
        z_band_top = max(z_freezings)  # mask everything above (z_0C - half)
        z_band_lo = min(z_freezings) - args.band_half_km
        ax.axhspan(z_band_lo, 20.0, color="#888888", alpha=0.18,
                   label="excluded (melt + above-freezing noise)")
    draw_freeze_lines(ax, temps)
    ax.axvline(0, color="k", lw=0.6, alpha=0.6)
    ax.grid(alpha=0.3)
    ax.set_xlabel(r"E(z) / R$_{\mathrm{surface}}$ (km$^{-1}$)")
    ax.xaxis.set_major_formatter(tick_formatter())
    set_altitude_axis(ax)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Rain evaporation rate normalised by surface rain rate")

    out = args.figure_dir / f"evap_over_rain_profile_lead{args.lead}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print("\n".join(summary_lines), flush=True)
    print(f"  fig: {out}", flush=True)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Vertical profile of rain evaporation rate E(z) divided by surface "
            "rain rate R_surface for each ALARO experiment."
        )
    )
    p.add_argument("--lead", default="0024")
    p.add_argument(
        "--band-half-km",
        type=float,
        default=FREEZING_BAND_HALF_WIDTH_KM,
        help="Half-width (km) of the band around the 0 deg C isotherm to mask. "
             "QR evap-rs/cv blocks pick up snow-to-rain transition artefacts here.",
    )
    p.add_argument("--figure-dir", type=Path, default=FIG_DIR)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    plot_evap_over_rain(args)


if __name__ == "__main__":
    main()
