"""
Microphysical budget diagnostics: where does condensate end up?

Motivated by Kwinten's question: "How much evaporation/sublimation/melting do
you have relative to condensation — given an amount of condensation, where does
the condensate end up: stored in hydrometeors aloft, as precipitation, or
re-evaporated/sublimated back to vapour?"

Per-process microphysics rates (condensation, evap, sublim, melt, depos,
autoconversion, accretion) are not written to the default FA history — they
live inside APLPAR as intermediate arrays.  The per-species fluxes are in DDH
output (/gpfs/.../ALARO-RUNS/DDH/{control,graupel,2mom}/) if full detail is
needed.

This workflow works purely from the masked-NetCDF cache.  It computes four
diagnostics:

  B1 — Species storage partition.  Fraction of total condensate mass in each
       hydrometeor (QR / QS / QG / QL / QI), by height and hour-of-day.
       Tests Kwinten's G1M-graupel-aloft hypothesis: does G1M store more
       frozen mass at high altitude than C1M/G2M?

  B1 amount — Species storage amount.  Actual condensate mass in each
       hydrometeor, by height and hour-of-day, with a shared x-axis maximum
       across simulations.

  B2 — Surface-reach fraction.  surface_flux / max(column_flux) per hour.
       If the model produces a large column-max flux but only a small fraction
       reaches the surface, a lot is being re-evaporated below cloud base.

  B3 — Per-level residence time.  tau(z) = mass_above(z) / flux(z).
       If graupel has short tau aloft in G1M → less time for sublimation →
       supports Kwinten's residence-time hypothesis.

  B4 — Flux divergence profile.  -d(flux_total)/dz along the vertical.
       Positive = net microphysical source of precipitation at that level
       (condensation + collection dominate).
       Negative = net sink (evaporation/sublimation/melting dominate).
       This is a proxy for the per-process rates; in cloud-free layers
       (above cloud top / below cloud base), the sign is diagnostic
       of sublimation / evaporation alone.

Companion note — 3MT convective triggering
-------------------------------------------
Kwinten also asked: does 3MT use column moisture or only low-level moisture?
Inspection of the C46 source (arpifs/phys_dmn/accvud.F90, lines 929-938 and
1230-1296) shows that the trigger variable ZS4 accumulates L*dp*CVGQ from
surface (KLEV) to top (KTDIA).  The per-level activation test is:

    ZKUO2 = ZS4 + ZICVG + ZICVGL > 0      ! accvud.F90:935-938

So triggering does use column-integrated moisture convergence, not just
surface.  Mid/upper-level drying reduces ZS4 and can suppress convection at
low levels, supporting Kwinten's hypothesis.  Namelist controls: LCVGQM
(modulated CVGQ closure), LCAPE (CAPE-based alternative), RMULACVG (weight).

Usage:
    source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
    conda activate epygram
    python -m alaro_analysis.workflows.microphysics_budget \
        --max-days 30          # optional, limit for quick tests
        --force                # re-run even if cache exists
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import alaro_analysis.common.microphysics_budget_settings as science_settings
from alaro_analysis.analysis.microphysics_budget import (
    _finalize,
    accumulate,
    load_budget,
    save_budget,
)
from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.microphysics_budget_settings import (
    CACHE_DIR,
    DATA_ROOT,
    EXPERIMENTS,
    OUTPUT_DIR,
)
from alaro_analysis.data.cache_context import CacheContext
from alaro_analysis.data.microphysics_exports import (
    write_b1_species_amount_txt,
    write_b1_species_fraction_txt,
    write_b1b_mass_above_species_txt,
    write_b2_surface_reach_txt,
    write_b3_residence_time_txt,
    write_b4_flux_divergence_txt,
)
from alaro_analysis.plotting.microphysics_budget import (
    plot_b1_species_amount,
    plot_b1_species_fraction,
    plot_b1b_mass_above_species,
    plot_b2_surface_reach,
    plot_b3_residence_time,
    plot_b4_flux_divergence,
)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--max-days", type=int, default=None, help="Limit days per experiment (for quick tests)."
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute even if per-experiment cache exists."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS),
        choices=list(EXPERIMENTS),
        help="Subset of experiments to process.",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = parse_configured_args(parser, "microphysics_budget", argv=argv)
    contexts = {}
    for exp in args.experiments:
        settings = {
            **vars(args),
            **{
                name: getattr(science_settings, name)
                for name in [
                    "SPECIES",
                    "G",
                    "MIN_LEAD_HOUR",
                    "T_FREEZE",
                    "HOUR_BAND_DIURNAL",
                    "HOUR_BAND_DAYTIME",
                    "HOUR_BAND_DEFAULT",
                    "FLUX_FLOOR",
                ]
            },
        }
        contexts[exp] = CacheContext.from_netcdf(
            "microphysics_budget", settings, [args.data_root / exp / "masked-netcdf"]
        )

    finals: dict[str, dict[str, np.ndarray]] = {}
    for exp in args.experiments:
        cached = (
            None
            if args.force
            else load_budget(exp, cache_dir=args.cache_dir, context=contexts[exp])
        )
        if cached is None:
            cached = accumulate(
                exp, max_days=args.max_days, workers=args.workers, data_root=args.data_root
            )
            save_budget(exp, cached, cache_dir=args.cache_dir, context=contexts[exp])
        else:
            print(f"  budget {exp}: loaded cache ({cached['n_days']} days)")
        finals[exp] = _finalize(cached["sums"], cached["counts"])

    # Need all three experiments present for the comparison plots; fall back
    # to whichever subset is loaded if user asked for fewer.
    if set(finals) != set(EXPERIMENTS):
        print(f"  note: plotting with subset {list(finals)} only", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    b1_fraction = args.output_dir / "B1_species_fraction.png"
    b1_amount = args.output_dir / "B1_species_amount.png"
    b1b_mass_above = args.output_dir / "B1b_mass_above_species.png"
    b2_surface = args.output_dir / "B2_surface_reach_fraction.png"
    b3_residence = args.output_dir / "B3_residence_time.png"
    b4_divergence = args.output_dir / "B4_flux_divergence.png"

    plot_b1_species_fraction(finals, b1_fraction)
    write_b1_species_fraction_txt(finals, b1_fraction)
    plot_b1_species_amount(finals, b1_amount)
    write_b1_species_amount_txt(finals, b1_amount)
    plot_b1b_mass_above_species(finals, b1b_mass_above)
    write_b1b_mass_above_species_txt(finals, b1b_mass_above)
    plot_b2_surface_reach(finals, b2_surface)
    write_b2_surface_reach_txt(finals, b2_surface)
    plot_b3_residence_time(finals, b3_residence)
    write_b3_residence_time_txt(finals, b3_residence)
    plot_b4_flux_divergence(finals, b4_divergence)
    write_b4_flux_divergence_txt(finals, b4_divergence)


def __getattr__(name):
    """Compatibility for moved functions; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    main()
