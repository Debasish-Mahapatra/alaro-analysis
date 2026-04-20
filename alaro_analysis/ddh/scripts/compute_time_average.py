#!/usr/bin/env python3
"""
Compute the total time average (mean over all 730 days) of the DDH 24-hour
budgets for each experiment, variable, and budget term.

For every .dta file found across the daily directories, this script:
  1. Reads all daily realisations (one per DDH day directory).
  2. Computes the arithmetic mean of column-2 values across all days.
  3. Writes the result (same 2-column format: pressure, mean-value) into
     an output directory that mirrors the original structure but without
     the DDH<date> level.

Output layout
─────────────
  <OUT_ROOT>/<experiment>/<variable>/<term>.dta

where <term> is the budget-term part of the original filename, e.g.
  QV.DHFDLABOF+0024.dynam.dta  →  <OUT_ROOT>/control/QV/dynam.dta
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


def discover_terms(results_root: Path, experiment: str) -> dict:
    """
    Walk one experiment and build a mapping:
        (variable, term_name)  →  [list of Path to .dta files, one per day]

    The term_name is the part of the filename between the input-basename token
    and the .dta suffix, e.g. for
        QV.DHFDLABOF+0024.dynam.dta
    the term_name is "dynam".
    """
    exp_dir = results_root / experiment
    if not exp_dir.is_dir():
        print(f"  [WARNING] experiment directory not found: {exp_dir}", file=sys.stderr)
        return {}

    terms: dict[tuple[str, str], list[Path]] = defaultdict(list)

    # Iterate over day directories (DDH20140101, DDH20140102, …)
    day_dirs = sorted(
        d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("DDH")
    )

    for day_dir in day_dirs:
        # Iterate over variable directories (CT, QV, TTE, …)
        for var_dir in sorted(d for d in day_dir.iterdir() if d.is_dir()):
            data_dir = var_dir / "data"
            if not data_dir.is_dir():
                continue
            var_name = var_dir.name
            for dta_file in sorted(data_dir.glob("*.dta")):
                # Parse term name from filename
                # e.g. "QV.DHFDLABOF+0024.dynam.dta" → "dynam"
                parts = dta_file.stem.split(".")  # drops .dta
                # parts = ["QV", "DHFDLABOF+0024", "dynam"]
                if len(parts) >= 3:
                    term_name = ".".join(parts[2:])  # handles multi-dot terms
                else:
                    term_name = dta_file.stem
                terms[(var_name, term_name)].append(dta_file)

    return terms


def read_dta(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column .dta file → (pressure_levels, values)."""
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def compute_and_write_average(
    dta_files: list[Path],
    out_path: Path,
) -> int:
    """
    Read all daily .dta files, compute the element-wise mean of column 2,
    and write the result using the pressure levels from the first file.

    Returns the number of files successfully averaged.
    """
    values_stack = []
    pressure = None

    for f in dta_files:
        try:
            p, v = read_dta(f)
        except Exception as exc:
            print(f"  [WARNING] skipping {f}: {exc}", file=sys.stderr)
            continue

        if pressure is None:
            pressure = p
        values_stack.append(v)

    if not values_stack:
        return 0

    mean_values = np.mean(values_stack, axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write in the same whitespace-aligned format as the originals
    with open(out_path, "w") as fh:
        for p_val, m_val in zip(pressure, mean_values):
            fh.write(f"   {p_val:20.13f} {m_val:24.16E}\n")

    return len(values_stack)


def main():
    parser = argparse.ArgumentParser(
        description="Compute the full 2-year time average of DDH 24h budgets."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/Users/dev/ddhtoolbox/data/alaro-24h-budgets/results"),
        help="Root of the per-experiment results tree.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Users/dev/ddhtoolbox/data/alaro-24h-budgets/time_average"),
        help="Where to write the averaged .dta files.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["control", "graupel", "2mom"],
        help="Experiments to process.",
    )
    args = parser.parse_args()

    results_root = args.results_root
    output_root = args.output_root

    print(f"Results root : {results_root}")
    print(f"Output root  : {output_root}")
    print(f"Experiments  : {args.experiments}")
    print()

    for exp in args.experiments:
        print(f"═══ Processing experiment: {exp} ═══")
        terms = discover_terms(results_root, exp)

        if not terms:
            print(f"  No data found for experiment '{exp}'.\n")
            continue

        n_vars = len({k[0] for k in terms})
        n_terms = len(terms)
        n_days_example = len(next(iter(terms.values())))
        print(f"  Found {n_vars} variables, {n_terms} budget terms, ~{n_days_example} days each.")

        for (var_name, term_name), dta_files in sorted(terms.items()):
            out_path = output_root / exp / var_name / f"{term_name}.dta"
            n_averaged = compute_and_write_average(dta_files, out_path)
            print(f"  {var_name:>4s}/{term_name:<20s}  →  averaged {n_averaged:>4d} days  →  {out_path.name}")

        print()

    print("Done.")


if __name__ == "__main__":
    main()
