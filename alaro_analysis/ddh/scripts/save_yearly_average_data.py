#!/usr/bin/env python3
"""
Compute 2-year average (2014+2015) of DDH 24h budget components
for each variable, experiment, and component, and save the averaged
profiles as .dta files (same format as the originals: pressure  value).

Output structure:
  yearly-average-data/{experiment}/{variable}/data/{variable}.DHFDLABOF+0024.{component}.dta
"""

import os
import numpy as np

# ─── Configuration ────────────────────────────────────────────────────────────
RESULTS_DIR = "/mnt/scratch/MANAUS/DDH/alaro-24h-budgets/results"
OUTPUT_DIR = "/mnt/scratch/MANAUS/DDH/yearly-average-data"
EXPERIMENTS = ["control", "graupel", "2mom"]
EXP_LABELS = {"control": "C1M", "graupel": "G1M", "2mom": "G2M"}
VARIABLES = ["CT", "QG", "QI", "QL", "QR", "QS", "QV", "TKE", "TTE", "UU", "VV"]


def read_dta(filepath):
    """Read a .dta file: 2 columns (pressure, value), 87 levels."""
    try:
        data = np.loadtxt(filepath)
        return data[:, 0], data[:, 1]
    except Exception:
        return None, None


def get_components(experiment, variable):
    """Discover budget components for a given experiment/variable."""
    sample_dir = os.path.join(RESULTS_DIR, experiment, "DDH20140101", variable, "data")
    if not os.path.isdir(sample_dir):
        return []
    components = []
    prefix = f"{variable}.DHFDLABOF+0024."
    for f in sorted(os.listdir(sample_dir)):
        if f.startswith(prefix) and f.endswith(".dta"):
            comp = f[len(prefix):-4]
            components.append(comp)
    return components


def get_all_days(experiment):
    """Get all DDH day directories across both years."""
    exp_dir = os.path.join(RESULTS_DIR, experiment)
    return sorted(d for d in os.listdir(exp_dir) if d.startswith("DDH"))


def compute_and_save_avg(experiment, variable, component, output_base):
    """Compute the 2-year average profile and save as .dta file."""
    days = get_all_days(experiment)
    prefix = f"{variable}.DHFDLABOF+0024."
    profiles = []
    pressure = None

    for day in days:
        filepath = os.path.join(
            RESULTS_DIR, experiment, day, variable, "data",
            f"{prefix}{component}.dta"
        )
        if not os.path.isfile(filepath):
            continue
        p, v = read_dta(filepath)
        if p is not None:
            if pressure is None:
                pressure = p
            profiles.append(v)

    if len(profiles) == 0:
        return 0

    avg = np.mean(profiles, axis=0)

    # Save in same format as original .dta files
    out_dir = os.path.join(output_base, experiment, variable, "data")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{prefix}{component}.dta")

    with open(out_file, 'w') as f:
        for p_val, v_val in zip(pressure, avg):
            f.write(f"  {p_val:25.16E}  {v_val:25.16E}\n")

    return len(profiles)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("DDH 24h Budget — Save 2-Year Average Data")
    print("=" * 60)
    print(f"Source:      {RESULTS_DIR}")
    print(f"Output:      {OUTPUT_DIR}")
    print(f"Experiments: {list(EXP_LABELS.values())}")
    print(f"Variables:   {VARIABLES}")
    print(f"Averaging over: 2014 + 2015 (730 days)")
    print()

    for exp in EXPERIMENTS:
        n = len(get_all_days(exp))
        print(f"  {EXP_LABELS[exp]} ({exp}): {n} total days")
    print()

    total_files = 0
    for variable in VARIABLES:
        print(f"Processing {variable}...")
        for experiment in EXPERIMENTS:
            components = get_components(experiment, variable)
            for comp in components:
                n_days = compute_and_save_avg(experiment, variable, comp, OUTPUT_DIR)
                if n_days > 0:
                    total_files += 1
                    print(f"  {EXP_LABELS[experiment]}/{variable}/{comp}: averaged {n_days} days")
                else:
                    print(f"  {EXP_LABELS[experiment]}/{variable}/{comp}: NO DATA")

    print(f"\nDone! Saved {total_files} averaged .dta files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
