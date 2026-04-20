#!/usr/bin/env python3
"""
Extract mean temperature profiles (VCT0, VCT1) from raw DDH binary files.

Pipeline:
  1. For each experiment and day, run ddht + ddhi to extract VCT0 (initial T)
     and VCT1 (final T) in Kelvin.
  2. Compute T_mean = (VCT0 + VCT1) / 2  for each day.
  3. Save daily results to:
       <OUTPUT_DIR>/<experiment>/<DDHdate>/CT_TEMP/data/CT_TEMP.DHFDLABOF+0024.<comp>.dta
  4. Compute 2-year average and save to:
       <YEARLY_DIR>/<experiment>/CT_TEMP/data/CT_TEMP.DHFDLABOF+0024.<comp>.dta

Uses the same ddht/ddhi tools as the budget pipeline but with a minimal
extraction list targeting only VCT0 and VCT1 articles.
"""

import os
import sys
import shutil
import subprocess
import tempfile
import numpy as np

# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "untar-data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "temperature-profiles")
YEARLY_DIR = os.path.join(SCRIPT_DIR, "..", "yearly-average-data")

CONVERSION_LIST = os.path.join(
    SCRIPT_DIR, "..", "alaro-24h-budgets", "_runtime",
    "ddh_budget_lists", "alaro", "conversion_list"
)

DDHT = shutil.which("ddht")
DDHI = shutil.which("ddhi")

EXPERIMENTS = ["control", "graupel", "2mom"]
EXP_LABELS = {"control": "C1M", "graupel": "G1M", "2mom": "G2M"}
INPUT_FILE = "DHFDLABOF+0024"
VARIABLE = "CT_TEMP"   # our output "variable" name

# ddht extraction list: read VCT0 and VCT1 from the binary
LC_DDHT = """\
0
VCT0 +
ECR"VCT0"
0
VCT1 +
ECR"VCT1"
"""

# ddhi processing list: which articles to output
LC_DDHI = """\
VCT0
VCT1
"""


def get_all_days(experiment):
    """Return sorted list of DDH day directory names."""
    exp_dir = os.path.join(RAW_DATA_DIR, experiment, "output")
    if not os.path.isdir(exp_dir):
        return []
    return sorted(d for d in os.listdir(exp_dir) if d.startswith("DDH"))


def extract_temperature(input_path, work_dir):
    """Extract VCT0 and VCT1 from a single DDH binary file.

    Returns dict {'VCT0': (pressure, values), 'VCT1': (pressure, values)}
    or None on failure.
    """
    # Copy input file to work directory
    input_basename = os.path.basename(input_path)
    shutil.copy2(input_path, os.path.join(work_dir, input_basename))

    # Write ddht command list
    with open(os.path.join(work_dir, "lc.ddht"), "w") as f:
        f.write(LC_DDHT)

    # Write ddhi processing list
    with open(os.path.join(work_dir, "lc.ddhi"), "w") as f:
        f.write(LC_DDHI)

    # Copy conversion list
    shutil.copy2(CONVERSION_LIST, os.path.join(work_dir, "ddhi_list.tmp"))

    # Run ddht
    s_file = f"{input_basename}.s"
    cmd_ddht = [
        DDHT, "-cCALC",
        f"-1{input_basename}",
        f"-s{s_file}",
        "-llc.ddht"
    ]
    result = subprocess.run(cmd_ddht, cwd=work_dir,
                            capture_output=True, text=True)
    if result.returncode != 0:
        return None

    # Run ddhi
    cmd_ddhi = [
        DDHI, "-1VP", "-stmp",
        "-llc.ddhi", "-Fddhi_list.tmp",
        "-ymax15.", s_file
    ]
    result = subprocess.run(cmd_ddhi, cwd=work_dir,
                            capture_output=True, text=True)
    if result.returncode != 0:
        return None

    # Read the output .dta files
    output = {}
    for comp in ["VCT0", "VCT1"]:
        dta_path = os.path.join(work_dir, f"tmp.{comp}.dta")
        if os.path.isfile(dta_path):
            data = np.loadtxt(dta_path)
            output[comp] = (data[:, 0], data[:, 1])  # pressure (neg), values

    return output if len(output) == 2 else None


def save_dta(filepath, pressure, values):
    """Save a .dta file in the same format as the budget pipeline."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for p, v in zip(pressure, values):
            f.write(f"  {p:25.16E}  {v:25.16E}\n")


def process_experiment(experiment):
    """Extract temperature profiles for all days of one experiment."""
    days = get_all_days(experiment)
    if not days:
        print(f"  No days found for {experiment}")
        return 0

    exp_label = EXP_LABELS[experiment]
    n_ok = 0

    for i, day in enumerate(days):
        input_path = os.path.join(
            RAW_DATA_DIR, experiment, "output", day, INPUT_FILE
        )
        if not os.path.isfile(input_path):
            continue

        # Output directory (daily)
        day_out = os.path.join(OUTPUT_DIR, experiment, day, VARIABLE, "data")

        # Check if already done
        vct0_out = os.path.join(day_out,
                                f"{VARIABLE}.{INPUT_FILE}.VCT0.dta")
        if os.path.isfile(vct0_out):
            n_ok += 1
            continue

        # Extract in a temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            result = extract_temperature(input_path, tmpdir)
            if result is None:
                print(f"  FAIL: {exp_label}/{day}")
                continue

            pressure = result["VCT0"][0]
            vct0 = result["VCT0"][1]
            vct1 = result["VCT1"][1]
            t_mean = (vct0 + vct1) / 2.0

            # Save VCT0, VCT1, and T_MEAN
            prefix = f"{VARIABLE}.{INPUT_FILE}"
            save_dta(os.path.join(day_out, f"{prefix}.VCT0.dta"),
                     pressure, vct0)
            save_dta(os.path.join(day_out, f"{prefix}.VCT1.dta"),
                     pressure, vct1)
            save_dta(os.path.join(day_out, f"{prefix}.T_MEAN.dta"),
                     pressure, t_mean)

        n_ok += 1
        if (i + 1) % 100 == 0:
            print(f"  {exp_label}: {i+1}/{len(days)} days processed")

    print(f"  {exp_label}: {n_ok}/{len(days)} days OK")
    return n_ok


def compute_yearly_average(experiment):
    """Compute 2-year average of temperature profiles and save."""
    days = get_all_days(experiment)
    exp_label = EXP_LABELS[experiment]

    for comp in ["VCT0", "VCT1", "T_MEAN"]:
        profiles = []
        pressure = None
        prefix = f"{VARIABLE}.{INPUT_FILE}"

        for day in days:
            fpath = os.path.join(
                OUTPUT_DIR, experiment, day, VARIABLE, "data",
                f"{prefix}.{comp}.dta"
            )
            if not os.path.isfile(fpath):
                continue
            data = np.loadtxt(fpath)
            if pressure is None:
                pressure = data[:, 0]
            profiles.append(data[:, 1])

        if len(profiles) == 0:
            print(f"  {exp_label}/{comp}: no data")
            continue

        avg = np.mean(profiles, axis=0)
        out_path = os.path.join(
            YEARLY_DIR, experiment, VARIABLE, "data",
            f"{prefix}.{comp}.dta"
        )
        save_dta(out_path, pressure, avg)
        print(f"  {exp_label}/{comp}: averaged {len(profiles)} days -> "
              f"{os.path.basename(out_path)}")


def main():
    print("=" * 60)
    print("DDH Temperature Profile Extraction")
    print("=" * 60)
    print(f"  Raw data:  {os.path.abspath(RAW_DATA_DIR)}")
    print(f"  Daily out: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Yearly:    {os.path.abspath(YEARLY_DIR)}")
    print(f"  ddht:      {DDHT}")
    print(f"  ddhi:      {DDHI}")
    print()

    if not DDHT or not DDHI:
        print("ERROR: ddht and/or ddhi not found in PATH")
        sys.exit(1)

    if not os.path.isfile(CONVERSION_LIST):
        print(f"ERROR: conversion list not found: {CONVERSION_LIST}")
        sys.exit(1)

    # Step 1: Extract daily temperature profiles
    print("Step 1: Extracting daily temperature profiles ...")
    for exp in EXPERIMENTS:
        print(f"\nProcessing {EXP_LABELS[exp]} ({exp}):")
        process_experiment(exp)

    # Step 2: Compute yearly averages
    print("\n" + "=" * 60)
    print("Step 2: Computing 2-year average temperature profiles ...")
    for exp in EXPERIMENTS:
        compute_yearly_average(exp)

    print("\nDone!")


if __name__ == "__main__":
    main()
