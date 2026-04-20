#!/usr/bin/env python3
"""
Compute mean diurnal cycle from hourly-extracted DDH state variables.

Reads hourly .dta files from diurnal-hourly-data/, deaccumulates
(state(h) - state(h-1)), averages over all days, and saves as .npy
files for the Hovmoller plotting script.
"""

import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOURLY_DIR = os.path.join(SCRIPT_DIR, "..", "diurnal-hourly-data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "diurnal-cycle-data")

EXPERIMENTS = ["control", "graupel", "2mom"]
EXP_LABELS = {"control": "C1M", "graupel": "G1M", "2mom": "G2M"}

# Map budget variable → (V??0 article, V??1 article)
VAR_ARTICLES = {
    "CT":  ("VCT0", "VCT1"),
    "QV":  ("VQV0", "VQV1"),
    "QL":  ("VQL0", "VQL1"),
    "QI":  ("VQN0", "VQN1"),
    "QR":  ("VQR0", "VQR1"),
    "QS":  ("VQS0", "VQS1"),
    "QG":  ("VQG0", "VQG1"),
    "UU":  ("VUU0", "VUU1"),
    "VV":  ("VVV0", "VVV1"),
    "TKE": ("VTK0", "VTK1"),
    "TTE": ("VTT0", "VTT1"),
}

N_HOURS = 24
N_LEV = 87


def read_dta(fpath):
    if not os.path.isfile(fpath):
        return None, None
    data = np.loadtxt(fpath)
    return np.abs(data[:, 0]), data[:, 1]


def process_experiment(experiment):
    exp_dir = os.path.join(HOURLY_DIR, experiment)
    if not os.path.isdir(exp_dir):
        print(f"  {experiment}: no data directory")
        return

    days = sorted(d for d in os.listdir(exp_dir) if d.startswith("DDH"))
    print(f"  {EXP_LABELS[experiment]}: {len(days)} days found")

    # Accumulators
    diurnal_sum = {var: np.zeros((N_HOURS, N_LEV)) for var in VAR_ARTICLES}
    diurnal_count = {var: np.zeros((N_HOURS, N_LEV)) for var in VAR_ARTICLES}
    pressure = None

    for day in days:
        day_dir = os.path.join(exp_dir, day)

        # Check all 24 hours exist
        all_ok = all(
            os.path.isdir(os.path.join(day_dir, f"hour_{h:02d}"))
            for h in range(1, 25)
        )
        if not all_ok:
            continue

        for var, (art0, art1) in VAR_ARTICLES.items():
            for h in range(1, N_HOURS + 1):
                hour_dir = os.path.join(day_dir, f"hour_{h:02d}")

                # Current state
                _, v_curr = read_dta(
                    os.path.join(hour_dir, f"tmp.{art1}.dta"))
                if v_curr is None:
                    continue

                # Previous state
                if h == 1:
                    # Use initial state V??0 from hour 01
                    _, v_prev = read_dta(
                        os.path.join(hour_dir, f"tmp.{art0}.dta"))
                else:
                    prev_dir = os.path.join(day_dir, f"hour_{h-1:02d}")
                    _, v_prev = read_dta(
                        os.path.join(prev_dir, f"tmp.{art1}.dta"))
                if v_prev is None:
                    continue

                # Get pressure from first successful read
                if pressure is None:
                    pressure, _ = read_dta(
                        os.path.join(hour_dir, f"tmp.{art1}.dta"))

                change = v_curr - v_prev
                n = min(len(change), N_LEV)
                diurnal_sum[var][h - 1, :n] += change[:n]
                diurnal_count[var][h - 1, :n] += 1

    # Compute mean and save
    out_dir = os.path.join(OUTPUT_DIR, experiment)
    os.makedirs(out_dir, exist_ok=True)

    if pressure is not None:
        np.save(os.path.join(out_dir, "pressure.npy"), pressure)

    for var in VAR_ARTICLES:
        with np.errstate(invalid='ignore'):
            mean = np.where(diurnal_count[var] > 0,
                            diurnal_sum[var] / diurnal_count[var], 0.0)
        np.save(os.path.join(out_dir, f"{var}_diurnal.npy"), mean)
        n_days = int(diurnal_count[var].max())
        print(f"    {var}: {mean.shape}, {n_days} days averaged")


def main():
    print("=" * 60)
    print("Compute Mean Diurnal Cycle")
    print(f"  Input:  {os.path.abspath(HOURLY_DIR)}")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)

    for exp in EXPERIMENTS:
        print(f"\n{EXP_LABELS[exp]}:")
        process_experiment(exp)

    print("\nDone! Now run: python plot_diurnal_hovmoller.py")


if __name__ == "__main__":
    main()
