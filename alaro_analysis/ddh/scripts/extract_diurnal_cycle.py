#!/usr/bin/env python3
"""
Extract hourly state variables from raw DDH files and compute the
mean diurnal cycle for all budget variables.

Pipeline:
  1. For each experiment, sample every Nth day
  2. For each sampled day, extract V??1 from all 24 hourly files
     (and V??0 from +0001 for the initial state)
  3. Compute hourly change: state(h) - state(h-1)
  4. Average over all sampled days → mean diurnal cycle [24 x 87]
  5. Save to diurnal-cycle-data/ for plotting

Note: V??1 are instantaneous state values (not accumulated), so
differencing consecutive hours gives the tendency for that hour.
The BUDGET components (fluxes/tendencies) are accumulated, but the
state variables are not — they are snapshots at each output time.
"""

import os
import sys
import shutil
import subprocess
import numpy as np

# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "untar-data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "diurnal-cycle-data")
WORK_DIR = os.path.join(SCRIPT_DIR, "..", "_work_diurnal")

CONVERSION_LIST = os.path.join(
    SCRIPT_DIR, "..", "alaro-24h-budgets", "_runtime",
    "ddh_budget_lists", "alaro", "conversion_list"
)

DDHT = shutil.which("ddht")
DDHI = shutil.which("ddhi")

EXPERIMENTS = ["control", "graupel", "2mom"]
EXP_LABELS = {"control": "C1M", "graupel": "G1M", "2mom": "G2M"}

# Sample every Nth day for efficiency (1 = all days, 5 = every 5th)
SAMPLE_EVERY = 5
N_HOURS = 24

# Map from budget variable name to DDH state variable article names
# V??0 = initial state, V??1 = state at forecast hour h
VAR_ARTICLES = {
    "CT":  ("VCT0", "VCT1"),   # Temperature (K)
    "QV":  ("VQV0", "VQV1"),   # Water vapour (g/kg)
    "QL":  ("VQL0", "VQL1"),   # Cloud water (g/kg)
    "QI":  ("VQN0", "VQN1"),   # Cloud ice (g/kg) — QI stored as QN
    "QR":  ("VQR0", "VQR1"),   # Rain (g/kg)
    "QS":  ("VQS0", "VQS1"),   # Snow (g/kg)
    "QG":  ("VQG0", "VQG1"),   # Graupel (g/kg)
    "UU":  ("VUU0", "VUU1"),   # Zonal wind (m/s)
    "VV":  ("VVV0", "VVV1"),   # Meridional wind (m/s)
    "TKE": ("VTK0", "VTK1"),   # TKE (J/kg)
    "TTE": ("VTT0", "VTT1"),   # TTE (J/kg)
}

# Units for the hourly CHANGE (delta per hour)
VAR_UNITS = {
    "CT": "K/h", "QV": "g/kg/h", "QL": "g/kg/h", "QI": "g/kg/h",
    "QR": "g/kg/h", "QS": "g/kg/h", "QG": "g/kg/h",
    "UU": "m/s/h", "VV": "m/s/h", "TKE": "J/kg/h", "TTE": "J/kg/h",
}

# Build ddht command list (extract all state variables in one pass)
_all_articles = set()
for v0, v1 in VAR_ARTICLES.values():
    _all_articles.add(v0)
    _all_articles.add(v1)

LC_DDHT = ""
for art in sorted(_all_articles):
    LC_DDHT += f"0\n{art} +\nECR\"{art}\"\n"

LC_DDHI = "\n".join(sorted(_all_articles)) + "\n"


def get_all_days(experiment):
    exp_dir = os.path.join(RAW_DATA_DIR, experiment, "output")
    if not os.path.isdir(exp_dir):
        return []
    return sorted(d for d in os.listdir(exp_dir) if d.startswith("DDH"))


def extract_hourly_states(input_path, work_dir):
    """Extract all V??0/V??1 state variables from one hourly DDH file.

    Returns dict of article_name -> (pressure, values) or None on failure.
    """
    input_basename = os.path.basename(input_path)
    shutil.copy2(input_path, os.path.join(work_dir, input_basename))

    with open(os.path.join(work_dir, "lc.ddht"), "w") as f:
        f.write(LC_DDHT)
    with open(os.path.join(work_dir, "lc.ddhi"), "w") as f:
        f.write(LC_DDHI)
    shutil.copy2(CONVERSION_LIST, os.path.join(work_dir, "ddhi_list.tmp"))

    s_file = f"{input_basename}.s"

    # Use os.system with cd to avoid dyld issues in subprocess
    saved_cwd = os.getcwd()
    os.chdir(work_dir)

    rc = os.system(
        f'{DDHT} -cCALC -1{input_basename} -s{s_file} -llc.ddht '
        f'>/dev/null 2>&1'
    )
    if rc != 0:
        os.chdir(saved_cwd)
        return None

    rc = os.system(
        f'{DDHI} -1VP -stmp -llc.ddhi -Fddhi_list.tmp '
        f'-ymax15. {s_file} >/dev/null 2>&1'
    )
    os.chdir(saved_cwd)
    if rc != 0:
        return None

    # Read output .dta files
    result = {}
    for art in _all_articles:
        dta = os.path.join(work_dir, f"tmp.{art}.dta")
        if os.path.isfile(dta):
            data = np.loadtxt(dta)
            result[art] = (data[:, 0], data[:, 1])

    # Clean up
    for f in os.listdir(work_dir):
        if f.startswith("tmp.") or f.endswith(".s") or f == input_basename:
            os.remove(os.path.join(work_dir, f))

    return result


def process_experiment(experiment):
    """Compute mean diurnal cycle for one experiment.

    Returns dict[var] -> np.array shape [24, 87] of hourly changes,
    and the pressure array.
    """
    days = get_all_days(experiment)
    sampled = days[::SAMPLE_EVERY]
    exp_label = EXP_LABELS[experiment]
    print(f"  {exp_label}: {len(sampled)} sampled days "
          f"(every {SAMPLE_EVERY}th of {len(days)})")

    # Accumulators: sum of hourly changes and count
    # diurnal[var][hour, level]
    n_lev = 87
    diurnal_sum = {var: np.zeros((N_HOURS, n_lev)) for var in VAR_ARTICLES}
    diurnal_count = {var: np.zeros((N_HOURS, n_lev)) for var in VAR_ARTICLES}
    pressure = None

    os.makedirs(WORK_DIR, exist_ok=True)

    for di, day in enumerate(sampled):
        day_dir = os.path.join(RAW_DATA_DIR, experiment, "output", day)

        # Extract state from all 24 hourly files
        hourly_states = {}  # hour -> {article: (p, v)}
        ok = True
        for h in range(1, N_HOURS + 1):
            fname = f"DHFDLABOF+{h:04d}"
            fpath = os.path.join(day_dir, fname)
            if not os.path.isfile(fpath):
                ok = False
                break
            states = extract_hourly_states(fpath, WORK_DIR)
            if states is None:
                ok = False
                break
            hourly_states[h] = states

        if not ok:
            continue

        # Get pressure from any extracted file
        if pressure is None:
            sample_art = list(VAR_ARTICLES.values())[0][1]
            if sample_art in hourly_states[1]:
                pressure = np.abs(hourly_states[1][sample_art][0])

        # Compute hourly changes for each variable
        for var, (art0, art1) in VAR_ARTICLES.items():
            for h in range(1, N_HOURS + 1):
                if art1 not in hourly_states[h]:
                    continue

                v_current = hourly_states[h][art1][1]

                if h == 1:
                    # First hour: state(1) - initial state
                    if art0 in hourly_states[1]:
                        v_prev = hourly_states[1][art0][1]
                    else:
                        continue
                else:
                    # Subsequent hours: state(h) - state(h-1)
                    if art1 in hourly_states[h - 1]:
                        v_prev = hourly_states[h - 1][art1][1]
                    else:
                        continue

                change = v_current - v_prev
                diurnal_sum[var][h - 1, :len(change)] += change
                diurnal_count[var][h - 1, :len(change)] += 1

        if (di + 1) % 20 == 0:
            print(f"    {di+1}/{len(sampled)} days done")

    print(f"    {len(sampled)} days done")

    # Compute mean
    diurnal_mean = {}
    for var in VAR_ARTICLES:
        with np.errstate(invalid='ignore'):
            mean = np.where(diurnal_count[var] > 0,
                            diurnal_sum[var] / diurnal_count[var], 0.0)
        diurnal_mean[var] = mean

    return diurnal_mean, pressure


def save_diurnal(experiment, diurnal_mean, pressure):
    """Save mean diurnal cycle data as .npy files."""
    exp_dir = os.path.join(OUTPUT_DIR, experiment)
    os.makedirs(exp_dir, exist_ok=True)

    np.save(os.path.join(exp_dir, "pressure.npy"), pressure)
    for var, data in diurnal_mean.items():
        np.save(os.path.join(exp_dir, f"{var}_diurnal.npy"), data)
        print(f"    {var}: saved {data.shape}")


def main():
    print("=" * 60)
    print("DDH Diurnal Cycle Extraction")
    print("=" * 60)
    print(f"  Raw data:  {os.path.abspath(RAW_DATA_DIR)}")
    print(f"  Output:    {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Sampling:  every {SAMPLE_EVERY}th day")
    print(f"  ddht:      {DDHT}")
    print(f"  ddhi:      {DDHI}")
    print()

    if not DDHT or not DDHI:
        print("ERROR: ddht/ddhi not found in PATH")
        sys.exit(1)

    for exp in EXPERIMENTS:
        print(f"\nProcessing {EXP_LABELS[exp]} ({exp}):")
        diurnal_mean, pressure = process_experiment(exp)
        if pressure is not None:
            save_diurnal(exp, diurnal_mean, pressure)
        else:
            print(f"  No data extracted for {exp}")

    # Clean up work directory
    shutil.rmtree(WORK_DIR, ignore_errors=True)

    print("\nDone! Diurnal cycle data saved to:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
