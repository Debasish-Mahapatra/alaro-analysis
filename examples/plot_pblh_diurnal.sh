#!/usr/bin/env bash
# Plot mean diurnal cycle of boundary-layer height for all experiments.
# Usage: nohup bash plot_pblh_diurnal.sh > plot_pblh_diurnal.log 2>&1 &

set -euo pipefail

source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
conda activate epygram

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python plot_pblh_diurnal.py \
    --control-dir /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO/control/masked-netcdf-2 \
    --graupel-dir /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO/graupel/masked-netcdf-2 \
    --twomom-dir  /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO/2mom/masked-netcdf-2 \
    --variable "CLPMHAUT.MOD.XFU" \
    --variable-label "Boundary layer height" \
    --variable-unit "m" \
    --output-dir /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/surface \
    --intermediate-dir /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/surface \
    --zoom-inset

echo "PBLH diurnal cycle plots complete."
