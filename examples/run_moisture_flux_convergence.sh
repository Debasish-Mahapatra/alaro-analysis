#!/usr/bin/env bash
# Vertically integrated moisture flux convergence, G1M vs C1M, over the full
# two-year period from raw FA files.
# Usage: nohup bash run_moisture_flux_convergence.sh > run_moisture_flux_convergence.log 2>&1 &

set -euo pipefail

source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
conda activate epygram

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python scripts/make_moisture_flux_convergence_g1m_c1m.py \
    --experiments control graupel \
    --workers 32 \
    "$@"

echo "Moisture flux convergence (G1M vs C1M) complete."
