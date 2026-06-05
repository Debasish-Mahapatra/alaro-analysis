#!/usr/bin/env bash
# Rebuild common-valid hourly rainfall (Radar, IMERG, C1M, G1M, G2M, G2M-XCU)
# from scratch: conservative CDO remapcon + radar mask + valid_time_mask +
# start-labelled hourly bins, all on one consistent grid.
# Usage: nohup bash run_precip_conservative_common_valid.sh > run_precip_conservative.log 2>&1 &

set -euo pipefail

source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
conda activate epygram

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# CDO is invoked by absolute path inside the workflow (no module load needed).
python alaro_analysis/workflows/precip_conservative_common_valid.py --workers 32 "$@"

echo "Conservative common-valid rainfall rebuild complete."
