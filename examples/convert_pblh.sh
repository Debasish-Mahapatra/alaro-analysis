#!/usr/bin/env bash
# Convert CLPMHAUT.MOD.XFU (PBL height) plus LCL inputs from FA to masked
# NetCDF for all experiments.
# Usage: nohup bash convert_pblh.sh > convert_pblh.log 2>&1 &

set -euo pipefail

source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
conda activate epygram

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASK=/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/mask/Radar_mask_latlon.nc
VARS=(
    "CLPMHAUT.MOD.XFU"
    "H00100TEMPERATUR"
    "H00100HUMI.SPECI"
    "H00100PRESSURE"
)

for EXP in control graupel 2mom; do
    INPUT="/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO/${EXP}/untar-output"
    OUTPUT="/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO/${EXP}/masked-netcdf-2"

    echo ""
    echo "============================================="
    echo "  Converting ${VARS[*]} for experiment: ${EXP}"
    echo "  Input:  ${INPUT}"
    echo "  Output: ${OUTPUT}"
    echo "============================================="
    echo ""

    python -m alaro_analysis.converter.cli \
        "${INPUT}" \
        "${OUTPUT}" \
        --vars "${VARS[@]}" \
        --mask-file "${MASK}" \
        --workers 16 \
        --no-overwrite \
        --skip-incomplete-days

    echo ""
    echo "[done] ${EXP} conversion complete."
    echo ""
done

echo "All experiments converted."
