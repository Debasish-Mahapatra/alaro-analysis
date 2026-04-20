#!/usr/bin/env bash
#
# Validate CT budget for C1M (control) using ddhtoolbox native tools.
# Strategy:
#   1. Cumulate all 730 raw DHFDLABOF+0024 files using ddh_cumul
#      (SOMME_PONDEREE weights by forecast range, so identical 24h files
#       get averaged correctly: sum / N)
#   2. Run ddhb on the cumulated file to produce CT budget
#   3. Compare with our Python-computed 2yr averages
#

set -euo pipefail

DDH_ROOT="/mnt/scratch/MANAUS/DDH"
TOOLBOX_ROOT="$DDH_ROOT/ddhtoolbox"
SRC_ROOT="$DDH_ROOT/untar-data"
WORK_DIR="$DDH_ROOT/validate-ct-c1m"
RUNTIME_BPS="$DDH_ROOT/alaro-24h-budgets/_runtime/ddh_budget_lists"

export DDHTOOLBOX="$TOOLBOX_ROOT"
export DDHI_LIST="$RUNTIME_BPS/alaro/conversion_list"
export DDHB_BPS="$RUNTIME_BPS"
export DDH_PLOT=dd2gr
export PATH="$TOOLBOX_ROOT/tools:$TOOLBOX_ROOT/tools/lfa:$TOOLBOX_ROOT/tools/.dd2gr/src:$PATH"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "============================================"
echo "  Validate CT budget for C1M (control)"
echo "  Using ddhtoolbox native averaging"
echo "============================================"
echo ""

# ─── Step 1: Collect all raw 24h files ────────────────────────────────────────
echo "Step 1: Collecting raw DHFDLABOF+0024 file paths..."
RAW_FILES=()
for day_dir in $(ls -d "$SRC_ROOT/control/output/DDH20"* | sort); do
    f="$day_dir/DHFDLABOF+0024"
    if [[ -f "$f" ]]; then
        RAW_FILES+=("$f")
    fi
done
echo "  Found ${#RAW_FILES[@]} raw files"

# ─── Step 2: Cumulate using ddh_cumul ─────────────────────────────────────────
CUMUL_FILE="$WORK_DIR/DHFDLABOF+0024.cumul"
echo ""
echo "Step 2: Cumulating ${#RAW_FILES[@]} files with ddh_cumul..."
echo "  (This uses SOMME_PONDEREE internally, which does a weighted average)"
echo "  Output: $CUMUL_FILE"

# ddh_cumul takes: output_file input1 input2 ... inputN
# For 730 files we might hit argument limits, so do it in batches
rm -f "$CUMUL_FILE"

BATCH_SIZE=50
BATCH_NUM=0
TEMP_FILES=()

for ((i=0; i<${#RAW_FILES[@]}; i+=BATCH_SIZE)); do
    BATCH=("${RAW_FILES[@]:$i:$BATCH_SIZE}")
    BATCH_NUM=$((BATCH_NUM+1))
    BATCH_FILE="$WORK_DIR/batch_${BATCH_NUM}.tmp"

    echo "  Batch $BATCH_NUM: files $((i+1))-$((i+${#BATCH[@]}))..."
    ddh_cumul "$BATCH_FILE" "${BATCH[@]}" 2>&1 | tail -1
    TEMP_FILES+=("$BATCH_FILE")
done

echo ""
echo "  Merging $BATCH_NUM batches..."
if [[ ${#TEMP_FILES[@]} -eq 1 ]]; then
    mv "${TEMP_FILES[0]}" "$CUMUL_FILE"
else
    # Cumulate all batch files together
    ddh_cumul "$CUMUL_FILE" "${TEMP_FILES[@]}" 2>&1 | tail -1
    rm -f "${TEMP_FILES[@]}"
fi

echo "  Cumulated file: $(ls -lh "$CUMUL_FILE" | awk '{print $5}')"

# ─── Step 3: Run ddhb for CT on the cumulated file ───────────────────────────
echo ""
echo "Step 3: Running ddhb for CT on cumulated file..."
DDHB_OUT="$WORK_DIR/ddhb_output"
mkdir -p "$DDHB_OUT"

ddhb -v "alaro/CT" -i "$CUMUL_FILE" -r "$DDHB_OUT" -o "$WORK_DIR/CT_validate.svg" 2>&1

echo ""
echo "Step 4: Budget output files:"
ls -la "$DDHB_OUT/budget.alaro/"*.dta 2>/dev/null || echo "  Checking alternative locations..."
find "$DDHB_OUT" -name "*.dta" -ls 2>/dev/null

echo ""
echo "============================================"
echo "  Done! Compare these .dta files with:"
echo "  $DDH_ROOT/alaro-24h-budgets/results/control/DDH20140101/CT/data/"
echo "============================================"
