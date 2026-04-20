#!/bin/bash
# Process one day's 24 hourly DDH files. Called by the parallel driver.
# Usage: extract_one_day.sh <experiment> <day> <worker_id>

experiment=$1
day=$2
worker_id=$3

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../untar-data"
OUT_DIR="$SCRIPT_DIR/../diurnal-hourly-data"
CONV_LIST="$SCRIPT_DIR/../alaro-24h-budgets/_runtime/ddh_budget_lists/alaro/conversion_list"
WORK="$SCRIPT_DIR/../_work_diurnal_${worker_id}"

daydir="$RAW_DIR/$experiment/output/$day"

# Skip if already done (all 24 hours)
if [ -f "$OUT_DIR/$experiment/$day/hour_24/tmp.VCT1.dta" ]; then
    exit 0
fi

mkdir -p "$WORK"

# Write extraction lists
cat > "$WORK/lc.ddht" << 'EOF'
0
VCT0 +
ECR"VCT0"
0
VCT1 +
ECR"VCT1"
0
VQV0 +
ECR"VQV0"
0
VQV1 +
ECR"VQV1"
0
VQL0 +
ECR"VQL0"
0
VQL1 +
ECR"VQL1"
0
VQN0 +
ECR"VQN0"
0
VQN1 +
ECR"VQN1"
0
VQR0 +
ECR"VQR0"
0
VQR1 +
ECR"VQR1"
0
VQS0 +
ECR"VQS0"
0
VQS1 +
ECR"VQS1"
0
VQG0 +
ECR"VQG0"
0
VQG1 +
ECR"VQG1"
0
VUU0 +
ECR"VUU0"
0
VUU1 +
ECR"VUU1"
0
VVV0 +
ECR"VVV0"
0
VVV1 +
ECR"VVV1"
0
VTK0 +
ECR"VTK0"
0
VTK1 +
ECR"VTK1"
0
VTT0 +
ECR"VTT0"
0
VTT1 +
ECR"VTT1"
EOF

printf "VCT0\nVCT1\nVQL0\nVQL1\nVQN0\nVQN1\nVQR0\nVQR1\nVQS0\nVQS1\nVQG0\nVQG1\nVQV0\nVQV1\nVUU0\nVUU1\nVVV0\nVVV1\nVTK0\nVTK1\nVTT0\nVTT1\n" > "$WORK/lc.ddhi"
cp "$CONV_LIST" "$WORK/ddhi_list.tmp"

cd "$WORK"

for h in $(seq 1 24); do
    hh=$(printf "%02d" $h)
    hfile="DHFDLABOF+00${hh}"
    hourout="$OUT_DIR/$experiment/$day/hour_${hh}"

    [ -f "$hourout/tmp.VCT1.dta" ] && continue
    [ ! -f "$daydir/$hfile" ] && continue

    cp "$daydir/$hfile" .
    ddht -cCALC -1"$hfile" -s"${hfile}.s" -llc.ddht >/dev/null 2>&1 || continue
    ddhi -1VP -stmp -llc.ddhi -Fddhi_list.tmp -ymax15. "${hfile}.s" >/dev/null 2>&1 || continue

    mkdir -p "$hourout"
    mv tmp.V*.dta "$hourout/" 2>/dev/null
    rm -f "$hfile" "${hfile}.s" tmp.*.doc
done

echo "$experiment/$day"
