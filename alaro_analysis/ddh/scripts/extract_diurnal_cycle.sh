#!/bin/bash
# Extract hourly state variables from raw DDH files for diurnal cycle analysis.
#
# For each experiment and sampled day, extracts V??1 (final state) from
# all 24 hourly DDH files using ddht+ddhi. Output goes to:
#   diurnal-hourly-data/<experiment>/<DDHdate>/hour_<HH>/tmp.V*.dta
#
# After extraction, run compute_diurnal_average.py to deaccumulate,
# average, and plot.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../untar-data"
OUT_DIR="$SCRIPT_DIR/../diurnal-hourly-data"
WORK_DIR="$SCRIPT_DIR/../_work_diurnal"
CONV_LIST="$SCRIPT_DIR/../alaro-24h-budgets/_runtime/ddh_budget_lists/alaro/conversion_list"

DDHT=$(which ddht)
DDHI=$(which ddhi)

SAMPLE_EVERY=5   # process every Nth day

# Write ddht command list (extract all state variables)
write_ddht_list() {
    cat > "$1" << 'DDHTEOF'
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
DDHTEOF
}

write_ddhi_list() {
    cat > "$1" << 'DDHIEOF'
VCT0
VCT1
VQL0
VQL1
VQN0
VQN1
VQR0
VQR1
VQS0
VQS1
VQG0
VQG1
VQV0
VQV1
VUU0
VUU1
VVV0
VVV1
VTK0
VTK1
VTT0
VTT1
DDHIEOF
}

echo "============================================================"
echo "DDH Diurnal Cycle Extraction"
echo "  Raw data:  $RAW_DIR"
echo "  Output:    $OUT_DIR"
echo "  Sampling:  every ${SAMPLE_EVERY}th day"
echo "============================================================"
echo

for experiment in control graupel 2mom; do
    exp_dir="$RAW_DIR/$experiment/output"
    if [ ! -d "$exp_dir" ]; then
        echo "Skipping $experiment (no data)"
        continue
    fi

    # Get sampled days
    all_days=($(ls "$exp_dir" | grep "^DDH" | sort))
    n_total=${#all_days[@]}
    echo "Processing $experiment ($n_total total days, sampling every ${SAMPLE_EVERY}th)..."

    count=0
    n_ok=0
    for day in "${all_days[@]}"; do
        count=$((count + 1))
        if [ $((count % SAMPLE_EVERY)) -ne 1 ]; then
            continue
        fi

        day_dir="$exp_dir/$day"

        # Check all 24 hourly files exist
        all_exist=true
        for h in $(seq -w 1 24); do
            hfile="$day_dir/DHFDLABOF+00${h}"
            if [ ! -f "$hfile" ]; then
                all_exist=false
                break
            fi
        done
        if [ "$all_exist" = false ]; then
            continue
        fi

        # Process each hour
        day_ok=true
        for h in $(seq 1 24); do
            hh=$(printf "%02d" $h)
            hfile="DHFDLABOF+00${hh}"
            input_path="$day_dir/$hfile"
            hour_out="$OUT_DIR/$experiment/$day/hour_${hh}"

            # Skip if already done
            if [ -f "$hour_out/tmp.VCT1.dta" ]; then
                continue
            fi

            # Set up work directory
            mkdir -p "$WORK_DIR"
            cp "$input_path" "$WORK_DIR/$hfile"
            write_ddht_list "$WORK_DIR/lc.ddht"
            write_ddhi_list "$WORK_DIR/lc.ddhi"
            cp "$CONV_LIST" "$WORK_DIR/ddhi_list.tmp"

            # Run ddht + ddhi from within the work directory
            cd "$WORK_DIR"
            $DDHT -cCALC -1"$hfile" -s"${hfile}.s" -llc.ddht >/dev/null 2>&1
            if [ $? -ne 0 ]; then
                day_ok=false
                cd "$SCRIPT_DIR"
                rm -f "$WORK_DIR/$hfile" "$WORK_DIR/${hfile}.s"
                continue
            fi

            $DDHI -1VP -stmp -llc.ddhi -Fddhi_list.tmp -ymax15. "${hfile}.s" >/dev/null 2>&1
            if [ $? -ne 0 ]; then
                day_ok=false
                cd "$SCRIPT_DIR"
                rm -f "$WORK_DIR/$hfile" "$WORK_DIR/${hfile}.s"
                continue
            fi

            # Move results to output directory
            mkdir -p "$hour_out"
            mv tmp.V*.dta "$hour_out/" 2>/dev/null
            cd "$SCRIPT_DIR"

            # Clean up work files
            rm -f "$WORK_DIR/$hfile" "$WORK_DIR/${hfile}.s" "$WORK_DIR"/tmp.*.doc
        done

        if [ "$day_ok" = true ]; then
            n_ok=$((n_ok + 1))
        fi

        if [ $((n_ok % 10)) -eq 0 ] && [ $n_ok -gt 0 ]; then
            echo "  $experiment: $n_ok days done"
        fi
    done
    echo "  $experiment: $n_ok days completed"
done

# Clean up work directory
rm -rf "$WORK_DIR"

echo
echo "Done! Hourly data saved to: $OUT_DIR"
echo "Now run: python compute_diurnal_average.py"
