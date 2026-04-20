#!/usr/bin/env bash
# Run ddhb on every (experiment, day, variable) to produce per-block .dta files.
#
# Layout:
#   SRC = /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-untar/{exp}/output/DDH20YYMMDD/DHFDLABOF+0024
#   OUT = /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/{exp}/{day}/{var}/*.dta
#   LOG = /mnt/HDS_CLIMATE/CLIMATE/deba/alaro-analysis/cache/logs/run_ddh_budgets.log
#
# Per-experiment runtime budget-list directories are set up under OUT/_runtime/{exp}/
# with the correct CT.fbl (2-ice for control C1M, 3-ice for graupel G1M / 2mom G2M).
# control has no graupel, so QG is skipped.

set -euo pipefail

TOOLBOX=/mnt/HDS_CLIMATE/CLIMATE/deba/ddhtoolbox
SRC=/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-untar
OUT_BASE=/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed
LOGDIR=/mnt/HDS_CLIMATE/CLIMATE/deba/alaro-analysis/cache/logs
# Per-lead input:  DHFDLABOF+<HHHH>.  Override via LEAD env var (e.g. LEAD=0012).
LEAD=${LEAD:-0024}
INPUT=DHFDLABOF+$LEAD
# Y-axis coordinate:  VZ (altitude km) or VP (pressure hPa).  Default VZ.
YCOOR=${YCOOR:-VZ}
# Output root; keep separate trees per lead so re-runs do not clobber each other.
OUT_TREE=${OUT_TREE:-"$LEAD"}
JOBS=${JOBS:-32}
OUT="$OUT_BASE/lead${OUT_TREE}_${YCOOR}"
MAIN_LOG="$LOGDIR/run_ddh_budgets_${OUT_TREE}_${YCOOR}.log"
FAIL_LOG="$LOGDIR/run_ddh_budgets_${OUT_TREE}_${YCOOR}.fail.log"
mkdir -p "$OUT" "$LOGDIR"

# --------------------------------------------------------------------------
# Prepare per-experiment runtime budget-list directories.
# --------------------------------------------------------------------------
prepare_runtime() {
  local exp=$1
  local ct_choice=$2   # "2ice" for control, "3ice" for graupel / 2mom
  local rt="$OUT/_runtime/$exp/ddh_budget_lists/alaro"
  mkdir -p "$rt"

  # Copy the per-variable fbl files we need.
  cp "$TOOLBOX/ddh_budget_lists/alaro/"{QV.fbl,QL.fbl,QI.fbl,QR.fbl,QS.fbl,TKE.fbl,TTE.fbl,UU.fbl,VV.fbl} "$rt/"
  # Graupel fbl only for 3-ice experiments.
  if [[ "$ct_choice" == "3ice" ]]; then
    cp "$TOOLBOX/ddh_budget_lists/alaro/QG.fbl" "$rt/"
    cp "$TOOLBOX/ddh_budget_lists/alaro/CT.fbl-3ice" "$rt/CT.fbl"
  else
    cp "$TOOLBOX/ddh_budget_lists/alaro/CT3.fbl-2ice" "$rt/CT.fbl"
  fi
  cp "$TOOLBOX/ddh_budget_lists/alaro/conversion_list" "$rt/conversion_list"

  # Upstream Q*.fbl files have the main variable name commented out on the
  # line after MAIN (e.g. "MAIN QL" then "# QL"), which makes ddhi divide by
  # zero normalising mass and SIGFPE.  Uncomment it.
  for f in QL QI QR QS QG UU VV TKE TTE; do
    fbl="$rt/$f.fbl"
    [[ -f "$fbl" ]] || continue
    sed -i "0,/^# $f$/s//$f/" "$fbl"
  done

  # The DDH files use "QN" (neige = ice) as the main ice variable, not QI.
  # Rewrite MAIN QI → MAIN QN in QI.fbl, and patch conversion_list so the
  # usual VQI0 / VQI1 / VQIM references resolve via VQN0 / VQN1.  Also add
  # an explicit VQNM row so ddhi can use that name directly.
  if [[ -f "$rt/QI.fbl" ]]; then
    sed -i '0,/^MAIN QI$/s//MAIN QN/; 0,/^QI$/s//QN/' "$rt/QI.fbl"
  fi
  awk '
    /^VII0VQI0/ {
      print "VII0VQI0         VQN0                      ICE WATER : INITIAL VALUE                                   g/kg                     1000.000  0"
      next
    }
    /^VII1VQI1/ {
      print "VII1VQI1         VQN1                      ICE WATER : FINAL VALUE                                     g/kg                     1000.000  0"
      next
    }
    /^BUIIVQIM/ {
      print "BUIIVQIM         VQN1         VQN0         ICE WATER : MEAN TENDENCY                                   g/kg/day             86400000.000  0"
      next
    }
    { print }
  ' "$rt/conversion_list" > "$rt/conversion_list.tmp" && mv "$rt/conversion_list.tmp" "$rt/conversion_list"
  cat <<'EOF' >> "$rt/conversion_list"
BUIIVQNM         VQN1         VQN0         ICE WATER : MEAN TENDENCY                                   g/kg/day             86400000.000  0
EOF
  # QG.fbl also has a stray literal "0 -" in the auto-cv block that tickles
  # the same FPE; strip that block.
  if [[ -f "$rt/QG.fbl" ]]; then
    awk '
      /^BEGIN BLOCK auto-cv$/ { skip=1; next }
      skip && /^# -+$/          { skip=0; next }
      !skip                     { print }
    ' "$rt/QG.fbl" > "$rt/QG.fbl.tmp" && mv "$rt/QG.fbl.tmp" "$rt/QG.fbl"
  fi
}

# --------------------------------------------------------------------------
# One ddhb invocation for (exp, day, variable).
# --------------------------------------------------------------------------
run_one() {
  local exp=$1 day=$2 var=$3
  local src_day="$SRC/$exp/output/$day"
  local input="$src_day/$INPUT"
  local out_dir="$OUT/$exp/$day/$var"
  local done_flag="$out_dir/done.ok"
  local log_file="$out_dir/run.log"

  if [[ -f "$done_flag" ]]; then
    echo "SKIP $exp $day $var"
    return 0
  fi
  if [[ ! -f "$input" ]]; then
    echo "MISSING $exp $day $var ($input)" | tee -a "$FAIL_LOG"
    return 1
  fi

  mkdir -p "$out_dir" "$OUT/_work"
  local work_dir
  work_dir=$(mktemp -d "$OUT/_work/${exp}_${day}_${var}.XXXXXX")

  {
    export DDHTOOLBOX="$TOOLBOX"
    export DDHB_BPS="$OUT/_runtime/$exp/ddh_budget_lists"
    export DDHI_LIST="$OUT/_runtime/$exp/ddh_budget_lists/alaro/conversion_list"
    # DDH_PLOT intentionally unset so ddhb skips SVG rendering (we only need
    # the .dta files).  See ddhb line 316 "no DDH_PLOT provided".
    unset DDH_PLOT
    export PATH="$TOOLBOX/tools:$TOOLBOX/tools/lfa:$TOOLBOX/tools/.dd2gr/src:$PATH"

    cd "$src_day"
    ddhb -v "alaro/$var" -i "$INPUT" -Y "$YCOOR" -r "$work_dir"
  } >"$log_file" 2>&1

  if [[ -d "$work_dir/budget.alaro" ]]; then
    cp "$work_dir"/budget.alaro/*.dta "$out_dir/" 2>/dev/null || true
    touch "$done_flag"
    rm -rf "$work_dir"
    echo "OK $exp $day $var"
  else
    echo "FAIL $exp $day $var" | tee -a "$FAIL_LOG"
    rm -rf "$work_dir"
    return 1
  fi
}
export -f run_one
export TOOLBOX SRC OUT OUT_BASE LOGDIR FAIL_LOG INPUT LEAD YCOOR OUT_TREE

# --------------------------------------------------------------------------
# Worker entry point (for xargs).
# --------------------------------------------------------------------------
if [[ ${1:-} == "--worker" ]]; then
  # Ensure runtime exists (idempotent); workers may race on first call but
  # cp/mkdir are safe.
  exp_rt="$OUT/_runtime/$2/ddh_budget_lists/alaro"
  if [[ ! -f "$exp_rt/CT.fbl" ]]; then
    case "$2" in
      control) prepare_runtime control 2ice ;;
      graupel) prepare_runtime graupel 3ice ;;
      2mom)    prepare_runtime 2mom    3ice ;;
    esac
  fi
  run_one "$2" "$3" "$4"
  exit $?
fi

# --------------------------------------------------------------------------
# Main: prepare runtimes and fire xargs.
# --------------------------------------------------------------------------
: > "$MAIN_LOG"
: > "$FAIL_LOG"
mkdir -p "$OUT/_work"

echo "[$(date)] Preparing runtime budget-list dirs..." | tee -a "$MAIN_LOG"
prepare_runtime control 2ice
prepare_runtime graupel 3ice
prepare_runtime 2mom    3ice

declare -A VARS_2ICE=( [list]="QV QL QI QR QS UU VV" )
declare -A VARS_3ICE=( [list]="QV QL QI QR QS QG UU VV" )

tasks_file="$LOGDIR/run_ddh_budgets_${OUT_TREE}_${YCOOR}.tasks"
: > "$tasks_file"
for exp in control graupel 2mom; do
  if [[ "$exp" == "control" ]]; then
    vars=${VARS_2ICE[list]}
  else
    vars=${VARS_3ICE[list]}
  fi
  find "$SRC/$exp/output" -maxdepth 1 -mindepth 1 -type d -name 'DDH20*' | sort | while read -r day_dir; do
    day=$(basename "$day_dir")
    for v in $vars; do
      printf '%s\t%s\t%s\n' "$exp" "$day" "$v" >> "$tasks_file"
    done
  done
done

n_tasks=$(wc -l < "$tasks_file")
echo "[$(date)] $n_tasks tasks queued across $JOBS workers" | tee -a "$MAIN_LOG"

awk -F '\t' '{printf "%s\0%s\0%s\0", $1, $2, $3}' "$tasks_file" \
  | xargs -0 -n3 -P"$JOBS" "$0" --worker \
  >> "$MAIN_LOG" 2>&1 || true

ok=$(grep -c '^OK '   "$MAIN_LOG" || true)
sk=$(grep -c '^SKIP ' "$MAIN_LOG" || true)
bad=$(grep -c '^FAIL\|^MISSING' "$MAIN_LOG" || true)
echo "[$(date)] DONE: $ok OK, $sk SKIP, $bad FAIL (see $FAIL_LOG)" | tee -a "$MAIN_LOG"

# Clean empty _work dir.
rmdir "$OUT/_work" 2>/dev/null || true
