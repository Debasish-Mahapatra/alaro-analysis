#!/usr/bin/env bash

set -euo pipefail

DDH_ROOT=${DDH_ROOT:-/mnt/scratch/MANAUS/DDH}
TOOLBOX_ROOT=${TOOLBOX_ROOT:-$DDH_ROOT/ddhtoolbox}
SRC_ROOT=${SRC_ROOT:-$DDH_ROOT/untar-data}
OUT_ROOT=${OUT_ROOT:-$DDH_ROOT/alaro-24h-budgets}
INPUT_BASENAME=${INPUT_BASENAME:-DHFDLABOF+0024}
JOBS=${JOBS:-$(nproc)}

RUNTIME_ROOT="$OUT_ROOT/_runtime"
RUNTIME_BPS_ROOT="$RUNTIME_ROOT/ddh_budget_lists"
RUNTIME_ALARO_DIR="$RUNTIME_BPS_ROOT/alaro"
RUNTIME_CONV_LIST="$RUNTIME_ALARO_DIR/conversion_list"
LOG_ROOT="$OUT_ROOT/logs"
WORK_ROOT="$OUT_ROOT/_work"

EXPERIMENTS=(control graupel 2mom)
VARIABLES=(CT QG QI QL QR QS QV TKE TTE UU VV)

usage() {
  cat <<EOF
Usage: $(basename "$0") [--worker EXP DAY VAR]

Environment overrides:
  DDH_ROOT=/mnt/scratch/MANAUS/DDH
  TOOLBOX_ROOT=\$DDH_ROOT/ddhtoolbox
  SRC_ROOT=\$DDH_ROOT/untar-data
  OUT_ROOT=\$DDH_ROOT/alaro-24h-budgets
  INPUT_BASENAME=DHFDLABOF+0024
  JOBS=\$(nproc)
EOF
}

prepare_runtime_config() {
  mkdir -p "$RUNTIME_ALARO_DIR" "$LOG_ROOT" "$WORK_ROOT"

  cp "$TOOLBOX_ROOT/ddh_budget_lists/alaro/"{QG.fbl,QI.fbl,QL.fbl,QR.fbl,QS.fbl,QV.fbl,TKE.fbl,TTE.fbl,UU.fbl,VV.fbl} \
    "$RUNTIME_ALARO_DIR/"
  cp "$TOOLBOX_ROOT/ddh_budget_lists/alaro/CT.fbl-3ice" "$RUNTIME_ALARO_DIR/CT.fbl"

  sed -i '0,/^MAIN QI$/s//MAIN QN/;0,/^QI$/s//QN/' "$RUNTIME_ALARO_DIR/QI.fbl"

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
  ' "$TOOLBOX_ROOT/ddh_budget_lists/alaro/conversion_list" > "$RUNTIME_CONV_LIST"

  cat <<'EOF' >> "$RUNTIME_CONV_LIST"
BUIIVQNM         VQN1         VQN0         ICE WATER : MEAN TENDENCY                                   g/kg/day             86400000.000  0
EOF
}

task_output_dir() {
  local exp=$1
  local day=$2
  local var=$3
  printf '%s/results/%s/%s/%s\n' "$OUT_ROOT" "$exp" "$day" "$var"
}

run_one() {
  local exp=$1
  local day=$2
  local var=$3

  local src_day="$SRC_ROOT/$exp/output/$day"
  local input_path="$src_day/$INPUT_BASENAME"
  local out_dir
  out_dir=$(task_output_dir "$exp" "$day" "$var")
  local data_dir="$out_dir/data"
  local docs_dir="$out_dir/docs"
  local plot_dir="$out_dir/plot"
  local plot_path="$plot_dir/$var.svg"
  local log_path="$out_dir/run.log"
  local done_path="$out_dir/done.ok"
  local work_dir

  mkdir -p "$out_dir" "$data_dir" "$docs_dir" "$plot_dir"

  if [[ -f "$done_path" && -f "$plot_path" ]]; then
    printf 'SKIP %s %s %s\n' "$exp" "$day" "$var"
    return 0
  fi

  if [[ ! -f "$input_path" ]]; then
    printf 'MISSING %s\n' "$input_path" | tee "$log_path"
    return 1
  fi

  work_dir=$(mktemp -d "$WORK_ROOT/${exp}_${day}_${var}.XXXXXX")

  {
    printf 'Experiment: %s\n' "$exp"
    printf 'Day: %s\n' "$day"
    printf 'Variable: %s\n' "$var"
    printf 'Input: %s\n' "$input_path"
    printf 'Output: %s\n' "$out_dir"
    printf 'Workdir: %s\n\n' "$work_dir"

    export DDHTOOLBOX="$TOOLBOX_ROOT"
    export DDHI_LIST="$RUNTIME_CONV_LIST"
    export DDHB_BPS="$RUNTIME_BPS_ROOT"
    export DDH_PLOT=dd2gr
    export PATH="$TOOLBOX_ROOT/tools:$TOOLBOX_ROOT/tools/lfa:$TOOLBOX_ROOT/tools/.dd2gr/src:$PATH"

    (
      cd "$src_day"
      ddhb -v "alaro/$var" -i "$INPUT_BASENAME" -r "$work_dir" -o "$plot_path"
    )
  } >"$log_path" 2>&1

  cp "$work_dir"/budget.alaro/*.dta "$data_dir/"
  find "$work_dir" -maxdepth 1 -type f \( -name '*.doc' -o -name 'zddhb.*.graph.doc' \) -exec cp {} "$docs_dir/" \;

  touch "$done_path"
  rm -rf "$work_dir"
  printf 'DONE %s %s %s\n' "$exp" "$day" "$var"
}

run_all() {
  prepare_runtime_config

  local tasks_file="$LOG_ROOT/tasks.tsv"
  : > "$tasks_file"

  local exp
  local var
  local day_dir
  for exp in "${EXPERIMENTS[@]}"; do
    while IFS= read -r day_dir; do
      local day
      day=$(basename "$day_dir")
      for var in "${VARIABLES[@]}"; do
        printf '%s\t%s\t%s\n' "$exp" "$day" "$var" >> "$tasks_file"
      done
    done < <(find "$SRC_ROOT/$exp/output" -mindepth 1 -maxdepth 1 -type d -name 'DDH20*' | sort)
  done

  printf 'Prepared %s tasks in %s\n' "$(wc -l < "$tasks_file")" "$tasks_file"

  awk -F '\t' '{printf "%s\0%s\0%s\0", $1, $2, $3}' "$tasks_file" \
    | xargs -0 -n 3 -P "$JOBS" "$0" --worker
}

if [[ ${1:-} == "--worker" ]]; then
  if [[ $# -ne 4 ]]; then
    usage
    exit 1
  fi
  if [[ ! -f "$RUNTIME_CONV_LIST" || ! -f "$RUNTIME_ALARO_DIR/CT.fbl" ]]; then
    prepare_runtime_config
  fi
  run_one "$2" "$3" "$4"
  exit
fi

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

run_all
