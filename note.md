# Server Run Notes: FA -> NetCDF Conversion

## Environment (kili server)

```bash
source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
conda activate epygram
conda --version
```

Expected from your setup:
- `conda 23.11.0`
- env: `epygram`

## Script

- `convert_run_fa_to_netcdf.py`

This script converts only the variables that are **not commented** in the variable blocks at the top of the script.

## Currently active (uncommented) variables

- `DD_OMEGA`
- `DD_MESH_FRAC`
- `CV_PREC_FLUX`
- `ST_PREC_FLUX`
- `PRESSURE`
- `NC_LIQUID_WA`
- `KT273GRAUPEL`
- `KT273RAIN`
- `KT273SNOW`
- `KT273DD_OMEGA`
- `KT273DD_MESH_FRA`
- `KT273HUMI.SPECIF`
- `KT273TEMPERATUR`

No pressure-level or surface vars are currently active.

## Missing variable notes

The converter now tries robust alias resolution for FA naming variants, including:
- `_` vs `.` substitutions
- known fallbacks (for example `KT273TEMPERATUR` -> `KT273TEMPERATURE`)
- one-character truncation/extension prefix matching

If a variable is still reported as `Missing variables (skipped)`, it is likely not present in that experiment/file set.

## Basic command (server)

```bash
python convert_run_fa_to_netcdf.py <input_root> <output_root>
```

Example shape of paths:
- `<input_root>` contains day folders: `pfYYYYMMDD/pfABOFABOF+HHHH`
- outputs written to: `<output_root>/<VAR>/pfYYYYMMDD/pfABOFABOF+HHHH.nc`

## Recommended command (explicit flags)

```bash
python convert_run_fa_to_netcdf.py \
  <input_root> \
  <output_root> \
  --workers 16 \
  --include-init \
  --compress zlib \
  --level 1 \
  --overwrite \
  --skip-incomplete-days
```

## Your server command pattern (same bbox + mask + date range)

`pfYYYYMMDD` requires 8 digits.  
Interpreted range here as `pf20140101` to `pf20160101` (`--end-date 20160101`).

```bash
source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
conda activate epygram

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=/mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/envs/epygram/bin/python
SCRIPT=/gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/python_scripts/convert_run_fa_to_netcdf.py
MASK=/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/mask/Radar_mask_latlon.nc
ROOT=/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS

for EXP in control 2mom graupel; do
  IN="$ROOT/$EXP/untar-output"
  OUT="$ROOT/$EXP/masked-netcdf"
  mkdir -p "$OUT"
  echo "[$(date)] START $EXP full-convert"
  "$PY" "$SCRIPT" "$IN" "$OUT" \
    --workers 16 \
    --start-date 20140101 --end-date 20160101 \
    --include-init \
    --bbox-west -67 --bbox-east -53 --bbox-south -10 --bbox-north 4 \
    --compress zlib --level 1 \
    --mask-file "$MASK" \
    --overwrite \
    > "$OUT/convert_20140101_20160101.log" 2>&1
  echo "[$(date)] DONE  $EXP full-convert"
done

echo "[$(date)] ALL DONE full-convert"
```

## Precise `nohup` run style (as you use on server)

Recommended: force unbuffered stdout/stderr so progress lines appear during execution.

```bash
nohup env PYTHONUNBUFFERED=1 python3 -u convert_run_fa_to_netcdf.py \
  /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/untar-output \
  /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/masked-netcdf \
  --start-date 20140101 --end-date 20160101 \
  --workers 16 \
  --bbox-west -67 --bbox-east -53 --bbox-south -10 --bbox-north 4 \
  --compress zlib --level 1 \
  --mask-file /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/mask/Radar_mask_latlon.nc \
  --overwrite \
  > /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/masked-netcdf/full_conversion.log 2>&1 &
```

If you want the exact old style (without `-u`), it still works, but log updates may be delayed due to buffering:

```bash
nohup python3 convert_run_fa_to_netcdf.py \
  /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/untar-output \
  /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/masked-netcdf \
  --start-date 20140101 --end-date 20160101 \
  --workers 16 \
  --bbox-west -67 --bbox-east -53 --bbox-south -10 --bbox-north 4 \
  --compress zlib --level 1 \
  --mask-file /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/mask/Radar_mask_latlon.nc \
  --overwrite \
  > /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/masked-netcdf/full_conversion.log 2>&1 &
```

Watch live progress:

```bash
tail -f /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/masked-netcdf/full_conversion.log
```

## Stop old-name jobs on server

List old script processes:

```bash
ps aux | grep '[c]onvert_control_run_fa_to_netcdf.py'
```

Graceful stop (SIGTERM):

```bash
pkill -f 'convert_control_run_fa_to_netcdf.py'
```

If still running after ~10 seconds, force stop:

```bash
pkill -9 -f 'convert_control_run_fa_to_netcdf.py'
```

Verify nothing remains:

```bash
ps aux | grep '[c]onvert_control_run_fa_to_netcdf.py'
```

Background launch form:

```bash
(
  source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
  conda activate epygram
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  PY=/mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/envs/epygram/bin/python
  SCRIPT=/gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/python_scripts/convert_run_fa_to_netcdf.py
  MASK=/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/mask/Radar_mask_latlon.nc
  ROOT=/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS
  for EXP in control 2mom graupel; do
    IN="$ROOT/$EXP/untar-output"
    OUT="$ROOT/$EXP/masked-netcdf"
    mkdir -p "$OUT"
    "$PY" "$SCRIPT" "$IN" "$OUT" \
      --workers 16 \
      --start-date 20140101 --end-date 20160101 \
      --include-init \
      --bbox-west -67 --bbox-east -53 --bbox-south -10 --bbox-north 4 \
      --compress zlib --level 1 \
      --mask-file "$MASK" \
      --overwrite \
      > "$OUT/convert_20140101_20160101.log" 2>&1
  done
) > /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/convert_all_20140101_20160101.log 2>&1 &
```

If you only want `control`:

```bash
EXP=control
IN="/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/$EXP/untar-output"
OUT="/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/$EXP/masked-netcdf"
/mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/envs/epygram/bin/python \
  /gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/python_scripts/convert_run_fa_to_netcdf.py \
  "$IN" "$OUT" \
  --workers 16 \
  --start-date 20140101 --end-date 20160101 \
  --include-init \
  --bbox-west -67 --bbox-east -53 --bbox-south -10 --bbox-north 4 \
  --compress zlib --level 1 \
  --mask-file /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/mask/Radar_mask_latlon.nc \
  --overwrite
```

## Useful optional flags

- Date window:
```bash
--start-date YYYYMMDD --end-date YYYYMMDD
```

- If you want to skip `+0000`:
```bash
--exclude-init
```

- Optional spatial mask:
```bash
--mask-file /path/to/mask.nc --mask-var <var_name> --mask-threshold 0.5
```

- ROI bbox (defaults are Amazon subset):
```bash
--bbox-west -67 --bbox-east -53 --bbox-south -10 --bbox-north 4
```

## What info is needed before you run

1. `input_root` absolute path.
2. `output_root` absolute path.
3. Date range (`start/end`) or full archive.
4. Keep `+0000` or exclude it.
5. Keep default bbox or use a new bbox.
6. Whether to apply external mask (`mask-file`, `mask-var`).
7. Worker count appropriate for server load.

## Run checks after completion

- Summary file: `<output_root>/summary.json`
- Skipped days log: `<output_root>/skipped_days.log`
- Failures log: `<output_root>/failures.log`
