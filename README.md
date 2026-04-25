# alaro-analysis

Reusable analysis toolkit for ALARO NWP model outputs over the Amazon region.
FA conversion uses [faxarray](https://github.com/UMR-CNRM/faxarray) inside the
HPC `epygram` environment.

## Installation

For the HPC, install locally inside the conda environment. This does not
require `sudo` and the commands are available only while the environment is
active.

```bash
cd /mnt/HDS_CLIMATE/CLIMATE/deba/alaro-analysis
conda activate epygram
python -m pip install -e ".[full,dev]"
```

Requires Python 3.11+ and the `epygram` conda environment for FA file access.
The `full` extra installs the FA and plotting dependencies used on the HPC.

On macOS, do not install the FA extra because the `epygram` stack is not
available there. For local development or cached NetCDF analysis only, use:

```bash
python -m pip install -e ".[dev]"
```

This local install is useful for tests, config work, and workflows that read
already-converted NetCDF or `.npz` caches. FA conversion remains an HPC task.
Use an `alaro.toml` file to point local commands at local data paths.

## Command line workflows

The package installs reusable commands for the main workflows:

```bash
alaro-convert --help
alaro-surface --help
alaro-temperature --help
alaro-hydrometeor --help
alaro-diagnostics --help
alaro-radiation-compare --help
alaro-pair-analysis --help
alaro-panel-anomaly --help
```

`alaro-convert` requires the HPC FA stack. The other commands expect the
scientific dependencies needed by the workflow, such as `xarray`, `netCDF4`,
`matplotlib`, and for some plots `cmaps`.

Typical HPC usage keeps the built-in HPC paths and only passes the analysis
choices:

```bash
alaro-hydrometeor --variables RAIN SNOW GRAUPEL
alaro-temperature --seasons wet dry
alaro-surface --variable SFX.RN
```

The old module and example-script entrypoints still work, for example:

```bash
python -m alaro_analysis.workflows.hydrometeor --variables RAIN
python examples/plot_pblh_diurnal.py --seasons wet dry
```

## Optional config file

Workflow commands use this precedence:

1. command line arguments
2. values from `--config PATH`, or `./alaro.toml` when present
3. built-in HPC defaults

Config keys use the argparse destination names. Put shared values in
`[defaults]` and workflow-specific values in `[workflows.<name>]`.

```toml
[defaults]
utc_offset_hours = -4
n_workers = 16

[workflows.hydrometeor]
output_dir = "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures"
intermediate_dir = "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data"

[workflows.surface]
variable = "SFX.RN"
output_dir = "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/surface"
intermediate_dir = "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/surface"
```

## Quick start

```python
from alaro_analysis import ExperimentSet

exps = ExperimentSet.from_three_dirs(
    control="/path/to/control/masked-netcdf-2",
    graupel="/path/to/graupel/masked-netcdf-2",
    twomom="/path/to/2mom/masked-netcdf-2",
)

# Compute and plot in one call
exps.plot_surface_diurnal(
    "CLPMHAUT.MOD.XFU", "pblh_diurnal.png",
    label="Boundary layer height", unit="m",
)

# Or just get the data
data = exps.compute_surface_diurnal("CLPMHAUT.MOD.XFU")
# data["control"] -> ndarray of shape (24,)
```

## FA-to-NetCDF conversion

```python
exps = ExperimentSet.from_three_dirs(
    control="/path/to/control/masked-netcdf-2",
    graupel="/path/to/graupel/masked-netcdf-2",
    twomom="/path/to/2mom/masked-netcdf-2",
    fa_control="/path/to/control/untar-output",
    fa_graupel="/path/to/graupel/untar-output",
    fa_twomom="/path/to/2mom/untar-output",
)

exps.convert("CLPMHAUT.MOD.XFU", mask_file="/path/to/Radar_mask_latlon.nc")
```

Or from the command line:

```bash
python -m alaro_analysis.converter.cli \
    /path/to/untar-output /path/to/masked-netcdf \
    --vars "CLPMHAUT.MOD.XFU" \
    --mask-file /path/to/Radar_mask_latlon.nc
```

## Experiments

| Label   | Name     | Description                    |
|---------|----------|--------------------------------|
| C1M     | control  | 1-moment microphysics baseline |
| G1M     | graupel  | 1-moment with graupel          |
| G2M     | 2mom     | 2-moment microphysics          |
| G2M-XCU | 2mom-xcu | 2-moment with XCU (planned)    |

## Advanced usage

For low-level building blocks, import from submodules directly:

```python
from alaro_analysis.analysis.profiles import compute_diurnal_profile
from alaro_analysis.analysis.derived import compute_theta_e_field
from alaro_analysis.plotting.panels import plot_three_panel_diurnal
```

## Tests

```bash
pytest
```
