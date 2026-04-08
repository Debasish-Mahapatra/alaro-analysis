# alaro-analysis

Reusable analysis toolkit for ALARO NWP model outputs over the Amazon region, built on top of [faxarray](https://github.com/UMR-CNRM/faxarray).

## Installation

```bash
pip install -e .
```

Requires Python 3.11+ and the `epygram` conda environment for FA file access.

## Package structure

```
alaro_analysis/
    analysis/          # Composable analysis building blocks
        profiles.py    # Diurnal profile accumulation (vertical + surface)
        caching.py     # Cache-aware computation wrappers
        derived.py     # Thermodynamic fields (theta_e, MSE, RH, dp)
        experiment.py  # ExperimentSet: multi-experiment orchestration
    common/            # Shared utilities and constants
    converter/         # FA-to-NetCDF conversion pipeline
    data/              # File discovery, I/O, caching
    plotting/          # Scales, styles, reusable plot builders
    workflows/         # Complete analysis scripts (surface, hydrometeor, etc.)
examples/              # Top-level runner scripts
tests/                 # Unit tests
```

## Quick start

```python
from alaro_analysis.analysis import ExperimentSet, compute_surface_diurnal_cycle
from alaro_analysis.data.discovery import collect_file_records
from alaro_analysis.plotting.panels import plot_surface_diurnal_cycle

# Set up experiments
exps = ExperimentSet.from_three_dirs(
    control="/path/to/control/masked-netcdf-2",
    graupel="/path/to/graupel/masked-netcdf-2",
    twomom="/path/to/2mom/masked-netcdf-2",
)

# Compute diurnal cycle for any variable
var_maps = exps.discover_variable_maps()
line_data = {}
for exp in exps.experiments:
    var_name = exps.resolve_var_name(exp, ["CLPMHAUT.MOD.XFU"], variable_maps=var_maps)
    var_dir = exps.experiment_dirs[exp] / var_name
    records = collect_file_records(var_dir, max_days=None, allowed_months=None, utc_offset_hours=-4)
    mean, counts, used = compute_surface_diurnal_cycle(records, var_name)
    line_data[exp] = mean

# Plot
plot_surface_diurnal_cycle(
    line_data, output_file,
    variable_label="Boundary layer height", variable_unit="m",
    period_label="Full 2-year",
)
```

## FA-to-NetCDF conversion

From the command line:

```bash
python -m alaro_analysis.converter.cli \
    /path/to/untar-output /path/to/masked-netcdf \
    --vars "CLPMHAUT.MOD.XFU" \
    --mask-file /path/to/Radar_mask_latlon.nc \
    --workers 16
```

Or programmatically:

```python
from alaro_analysis.converter import run_conversion
from alaro_analysis.converter.models import RunConfig

cfg = RunConfig(
    input_root="/path/to/untar-output",
    output_root="/path/to/masked-netcdf",
    workers=16,
    mask_file="/path/to/Radar_mask_latlon.nc",
    # ... other options
)
summary = run_conversion(cfg, requested_vars=["CLPMHAUT.MOD.XFU"])
```

## Experiments

| Label | Name      | Description                    |
|-------|-----------|--------------------------------|
| C1M   | control   | 1-moment microphysics baseline |
| G1M   | graupel   | 1-moment with graupel          |
| G2M   | 2mom      | 2-moment microphysics          |

## Tests

```bash
pytest
```
