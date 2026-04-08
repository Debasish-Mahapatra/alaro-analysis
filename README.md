# alaro-analysis

Reusable analysis toolkit for ALARO NWP model outputs over the Amazon region, built on top of [faxarray](https://github.com/UMR-CNRM/faxarray).

## Installation

```bash
pip install -e .
```

Requires Python 3.11+ and the `epygram` conda environment for FA file access.

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
