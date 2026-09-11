# alaro-analysis

Analysis of ALARO model output, observations, and DDH budgets. The package
provides reusable numerical routines, commands for recurring analyses, and a
catalog of study recipes. Python 3.11 or newer is required.

## Install

```bash
python -m pip install -e '.[dev]'
```

The base install supports converted NetCDF data and plotting. For FA reads,
activate an environment with a working EPyGrAM installation, then install
`.[fa]`. `.[full,dev]` adds the optional plotting, thermodynamics, and tracking
libraries; it does not provision the native EPyGrAM system libraries.
[The tested environment](docs/environment.md) records the versions used for
verification.

## Start with a command

```bash
alaro-temperature --help
alaro-hydrometeor --help
alaro-diagnostics --help
alaro-recipe list
```

Existing commands and `python -m alaro_analysis.workflows.<name>` entrypoints
remain available. Existing `scripts/*.py` paths are compatibility launchers.
Their implementations now live in `alaro_analysis/recipes/`, grouped by study.

For example, after configuring the source directories:

```bash
alaro-hydrometeor --variables RAIN SNOW --analysis-modes full
alaro-temperature --seasons wet dry
alaro-surface --variable SFX.RN
```

Inspect a study recipe before running it:

```bash
alaro-recipe list paper1
alaro-recipe show make_paper1_strong_updraft_rain
alaro-recipe run make_paper1_strong_updraft_rain -- --help
```

Some historical recipes have fixed study settings and no argument parser.
Their catalog records identify this; consult the recipe before execution.

## Configure paths and analyses

Copy [config/alaro.example.toml](config/alaro.example.toml) to `alaro.toml`, or
pass `--config /path/to/settings.toml`. Inspect the resolved arguments without
starting a data-processing job:

```bash
alaro-hydrometeor --config alaro.toml --show-config
```

The precedence is built-in defaults, `[paths]` relocation, shared `[defaults]`,
`[workflows.<name>]`, then command-line arguments. Unknown workflow keys are
errors. Shared defaults irrelevant to a particular command are ignored.
Recipe tables use the recipe's module filename as their workflow name.

The environment variables `ALARO_DATA_ROOT`, `ALARO_RUNS_ROOT`, and
`ALARO_WORKSPACE_ROOT` relocate default roots before modules load. Historical
HPC locations remain the defaults. Explicit CLI paths override them. Recipes
with fixed module settings use the environment roots; `[paths]` relocates
**parser defaults**, not arbitrary module constants.

Worker options retain their existing spelling (`--n-workers` or `--workers`);
the shared process helper caps requests at 32. Figures retain their existing
filenames and study layouts.

## Use the numerical API

```python
from pathlib import Path
from alaro_analysis.data.profile_products import compute_profile, load_profile

root = Path('/data/ALARO/control/masked-netcdf')
mean, counts, n_files, sample_file = compute_profile(
    root, 'TEMPERATURE', utc_offset_hours=-4,
)
# mean and counts: (level, 24 local hours).
# Each finite spatial-mean profile has equal weight, irrespective of cell count.

product = load_profile(
    Path('/data/cache/control_temperature.npz'), root, 'TEMPERATURE',
    utc_offset_hours=-4,
)
print(product.mean.shape, product.n_files)
```

`compute_profile` also supports `UPDRAFT_FLUX`, `UPDRAFT_EXTENT`, and
`UPDRAFT_INTENSITY`. The derived products resolve their omega and mesh-fraction
inputs through the same reader and reducer.

For FA conversion:

```bash
alaro-convert /data/untar-output /data/masked-netcdf \
    --vars H00100TEMPERATUR H00100PRESSURE \
    --mask-file /data/Radar_mask_latlon.nc
```

## Find the implementation

| Location | Responsibility |
| --- | --- |
| `common/` | Experiment identities, paths, settings, coordinates, process helpers |
| `data/` | Readers, source discovery, profile products, cache formats, exports |
| `analysis/` | Numerical reductions and physical diagnostics |
| `dsd/` | Surface, vertical-profile, and grid-cell DSD calculations |
| `plotting/` | Figure construction from computed products |
| `workflows/` | Argument parsing and execution stages for reusable commands |
| `ddh/` | DDH extraction, aggregation, layer metrics, and budget figures |
| `recipes/` | Named paper, supplementary, IOP, DSD, diagnostic, and maintenance runs |
| `scripts/` | Compatibility launchers for the recipes |

See [architecture and extension guidance](docs/architecture.md),
[cache and scientific conventions](docs/cache-and-science.md), and
[migration notes](docs/migration.md).

## Validate changes

```bash
python -m pytest -q
ruff check .
ruff format --check alaro_analysis scripts tests
```

Tests cover numerical reductions, masks, cache provenance, configuration,
DDH aggregation, recipe isolation, dependency direction, and command help.
They do not run the multi-year archives or regenerate publication figures.
