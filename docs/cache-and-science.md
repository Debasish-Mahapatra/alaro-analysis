# Cache and scientific conventions

## Cache policy

All package and recipe NPZ writers use the atomic writers in `data/cache.py`.
They write a temporary file in the target directory, flush it, then replace the
finished target. A failed write leaves the previous cache intact.

Shared diurnal/height products, DSD CFAD products, surface/radiation/extended
diagnostics, convective-feedback, microphysics-budget, updraft-hydrometeor,
and strong-updraft study profiles validate calculation settings and source
inventories. Old unsigned caches are recomputed by these product loaders.
Low-level legacy cache readers remain available for deliberate offline access.
Historical study-specific caches not using a validated product loader retain
their original read policy; atomic writing alone does not validate provenance.

A source inventory records resolved paths, sizes, and nanosecond modification
times. Additions, removals, and replacements change its fingerprint. This is
not a content checksum: externally changing bytes while preserving both size
and timestamp requires explicit recomputation. Inventories are captured once
per context; use stable input files for a run.

The `__signature__` key holds the numerical cache identity. Where supported,
`__metadata__` contains JSON with schema/calculation versions, settings and
source inventory. Readers return numerical arrays separately from metadata.
Counts retain their existing array keys; new DDH aggregates also store a
separate contribution count for each block. Schema version 2 is a format
version, not the software package version.

A limited profile run (`--max-days`) does not overwrite its full-period cache.
Other validated products include the requested population in their signature;
a cache produced by a limited run cannot pass validation for a complete run.
A full run may therefore need to recompute after an exploratory run.

## Numerical population

- Shared diurnal profiles have dimensions `(level, local_hour)`. Each finite
  spatial-mean profile contributes one sample to its hour and level. Cells are
  averaged within a file before profiles are averaged across files.
- Grid-cell CFADs pool cells; domain-profile CFADs pool domain means. They stay
  separate in `dsd/gridcell_cfad.py` and `dsd/profile_cfad.py`. Strong-convection
  thresholds select a different population again.
- Masked or non-finite cells are missing, not zero. For domain-mean updraft
  flux/intensity, a valid inactive cell contributes zero; an out-of-domain or
  missing input contributes NaN. Flux is `-omega * mesh_fraction / g` for active
  cells. Correcting the old missing-cell dilution changes affected results.
- NetCDF readers resolve requested variable names and named dimensions. A
  singleton variable fallback requires explicit opt-in. The shared shape
  alignment utility rejects mismatches unless its legacy truncation option is
  explicitly selected. Some older plotting adapters retain their prior
  truncation behavior; they are not a substitute for coordinate regridding.
- Lead-hour and accumulation rules remain analysis-specific. Instantaneous
  diurnal profiles exclude `+0024`; accumulated-rainfall workflows need the
  endpoints used in their deaccumulation. Do not unify those selections simply
  because their filenames look alike.
- Converted `GEOPOTENTIEL` heights used by the profile workflows are metres;
  raw FA/DDH quantities have their own readers and conversions.

## DDH incomplete data

Aggregation records seen/used days and issues in metadata, and uses per-block
contribution counts. Incomplete extraction markers exclude unfinished days.
Malformed or incompatible blocks fail by default. `--allow-partial` explicitly
permits skipping such blocks and records why they were skipped. Different
blocks can therefore have different populations; consult their counts before
interpreting closure differences.
