# Refactor verification

Baseline: commit `52c4fb5`. Implementation branch: `codex/streamline-analysis`.
Verification used the environment recorded in [environment.md](environment.md).

## Automated checks

- Baseline suite: 159 passed, one command-help subprocess failed with a native
  allocator error in the disdrometer import path.
- Refactored suite: **190 passed**. The formerly failing help command passes.
- Ruff correctness/import checks and formatting checks pass; `git diff --check`
  reports no whitespace errors.
- Every registered legacy workflow alias resolves to its owning module.
- The wheel builds and installs into a separate temporary directory.
- All **33 installed console commands** display help with EPyGrAM, faxarray,
  Cartopy, cmaps, SciPy, MetPy, and tobac imports deliberately unavailable.
  This checks help/import isolation; using those optional features still
  requires their dependencies.
- The installed wheel contains all **59 catalogued recipes** and their modules.

## Numerical and execution checks

A synthetic NetCDF case includes three experiments, three levels, 24 local-hour
bins, and a missing grid cell. The original temperature reducer and reader were
loaded from the baseline commit and compared with the shared reducer. For both
temperature and rain, means, contribution counts, file totals, and sample-file
selection match exactly.

The temperature and hydrometeor commands completed end to end on this case,
producing temperature, rain, and derived-updraft-flux figures. A generated
updraft figure was visually inspected. Unit tests additionally cover serial and
process execution, source replacement/addition, limited-run cache isolation,
atomic-write failure, offline temperature overlays, named-axis transposition,
DDH incomplete/malformed blocks, and independent rain/graupel worker settings.

The updraft/downdraft domain-mean correction intentionally differs from the
baseline: missing cells no longer contribute zero. Tests assert the corrected
physical values and distinguish them from valid inactive cells.

## Limits

The multi-year FA/NetCDF/DDH archives and publication recipes were not executed
or regenerated. Numerical equivalence of every historical product has not been
established. The environment here did not expose the default model archive
locations, so source-backed execution checks used temporary synthetic files.
Legacy study cache policies and offline-read limitations are documented in
[cache-and-science.md](cache-and-science.md) and [migration.md](migration.md).
