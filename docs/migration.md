# Migration notes

## Existing commands and scripts

Console command names, module command paths, and script filenames are retained.
The additional `alaro-recipe` command lists and runs packaged recipes. Old
`scripts/` files are launchers; edit the implementation under `recipes/` when
maintaining a study. Install the package before invoking a launcher from an
unrelated working directory; launchers no longer change `sys.path`.

Internal imports should use the new owning modules. Historical workflow helper
names remain available through lazy compatibility exports. Mutating an old
workflow's module globals is not a supported way to configure shared functions;
pass arguments, a configuration file, or an immutable specification.

## Configuration

Use `--show-config` to inspect the exact arguments. Workflow table typos now
fail instead of being ignored. CLI abbreviations and aliases follow argparse's
precedence. Known experiment aliases are normalized for parsers advertising
canonical experiment choices; fixed historical study identities such as
`NO3M` keep their filesystem meaning.

Environment roots apply before module loading. TOML `[paths]` relocates parser
defaults; study constants without CLI options use environment roots. No input
files or output directories are renamed as part of this refactor.

## Results and caches

Shared product loaders invalidate unsigned or incompatible caches and recompute
when invoked. The refactor does not launch those historical processing jobs or
rewrite existing archives. The updraft missing-cell correction intentionally
changes affected domain means; regenerate those figures from a validated
product before comparing them with old outputs.

Hydrometeor freezing overlays retain the existing offline temperature-cache
fallback and report that source freshness cannot be checked. Profile commands
now return a failure status if any requested profile job fails, after rendering
the available complete comparisons.

Direct cache-reading APIs remain available when the sources are offline, but
cannot establish source freshness. Recipe-specific legacy caches may still
need their existing `--recompute` / `--force` option; see the recipe settings.
DDH partial-data processing now requires `--allow-partial` for malformed blocks.

## Public Python examples

The former README example used an `ExperimentSet` class that did not exist.
Use the concrete `compute_profile` / `load_profile` functions shown in the README,
or the domain-specific readers and calculations described in the architecture
guide. The package root continues to export `PeriodSpec`, `SpatialWindow`, and
`VerticalAxis`.
