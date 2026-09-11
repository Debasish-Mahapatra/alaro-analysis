# Verification environment

The refactor was developed with Python 3.11.14 in the existing HPC `epygram`
environment. These versions were read from that environment rather than
inferred from package requirements:

| Package | Version |
| --- | --- |
| NumPy | 2.4.2 |
| xarray | 2026.1.0 |
| netCDF4 | 1.7.2 |
| Matplotlib | 3.10.8 |
| SciPy | 1.17.0 |
| Cartopy | 0.25.0 |
| cmaps | 2.0.1 |
| MetPy | 1.7.1 |
| pandas | 2.3.3 |
| faxarray | 0.2.4 |
| EPyGrAM | 2.0.7 |
| pytest | 9.0.2 |
| Ruff | 0.16.7 |

This is a tested version snapshot, not a portable native-library lockfile.
EPyGrAM/FA access additionally requires the site's native runtime. Base NetCDF
workflows can be installed without the FA or tracking extras. `.[tracking]`
installs pandas and tobac; the full extra includes all optional Python features.

To check a new environment, run the tests and command-help checks first, then
run a small source-data case for the features it will actually use. Multi-year
scientific output equivalence requires comparing the generated products and
counts on those archives; passing the unit suite alone does not establish it.
