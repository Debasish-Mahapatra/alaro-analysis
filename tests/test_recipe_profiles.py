import numpy as np
import xarray as xr

from alaro_analysis.common.study_specs import GRAUPEL_PROFILE, RAIN_PROFILE
from alaro_analysis.data.strong_updraft_profiles import process_day
from alaro_analysis.recipes.paper1.make_paper1_strong_updraft_rain import parse_args


def test_graupel_settings_do_not_mutate_rain_defaults():
    rain_before = parse_args([], spec=RAIN_PROFILE)
    graupel = parse_args([], spec=GRAUPEL_PROFILE)
    rain_after = parse_args([], spec=RAIN_PROFILE)
    assert rain_before.output_dir == rain_after.output_dir
    assert graupel.output_dir != rain_before.output_dir
    assert "graupel" in str(graupel.cache_dir)
    assert "rain" in str(rain_after.cache_dir)


def test_worker_receives_species_without_global_mutation(tmp_path):
    experiment = tmp_path / "control"
    for variable, value in {
        "UD_OMEGA": -20,
        "UD_MESH_FRAC": 0.5,
        "GEOPOTENTIEL": 1000,
        "TEMPERATURE": 280,
        "RAIN": 1e-4,
        "GRAUPEL": 2e-4,
    }.items():
        path = experiment / "masked-netcdf" / variable / "pf20140101" / "pfABOFABOF+0000.nc"
        path.parent.mkdir(parents=True)
        xr.Dataset(
            {variable: (("time", "level", "y", "x"), np.full((1, 1, 1, 1), value))}
        ).to_netcdf(path)
    day = experiment / "masked-netcdf" / "UD_OMEGA" / "pf20140101"
    task = (experiment, day, 0, 23, 10.0, 0.0)
    rain = process_day((*task, RAIN_PROFILE))
    graupel = process_day((*task, GRAUPEL_PROFILE))
    assert rain["n_files_used"] == graupel["n_files_used"] == 1
    assert np.sum(rain["rain_sum"]) == 1e-4
    assert np.sum(graupel["rain_sum"]) == 2e-4
