"""The scientific distinction between inactive and missing cells must survive refactors."""

import numpy as np

from alaro_analysis.analysis.updrafts import updraft_intensity, updraft_mass_flux
from alaro_analysis.common.constants import G


def test_missing_cells_do_not_dilute_domain_mean():
    omega = np.array([-G, -G, np.nan, -G])
    mesh = np.array([1.0, 0.0, 1.0, np.nan])
    flux = updraft_mass_flux(omega, mesh)
    np.testing.assert_allclose(flux, [1.0, 0.0, np.nan, np.nan])
    assert np.nanmean(flux) == 0.5
    np.testing.assert_allclose(updraft_intensity(omega, mesh), [G, 0.0, np.nan, np.nan])


def test_signed_omega_and_all_missing_domain():
    np.testing.assert_allclose(updraft_mass_flux([G, -G], [0.5, 0.5]), [-0.5, 0.5])
    assert np.isnan(updraft_mass_flux([np.nan], [np.nan])).all()


def test_extended_diagnostics_use_the_same_valid_domain(tmp_path):
    import xarray as xr

    from alaro_analysis.analysis.diagnostic_profiles import compute_downdraft_profiles
    from alaro_analysis.common.models import SpatialWindow

    names = {name: name for name in ("UD_OMEGA", "UD_MESH_FRAC", "DD_OMEGA", "DD_MESH_FRAC")}
    files = {}
    values = {
        "UD_OMEGA": [-2 * G, 0, np.nan],
        "DD_OMEGA": [G, 0, np.nan],
        "UD_MESH_FRAC": [1, 0, np.nan],
        "DD_MESH_FRAC": [1, 0, np.nan],
    }
    for name, row in values.items():
        path = tmp_path / name / "pf20140101" / "pfABOFABOF+0000.nc"
        path.parent.mkdir(parents=True)
        xr.Dataset(
            {name: (("time", "level", "y", "x"), np.array(row).reshape(1, 1, 1, 3))}
        ).to_netcdf(path)
        files[name] = path
    result = compute_downdraft_profiles(
        "control", tmp_path, [(20, files["UD_OMEGA"])], names, SpatialWindow(None, None, None, None)
    )
    assert result["DD_FLUX"][0, 20] == 0.5
    assert result["NET_CONVECTIVE_FLUX"][0, 20] == 0.5
    assert result["DD_TO_UD_FLUX_RATIO"][0, 20] == 0.5
