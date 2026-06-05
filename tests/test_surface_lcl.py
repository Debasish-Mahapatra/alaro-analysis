import numpy as np

from alaro_analysis.workflows.surface import (
    coerce_pressure_to_pa,
    coerce_specific_humidity,
    infer_lcl_start_height_m,
    is_pblh_variable,
)


def test_pblh_variable_detection_ignores_punctuation():
    assert is_pblh_variable("CLPMHAUT.MOD.XFU")
    assert is_pblh_variable("CLPMHAUT_MOD_XFU")
    assert not is_pblh_variable("SFX.RN")


def test_lcl_start_height_is_inferred_from_height_level_name():
    assert infer_lcl_start_height_m(("H00100TEMPERATUR",), None) == 100.0
    assert infer_lcl_start_height_m(("SURFTEMPERATURE",), None) == 0.0
    assert infer_lcl_start_height_m(("H00100TEMPERATUR",), 25.0) == 25.0


def test_pressure_coercion_handles_log_pa_and_hpa():
    log_pa = np.log(np.array([100000.0, 90000.0]))
    np.testing.assert_allclose(coerce_pressure_to_pa(log_pa), [100000.0, 90000.0])
    np.testing.assert_allclose(coerce_pressure_to_pa(np.array([1000.0, 950.0])), [100000.0, 95000.0])
    np.testing.assert_allclose(coerce_pressure_to_pa(np.array([100000.0])), [100000.0])


def test_specific_humidity_coercion_handles_g_per_kg():
    np.testing.assert_allclose(coerce_specific_humidity(np.array([16.0, 10.0])), [0.016, 0.010])
    np.testing.assert_allclose(coerce_specific_humidity(np.array([0.016])), [0.016])
    assert np.isnan(coerce_specific_humidity(np.array([-1.0]))[0])
