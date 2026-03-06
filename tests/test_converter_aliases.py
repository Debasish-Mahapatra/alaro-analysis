from alaro_analysis.converter.aliases import REQUESTED_VAR_FALLBACK_ALIASES, var_to_ds_name


def test_var_to_ds_name_replaces_dot():
    assert var_to_ds_name("HUMI.SPECIFI") == "HUMI_SPECIFI"


def test_converter_aliases_contains_expected_fallback():
    assert "KT273TEMPERATUR" in REQUESTED_VAR_FALLBACK_ALIASES
    assert "KT273TEMPERATURE" in REQUESTED_VAR_FALLBACK_ALIASES["KT273TEMPERATUR"]
