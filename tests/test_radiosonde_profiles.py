from __future__ import annotations

from datetime import timezone
from pathlib import Path

import numpy as np

from alaro_analysis.workflows.radiosonde_profiles import (
    _parse_sonde_time,
    dewpoint_c_from_specific_humidity,
    relative_humidity_percent,
)


def test_parse_sonde_time_uses_filename_as_utc():
    parsed = _parse_sonde_time(Path("maosondewnpnM1.b1.20140101.054700.cdf"))

    assert parsed.tzinfo == timezone.utc
    assert parsed.isoformat() == "2014-01-01T05:47:00+00:00"


def test_dewpoint_and_rh_from_specific_humidity_are_physical():
    temperature_k = np.asarray([300.0])
    pressure_pa = np.asarray([100000.0])
    specific_humidity = np.asarray([0.014])

    dewpoint_c = dewpoint_c_from_specific_humidity(specific_humidity, pressure_pa)
    rh = relative_humidity_percent(specific_humidity, temperature_k, pressure_pa)

    assert float(dewpoint_c[0]) == np.float64(dewpoint_c[0])
    assert 18.0 < float(dewpoint_c[0]) < 21.0
    assert 60.0 < float(rh[0]) < 80.0
