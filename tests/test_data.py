import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from sensor_forecast import data


def test_complete_hourly_data(hourly):
    assert data.validate_hourly(hourly)["rows"] == 240


@pytest.mark.parametrize("bad", ["duplicate", "gap", "unordered", "missing_time", "half_hour"])
def test_invalid_time_axes_rejected(hourly, bad):
    if bad == "duplicate":
        hourly.loc[1, "timestamp"] = hourly.loc[0, "timestamp"]
    elif bad == "gap":
        hourly = hourly.drop(1)
    elif bad == "unordered":
        hourly = hourly.iloc[::-1]
    elif bad == "missing_time":
        hourly.loc[1, "timestamp"] = pd.NaT
    else:
        hourly.loc[1, "timestamp"] += pd.Timedelta(minutes=30)
    with pytest.raises(ValueError):
        data.validate_hourly(hourly)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "unknown"])
def test_nonfinite_or_nonnumeric_values_rejected(hourly, value):
    hourly["value"] = hourly.value.astype(object)
    hourly.loc[3, "value"] = value
    with pytest.raises(ValueError):
        data.validate_hourly(hourly)


def test_bad_schema_or_too_few_rows(hourly):
    for frame in [hourly.iloc[:1], hourly.drop(columns="value")]:
        with pytest.raises(ValueError):
            data.validate_hourly(frame)


def test_csv_roundtrip_and_public_fixture(hourly, tmp_path):
    path = tmp_path / "hourly.csv"
    hourly.to_csv(path, index=False)
    np.testing.assert_allclose(data.load_csv(path).value, hourly.value)
    fixture = data.load_csv(Path(__file__).parent / "fixtures/weather.csv")
    assert data.validate_hourly(fixture)["rows"] == 240
    assert fixture.value.iloc[0] == 7  # Actual UCI TEMP on 2014-01-01 at midnight.


def test_checksum_rejects_corrupted_cache_without_network(tmp_path, monkeypatch):
    cache = tmp_path / "bad.zip"
    cache.write_bytes(b"corrupted")
    monkeypatch.setattr(data, "urlopen", lambda *a, **k: pytest.fail("Network was used"))
    with pytest.raises(ValueError, match="checksum"):
        data.fetch_weather(tmp_path / "weather.csv", cache)


def test_fetch_validates_and_selects_calendar_year(tmp_path, monkeypatch):
    content = "year,month,day,hour,TEMP\n2013,12,31,22,3\n2013,12,31,23,4\n"
    content += "2014,1,1,0,5\n2014,1,1,1,6\n"
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr(data.MEMBER, content)
    payload = stream.getvalue()
    monkeypatch.setattr(data, "SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(data, "urlopen", lambda *a, **k: BytesIO(payload))
    output, cache = tmp_path / "weather.csv", tmp_path / "cache.zip"
    quality = data.fetch_weather(output, cache)
    assert quality["rows"] == 4
    assert quality["selected_2014"]["rows"] == 2
    assert data.load_csv(output).value.tolist() == [5, 6]
    monkeypatch.setattr(data, "urlopen", lambda *a, **k: pytest.fail("Cache missed"))
    assert data.fetch_weather(output, cache) == quality
