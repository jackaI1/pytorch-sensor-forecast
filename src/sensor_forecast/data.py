"""Download pinned public weather observations and reject unsafe time axes."""

import hashlib
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip"
SHA256 = "40cdd29c24918e855133dce280ff59779bb0b5852898e05bafcf422a4cc0a175"
MEMBER = "PRSA_data_2010.1.1-2014.12.31.csv"


def validate_hourly(frame: pd.DataFrame) -> dict:
    """Fail before making positional lags. Never sort, fill, or drop bad rows."""
    if not {"timestamp", "value"}.issubset(frame.columns) or len(frame) < 2:
        raise ValueError("Need timestamp,value columns and at least two hourly rows")
    stamps = pd.to_datetime(frame["timestamp"], errors="raise")
    if stamps.isna().any():
        raise ValueError("Missing timestamp")
    duplicates = int(stamps.duplicated().sum())
    if duplicates:
        raise ValueError(f"Duplicate timestamps: {duplicates}")
    if not stamps.is_monotonic_increasing:
        raise ValueError("Timestamps must be increasing")
    if not stamps.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Gap or non-hourly timestamp spacing")
    values = pd.to_numeric(frame["value"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Missing or non-finite values")
    return {
        "rows": len(frame),
        "start": str(stamps.iloc[0]),
        "end": str(stamps.iloc[-1]),
        "duplicates": 0,
        "gaps": 0,
        "missing_values": 0,
    }


def load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["timestamp", "value"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    validate_hourly(frame)
    frame["value"] = pd.to_numeric(frame["value"])
    return frame


def fetch_weather(destination: Path, cache: Path = Path("data/beijing.zip")) -> dict:
    """Check archive bytes even on a cache hit; extract only the named CSV in memory."""
    if cache.exists():
        payload = cache.read_bytes()
    else:
        with urlopen(URL, timeout=60) as response:
            payload = response.read(5_000_001)
        if len(payload) > 5_000_000:
            raise ValueError("Unexpectedly large dataset download")
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError("Dataset checksum mismatch; remove the cache and verify the source")
    with ZipFile(BytesIO(payload)) as archive:
        raw = pd.read_csv(archive.open(MEMBER))
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw[["year", "month", "day", "hour"]]),
            "value": raw["TEMP"],
        }
    )
    quality = validate_hourly(frame)
    # Keep a complete calendar year, without cherry-picking windows by outcome.
    frame = frame.loc[frame.timestamp.dt.year == 2014].reset_index(drop=True)
    quality["selected_2014"] = validate_hourly(frame)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return quality
