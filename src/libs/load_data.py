"""Project-wide candle loading entry point.

The historical implementation lives in :mod:`libs.data_loader`.  TrendFinder
imports candles through this module so callers have one stable, explicit API.
"""

from datetime import datetime

import pandas as pd

try:
    from .data_loader import DATA_ROOT, DataLoader, str2datetime_jst
except ImportError:  # pragma: no cover - direct script/PYTHONPATH compatibility
    from data_loader import DATA_ROOT, DataLoader, str2datetime_jst


def load_candles(
    symbol: str,
    timeframe: str,
    start_time: datetime,
    end_time: datetime,
    *,
    data_root: str = DATA_ROOT,
) -> pd.DataFrame:
    """Load candles in ``[start_time, end_time)`` with JST-aware timestamps."""

    loader = DataLoader(symbol, data_root=data_root)
    return loader.load_data(timeframe, start_time, end_time)


__all__ = ["DATA_ROOT", "DataLoader", "load_candles", "str2datetime_jst"]
