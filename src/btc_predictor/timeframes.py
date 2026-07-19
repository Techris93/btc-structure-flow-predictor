from __future__ import annotations

import pandas as pd


def index_candles_by_close(frame: pd.DataFrame, interval: str | pd.Timedelta) -> pd.DataFrame:
    """Convert exchange open-time candles to the first timestamp they are knowable."""
    out = frame.copy()
    out.index = pd.to_datetime(out.index, utc=True) + pd.Timedelta(interval)
    out.index.name = "close_time"
    return out.sort_index()


def completed_timeframes(one_minute: pd.DataFrame, decision_time=None) -> dict[str, pd.DataFrame]:
    """Derive only fully closed 15m/1h/4h candles from close-indexed 1m bars."""
    x = one_minute.sort_index()
    if decision_time is not None:
        x = x.loc[x.index <= pd.Timestamp(decision_time)]
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for optional in ("taker_buy_volume", "trades"):
        if optional in x:
            agg[optional] = "sum"
    frames = {}
    for name, rule in (("15m", "15min"), ("1h", "1h"), ("4h", "4h")):
        y = x.resample(rule, label="right", closed="right", origin="epoch").agg(agg).dropna(subset=["open", "high", "low", "close"])
        frames[name] = y.loc[y.index <= x.index[-1]] if len(x) else y
    return frames
