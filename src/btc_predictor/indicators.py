from __future__ import annotations
import numpy as np
import pandas as pd

def atr(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = ohlc["close"].shift(1)
    tr = pd.concat([ohlc["high"] - ohlc["low"], (ohlc["high"] - prev).abs(), (ohlc["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().rename("atr")

def resample_ohlcv(ohlcv: pd.DataFrame, freq: str) -> pd.DataFrame:
    x = ohlcv.copy()
    x.index = pd.to_datetime(x.index, utc=True)
    return x.resample(freq, label="right", closed="right").agg({"open":"first", "high":"max", "low":"min", "close":"last", "volume":"sum"}).dropna()

def rolling_zscore(s: pd.Series, window: int = 100, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods or max(20, window // 5)
    mean, std = s.rolling(window, min_periods=min_periods).mean(), s.rolling(window, min_periods=min_periods).std(ddof=0)
    return (s - mean) / std.replace(0, np.nan)
