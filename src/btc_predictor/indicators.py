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


def ema(series: pd.Series, period: int = 20) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def adx(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average Directional Index (ADX) to quantify trend strength."""
    if len(ohlc) < period + 1:
        return pd.Series(np.nan, index=ohlc.index, name="adx")
    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]
    prev_close = close.shift(1)
    
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_smooth = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm_series = pd.Series(plus_dm, index=ohlc.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    minus_dm_series = pd.Series(minus_dm, index=ohlc.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    
    plus_di = 100.0 * (plus_dm_series / atr_smooth.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm_series / atr_smooth.replace(0, np.nan))
    
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().rename("adx")

