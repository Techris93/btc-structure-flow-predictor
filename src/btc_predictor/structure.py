from __future__ import annotations
import numpy as np
import pandas as pd
from .indicators import atr

def confirmed_pivots(ohlc: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    """Return pivots only at their confirmation timestamp (right bars later)."""
    n = len(ohlc)
    if n < left + right + 1:
        return pd.DataFrame(columns=["swing_id", "pivot_time", "available_at", "price", "kind", "atr"])
    high, low = ohlc["high"], ohlc["low"]
    window = left + right + 1
    roll_high = high.rolling(window, center=True).max().to_numpy()
    roll_low = low.rolling(window, center=True).min().to_numpy()
    highs, lows = high.to_numpy(), low.to_numpy()
    atr_values = atr(ohlc).to_numpy()
    index = ohlc.index
    available = index.to_numpy()
    rows = []
    for i in range(left, n - right):
        ts = index[i]
        if highs[i] >= roll_high[i]:
            rows.append({"swing_id": f"high:{pd.Timestamp(ts).isoformat()}:{highs[i]:.8f}", "pivot_time": ts, "available_at": available[i + right], "price": highs[i], "kind": "high", "atr": atr_values[i]})
        if lows[i] <= roll_low[i]:
            rows.append({"swing_id": f"low:{pd.Timestamp(ts).isoformat()}:{lows[i]:.8f}", "pivot_time": ts, "available_at": available[i + right], "price": lows[i], "kind": "low", "atr": atr_values[i]})
    return pd.DataFrame(rows)

def structure_events(ohlc: pd.DataFrame, left: int = 2, right: int = 2, buffer_atr: float = .1) -> pd.DataFrame:
    piv = confirmed_pivots(ohlc, left, right)
    if piv.empty: return pd.DataFrame(columns=["bias", "event", "level"])
    piv = piv.sort_values("available_at")
    p_kind = piv["kind"].to_numpy()
    p_price = piv["price"].to_numpy(dtype=float)
    p_avail = piv["available_at"].to_numpy()
    p_sid = piv["swing_id"].to_numpy()
    closes = ohlc["close"].to_numpy(dtype=float)
    atrs = atr(ohlc).to_numpy()
    times = ohlc.index.to_numpy()
    rows, bias, p_idx = [], "neutral", 0
    last_high = last_low = -1
    high_consumed = low_consumed = False
    for pos in range(len(times)):
        ts = times[pos]
        while p_idx < len(piv) and p_avail[p_idx] <= ts:
            if p_kind[p_idx] == "high":
                if p_idx != last_high: last_high, high_consumed = p_idx, False
            else:
                if p_idx != last_low: last_low, low_consumed = p_idx, False
            p_idx += 1
        a = atrs[pos]
        if not np.isfinite(a): continue
        close = closes[pos]
        if last_high >= 0 and not high_consumed and close > p_price[last_high] + buffer_atr * a:
            event = "BOS" if bias == "bullish" else "CHoCH" if bias == "bearish" else "BOS"; bias = "bullish"
            high_consumed = True
            rows.append({"timestamp":times[pos],"bias":bias,"event":event,"level":p_price[last_high],"swing_id":p_sid[last_high]})
        elif last_low >= 0 and not low_consumed and close < p_price[last_low] - buffer_atr * a:
            event = "BOS" if bias == "bearish" else "CHoCH" if bias == "bullish" else "BOS"; bias = "bearish"
            low_consumed = True
            rows.append({"timestamp":times[pos],"bias":bias,"event":event,"level":p_price[last_low],"swing_id":p_sid[last_low]})
    out = pd.DataFrame(rows)
    return out.set_index("timestamp") if not out.empty else out
