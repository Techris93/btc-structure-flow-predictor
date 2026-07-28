from __future__ import annotations
import numpy as np
import pandas as pd
from .indicators import atr

_pivots_cache: dict[tuple, pd.DataFrame] = {}
_events_cache: dict[tuple, pd.DataFrame] = {}
_MAX_CACHE_ENTRIES = 256


def _frame_key(ohlc: pd.DataFrame) -> tuple:
    last=ohlc.iloc[-1]
    return (ohlc.index[0],ohlc.index[-1],len(ohlc),float(last.open),float(last.high),float(last.low),float(last.close))

def confirmed_pivots(ohlc: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    """Return pivots only at their confirmation timestamp (right bars later)."""
    if ohlc.empty:
        return pd.DataFrame(columns=["swing_id", "pivot_time", "available_at", "price", "kind", "atr"])
    cache_key = (*_frame_key(ohlc), left, right)
    if cache_key in _pivots_cache:
        return _pivots_cache[cache_key]
    n = len(ohlc)
    if n < left + right + 1:
        return pd.DataFrame(columns=["swing_id", "pivot_time", "available_at", "price", "kind", "atr"])
    high, low = ohlc["high"], ohlc["low"]
    window = left + right + 1
    roll_high = high.rolling(window, center=True).max().to_numpy()
    roll_low = low.rolling(window, center=True).min().to_numpy()
    highs, lows = high.to_numpy(), low.to_numpy()
    atr_values = atr(ohlc).to_numpy()
    index_values = ohlc.index.to_numpy()
    valid=np.arange(left,n-right,dtype=int)
    high_pos=valid[highs[valid]>=roll_high[valid]]
    low_pos=valid[lows[valid]<=roll_low[valid]]
    positions=np.concatenate((high_pos,low_pos))
    kinds=np.concatenate((np.repeat("high",len(high_pos)),np.repeat("low",len(low_pos))))
    if len(positions):
        order=np.argsort(positions,kind="stable"); positions=positions[order]; kinds=kinds[order]
        prices=np.where(kinds=="high",highs[positions],lows[positions])
        pivot_times=index_values[positions]
        res=pd.DataFrame({
            "swing_id":[f"{kind}:{pd.Timestamp(ts).isoformat()}:{price:.8f}" for kind,ts,price in zip(kinds,pivot_times,prices)],
            "pivot_time":pivot_times,
            "available_at":index_values[positions+right],
            "price":prices,
            "kind":kinds,
            "atr":atr_values[positions],
        })
    else:
        res=pd.DataFrame(columns=["swing_id","pivot_time","available_at","price","kind","atr"])
    if len(_pivots_cache) >= _MAX_CACHE_ENTRIES: _pivots_cache.clear()
    _pivots_cache[cache_key] = res
    return res

def structure_events(ohlc: pd.DataFrame, left: int = 2, right: int = 2, buffer_atr: float = .1) -> pd.DataFrame:
    if ohlc.empty: return pd.DataFrame(columns=["bias", "event", "level"])
    cache_key = (*_frame_key(ohlc), left, right, buffer_atr)
    if cache_key in _events_cache:
        return _events_cache[cache_key]
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
    res = out.set_index("timestamp") if not out.empty else out
    if len(_events_cache) >= _MAX_CACHE_ENTRIES: _events_cache.clear()
    _events_cache[cache_key] = res
    return res
