from __future__ import annotations
import pandas as pd
from .indicators import atr

def confirmed_pivots(ohlc: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    """Return pivots only at their confirmation timestamp (right bars later)."""
    x = ohlc.copy(); x["atr"] = atr(x)
    rows = []
    for i in range(left, len(x) - right):
        h, l = x.iloc[i]["high"], x.iloc[i]["low"]
        window = x.iloc[i-left:i+right+1]
        if h >= window["high"].max(): rows.append({"pivot_time": x.index[i], "available_at": x.index[i+right], "price": h, "kind": "high", "atr": x.iloc[i]["atr"]})
        if l <= window["low"].min(): rows.append({"pivot_time": x.index[i], "available_at": x.index[i+right], "price": l, "kind": "low", "atr": x.iloc[i]["atr"]})
    return pd.DataFrame(rows)

def structure_events(ohlc: pd.DataFrame, left: int = 2, right: int = 2, buffer_atr: float = .1) -> pd.DataFrame:
    piv = confirmed_pivots(ohlc, left, right)
    if piv.empty: return pd.DataFrame(columns=["bias", "event", "level"])
    rows, last_high, last_low, bias, p_idx = [], None, None, "neutral", 0
    atrs = atr(ohlc)
    piv = piv.sort_values("available_at")
    for ts, bar in ohlc.iterrows():
        while p_idx < len(piv) and piv.iloc[p_idx].available_at <= ts:
            p = piv.iloc[p_idx]
            if p.kind == "high": last_high = p
            if p.kind == "low": last_low = p
            p_idx += 1
        a = atrs.loc[ts]
        if pd.isna(a): continue
        if last_high is not None and bar.close > last_high.price + buffer_atr*a:
            event = "BOS" if bias == "bullish" else "CHoCH" if bias == "bearish" else "BOS"; bias = "bullish"
            rows.append({"timestamp":ts,"bias":bias,"event":event,"level":last_high.price})
        elif last_low is not None and bar.close < last_low.price - buffer_atr*a:
            event = "BOS" if bias == "bearish" else "CHoCH" if bias == "bullish" else "BOS"; bias = "bearish"
            rows.append({"timestamp":ts,"bias":bias,"event":event,"level":last_low.price})
    out = pd.DataFrame(rows)
    return out.set_index("timestamp") if not out.empty else out
