from __future__ import annotations
import pandas as pd
import numpy as np
from .indicators import atr
from .models import Zone
from .structure import confirmed_pivots

def build_projected_zones(ohlc: pd.DataFrame, lookback: int = 300, pivot_left: int = 2, pivot_right: int = 2, expiry_bars: int = 120) -> list[Zone]:
    x = ohlc.tail(lookback).copy(); a = atr(x).ffill(); piv = confirmed_pivots(x, pivot_left, pivot_right); zones=[]
    for n, p in piv.iterrows():
        width = max(float(a.get(p.available_at, np.nan) or 0) * .12, p.price * .0005)
        if not np.isfinite(width): continue
        side = "above" if p.kind == "high" else "below"
        touches = int(((x.loc[:p.available_at, "high"] if p.kind == "high" else x.loc[:p.available_at, "low"]) - p.price).abs().le(width).sum())
        score = 1.0 + min(touches, 4) * .25
        zones.append(Zone(f"{p.kind}-{n}", "swing", side, p.price-width, p.price+width, score, p.pivot_time, p.available_at, sources=("confirmed_swing",), touches=touches))
    # Merge nearby levels while preserving earliest availability.
    zones.sort(key=lambda z: z.midpoint); merged=[]
    for z in zones:
        if merged and z.side == merged[-1].side and z.low <= merged[-1].high:
            q=merged[-1]; q.low=min(q.low,z.low); q.high=max(q.high,z.high); q.score=max(q.score,z.score)+.25; q.sources=tuple(set(q.sources+z.sources)); q.touches+=z.touches
        else: merged.append(z)
    return merged

def mark_zone_state(zones: list[Zone], ohlc: pd.DataFrame) -> list[Zone]:
    for z in zones:
        future = ohlc.loc[ohlc.index >= pd.Timestamp(z.available_at, tz="UTC") if ohlc.index.tz is not None else ohlc.index >= z.available_at]
        for ts, b in future.iterrows():
            if (z.side == "below" and b.low < z.low) or (z.side == "above" and b.high > z.high): z.swept_at = ts; break
    return zones
