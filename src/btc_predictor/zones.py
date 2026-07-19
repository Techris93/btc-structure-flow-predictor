from __future__ import annotations
import pandas as pd
import numpy as np
from .indicators import atr
from .models import Zone
from .structure import confirmed_pivots

def build_projected_zones(ohlc: pd.DataFrame, lookback: int = 300, pivot_left: int = 2, pivot_right: int = 2, expiry_bars: int = 120) -> list[Zone]:
    """Build an as-of zone book; every state transition uses only later closed bars."""
    x = ohlc.tail(lookback).copy(); a = atr(x).ffill(); piv = confirmed_pivots(x, pivot_left, pivot_right); zones=[]
    for n, p in piv.iterrows():
        width = max(float(a.get(p.available_at, np.nan) or 0) * .12, p.price * .0005)
        if not np.isfinite(width): continue
        side = "above" if p.kind == "high" else "below"
        available_pos = x.index.get_indexer([p.available_at])[0]
        expiry_pos = min(available_pos + expiry_bars, len(x) - 1)
        expires_at = x.index[expiry_pos] if available_pos + expiry_bars < len(x) else None
        future = x.iloc[available_pos + 1:]
        if expires_at is not None:
            future = future.loc[future.index < expires_at]
        touched = ((future["high"] >= p.price-width) & (future["low"] <= p.price+width))
        swept = (future["low"] < p.price-width) if side == "below" else (future["high"] > p.price+width)
        swept_at = swept[swept].index[0] if swept.any() else None
        before_sweep = future if swept_at is None else future.loc[future.index < swept_at]
        touches = int(((before_sweep["high"] >= p.price-width) & (before_sweep["low"] <= p.price+width)).sum())
        score = 1.0 - min(touches, 4) * .15
        zones.append(Zone(p.swing_id, "swing", side, p.price-width, p.price+width, score, p.pivot_time, p.available_at,
                          expires_at=expires_at, swept_at=swept_at, invalidated_at=swept_at, sources=("confirmed_swing",), touches=touches))
    # Deliberately do not mutate/merge older zones when a later pivot arrives.
    return sorted(zones, key=lambda z: (pd.Timestamp(z.available_at), z.zone_id))

def mark_zone_state(zones: list[Zone], ohlc: pd.DataFrame) -> list[Zone]:
    for z in zones:
        future = ohlc.loc[ohlc.index >= pd.Timestamp(z.available_at, tz="UTC") if ohlc.index.tz is not None else ohlc.index >= z.available_at]
        for ts, b in future.iterrows():
            if (z.side == "below" and b.low < z.low) or (z.side == "above" and b.high > z.high): z.swept_at = ts; z.invalidated_at = ts; break
    return zones
