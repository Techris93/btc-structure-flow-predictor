from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import os

from .indicators import atr
from .models import Zone
from .structure import confirmed_pivots, structure_events

_MAX_ZONE_CACHE_ENTRIES = 128


def _zone_id(kind, side, level, available_at, extra=""):
    raw=f"{kind}|{side}|{level:.8f}|{pd.Timestamp(available_at).isoformat()}|{extra}"
    return f"{kind}:{hashlib.sha1(raw.encode()).hexdigest()[:14]}"


def _candidate(kind, side, level, width, created, available, score, sources, expiry_bars):
    return {"zone_id":_zone_id(kind,side,level,available),"kind":kind,"side":side,"low":level-width,"high":level+width,
            "score":score,"created_at":created,"available_at":available,"sources":tuple(sources),"expiry_bars":expiry_bars}


def _period_levels(x, rule, kind, expiry_bars):
    shifted=x.copy(); shifted["period"]=(shifted.index-pd.Timedelta(nanoseconds=1)).tz_localize(None).to_period(rule)
    rows=[]
    for period, group in shifted.groupby("period"):
        available=period.end_time.tz_localize(x.index.tz)+pd.Timedelta(nanoseconds=1)
        if available>x.index[-1]: continue
        for label,column,side in (("high","high","above"),("low","low","below")):
            level=float(group[column].max() if label=="high" else group[column].min())
            rows.append((f"previous_{kind}_{label}",side,level,group.index[0],available,1.35,(f"previous_{kind}",),expiry_bars))
    return rows[-6:]


def _session_levels(x):
    y=x.copy(); opened=y.index-pd.Timedelta(minutes=15)
    hour=opened.hour
    session=np.select([hour<8,hour<16],["asia","london"],default="new_york")
    y["session"]=session; y["day"]=opened.floor("D")
    ends={"asia":8,"london":16,"new_york":24}; rows=[]
    for (day,name),group in y.groupby(["day","session"]):
        available=day+pd.Timedelta(hours=ends[name])
        if available>x.index[-1] or group.index[0]>available: continue
        for label,column,side in (("high","high","above"),("low","low","below")):
            level=float(group[column].max() if label=="high" else group[column].min())
            rows.append((f"{name}_{label}",side,level,group.index[0],available,1.2,("session_extreme",name),192))
    return rows[-18:]


def _profile_and_vwap(x, width):
    history=x.iloc[-193:-1] if len(x)>193 else x.iloc[:-1]
    if len(history)<40 or history.volume.sum()<=0: return []
    typical=(history.high+history.low+history.close)/3
    bucket_size=max(width*2,float(typical.iloc[-1])*.001)
    bucket=(typical/bucket_size).round()*bucket_size
    profile=history.groupby(bucket).volume.sum().sort_values()
    levels=[]; available=x.index[-1]; price=float(x.close.iloc[-1])
    if len(profile)>=3:
        hvn=float(profile.index[-1]); lvn=float(profile.index[0])
        for kind,level,score in (("volume_hvn",hvn,1.15),("volume_lvn",lvn,1.05)):
            levels.append((kind,"above" if level>price else "below",level,history.index[0],available,score,("volume_profile",),192))
    vwap=float((typical*history.volume).sum()/history.volume.sum())
    variance=float((((typical-vwap)**2)*history.volume).sum()/history.volume.sum()); deviation=variance**.5
    for name,level in (("anchored_vwap",vwap),("vwap_upper",vwap+deviation),("vwap_lower",vwap-deviation)):
        levels.append((name,"above" if level>price else "below",level,history.index[0],available,1.1,("vwap",),96))
    return levels


def build_projected_zones(ohlc: pd.DataFrame, lookback: int = 1000, pivot_left: int = 2, pivot_right: int = 2, expiry_bars: int = 120) -> list[Zone]:
    """Causal 15m zone book from swings, equal levels, periods, sessions, profile and VWAP."""
    if ohlc.empty: return []
    last=ohlc.iloc[-1]
    cache_key = (ohlc.index[0],ohlc.index[-1],len(ohlc),float(last.open),float(last.high),float(last.low),float(last.close),lookback,pivot_left,pivot_right,expiry_bars)
    if hasattr(build_projected_zones, "_cache") and cache_key in build_projected_zones._cache:
        return build_projected_zones._cache[cache_key]
    x=ohlc.tail(lookback).copy().sort_index(); a=atr(x).ffill(); piv=confirmed_pivots(x,pivot_left,pivot_right)
    if piv.empty:
        return []
    if x.empty: return []
    last_atr=max(float(a.iloc[-1]) if np.isfinite(a.iloc[-1]) else float(x.close.iloc[-1])*.005, float(x.close.iloc[-1])*.0003)
    base_width=max(last_atr*.12,float(x.close.iloc[-1])*.0005); raw=[]
    atr_by_time=dict(zip(x.index,a.to_numpy()))
    for p in piv.itertuples():
        atr_at=atr_by_time.get(p.available_at,np.nan)
        width=max((float(atr_at) if np.isfinite(atr_at) else last_atr)*.12,p.price*.0005)
        side="above" if p.kind=="high" else "below"
        # Only add swings that are not too close to each other
        raw.append(_candidate("swing",side,float(p.price),width,p.pivot_time,p.available_at,1.0,("confirmed_swing",),expiry_bars))
    ordered=piv.sort_values("available_at")
    for kind in ("high","low"):
        same=ordered[ordered.kind==kind]
        rows_list=list(same.itertuples())
        for p,q in zip(rows_list[:-1],rows_list[1:]):
            tolerance=round(max(base_width,float(q.price)*.00025),8)  # tighter: ~0.025% of price
            if abs(float(p.price)-float(q.price))<=tolerance:
                level=(float(p.price)+float(q.price))/2; side="above" if kind=="high" else "below"
                raw.append(_candidate(f"equal_{kind}s",side,level,tolerance,min(p.pivot_time,q.pivot_time),q.available_at,1.6,("equal_levels",p.swing_id,q.swing_id),192))
    for spec in _period_levels(x,"D","day",384)+_period_levels(x,"W","week",2688)+_session_levels(x)+_profile_and_vwap(x,base_width):
        kind,side,level,created,available,score,sources,expiry=spec
        raw.append(_candidate(kind,side,float(level),base_width,created,available,score,sources,expiry))
    events=structure_events(x)
    if not events.empty:
        for ts,event in events.tail(20).iterrows():
            side="below" if event.bias=="bullish" else "above"; level=float(event.level)
            raw.append(_candidate("untested_breakout",side,level,base_width,ts,ts,1.3,("structure_break",str(event.swing_id)),192))

    zones=[]; bar_lows=x.low.to_numpy(); bar_highs=x.high.to_numpy(); index_values=x.index
    for item in raw:
        available=pd.Timestamp(item.pop("available_at")).as_unit(x.index.unit,round_ok=True); pos=x.index.searchsorted(available,side="right")
        expiry_bars_for_zone=item.pop("expiry_bars"); expiry_pos=pos+expiry_bars_for_zone
        expires=index_values[expiry_pos] if expiry_pos<len(x) else None
        stop_pos=expiry_pos if expiry_pos<len(x) else len(x)
        seg_low=bar_lows[pos:stop_pos]; seg_high=bar_highs[pos:stop_pos]
        hits=(seg_low<item["low"]) if item["side"]=="below" else (seg_high>item["high"])
        first_hit=int(np.argmax(hits)) if hits.any() else None
        swept_at=index_values[pos+first_hit] if first_hit is not None else None
        end_pos=pos+first_hit if first_hit is not None else stop_pos
        touches=int(np.count_nonzero((bar_highs[pos:end_pos]>=item["low"])&(bar_lows[pos:end_pos]<=item["high"])))
        # Dynamic score: age decay + touch penalty + recency bonus
        age_bars = len(x) - pos
        age_decay = min(age_bars / 384, 0.25)  # cap at 0.25 penalty over ~4 days
        touch_penalty = min(touches, 4) * 0.15
        # Recency bonus for zones created recently
        recency_bonus = max(0, 0.1 - age_bars / 2000)
        item["score"] = item["score"] - touch_penalty - age_decay + recency_bonus
        zones.append(Zone(**item,available_at=available,expires_at=expires,swept_at=swept_at,invalidated_at=swept_at,touches=touches))
    # Limit zone explosion while preserving source-type diversity.
    # Period/session/profile/vwap zones are rare and informative, so keep them all.
    # Cap the noisy swing/equal-level groups.
    rare = [z for z in zones if z.kind not in ("swing", "equal_highs", "equal_lows")]
    swing_equal = [z for z in zones if z.kind in ("swing", "equal_highs", "equal_lows")]
    swing_equal = sorted(swing_equal, key=lambda z: (-z.score, pd.Timestamp(z.available_at)))
    max_swing_equal = int(os.getenv("MAX_SWING_EQUAL_ZONES", "80"))
    swing_equal = swing_equal[:max_swing_equal]
    final = rare + swing_equal
    # Hard overall cap as a safety valve
    final = sorted(final, key=lambda z: (-z.score, pd.Timestamp(z.available_at)))
    max_zones = int(os.getenv("MAX_PROJECTED_ZONES", "200"))
    final = final[:max_zones]
    unique={z.zone_id:z for z in final}
    res = sorted(unique.values(),key=lambda z:(pd.Timestamp(z.available_at),z.zone_id))
    if not hasattr(build_projected_zones, "_cache"): build_projected_zones._cache = {}
    if len(build_projected_zones._cache) >= _MAX_ZONE_CACHE_ENTRIES: build_projected_zones._cache.clear()
    build_projected_zones._cache[cache_key] = res
    return res


def mark_zone_state(zones: list[Zone], ohlc: pd.DataFrame) -> list[Zone]:
    for z in zones:
        future=ohlc.loc[ohlc.index>=pd.Timestamp(z.available_at)]
        for ts,b in future.iterrows():
            if (z.side=="below" and b.low<z.low) or (z.side=="above" and b.high>z.high): z.swept_at=ts; z.invalidated_at=ts; break
    return zones
