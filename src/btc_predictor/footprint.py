from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import rolling_zscore


def build_footprint(trades: pd.DataFrame, bar_freq="1min", price_bucket=25.0) -> pd.DataFrame:
    x=trades.copy(); x["time"]=pd.to_datetime(x["time"],utc=True); x["notional"]=x.price*x.qty
    x["bar"]=x.time.dt.ceil(bar_freq); x["price_level"]=(x.price/price_bucket).round()*price_bucket
    f=x.groupby(["bar","price_level","side"])["notional"].sum().unstack("side",fill_value=0)
    for column in ("buy","sell"):
        if column not in f: f[column]=0.0
    f["delta"]=f.buy-f.sell; f["total"]=f.buy+f.sell
    f["imbalance_ratio"]=(f.buy+1)/(f.sell+1)
    return f.sort_index()


def _finish_features(g, window):
    g=g.sort_index().fillna({"buy":0,"sell":0,"volume":0,"trades":0})
    g["delta"]=g.buy-g.sell; g["cvd"]=g.delta.cumsum()
    g["delta_z"]=rolling_zscore(g.delta,window,min_periods=min(20,window))
    g["intensity_z"]=rolling_zscore(g.volume,window,min_periods=min(20,window))
    g["price_response"]=g.price.diff()/g.volume.replace(0,np.nan)
    threshold=g.price.rolling(10,min_periods=3).std().fillna(0)*.25
    g["bullish_delta_reversal"]=(g.delta.shift(1)<0)&(g.delta>0)
    g["bearish_delta_reversal"]=(g.delta.shift(1)>0)&(g.delta<0)
    g["delta_reversal"]=g.bullish_delta_reversal
    g["sell_absorption"]=(g.delta_z<-1)&(g.price.diff().abs()<=threshold)
    g["buy_absorption"]=(g.delta_z>1)&(g.price.diff().abs()<=threshold)
    return g


def orderflow_features(trades: pd.DataFrame, freq="1min", window=100) -> pd.DataFrame:
    x=trades.copy(); x["time"]=pd.to_datetime(x["time"],utc=True); x["notional"]=x.price*x.qty
    x["signed"]=np.where(x.side.str.lower().eq("buy"),x.notional,-x.notional)
    g=x.set_index("time").resample(freq,label="right",closed="right").agg(
        price=("price","last"),buy=("signed",lambda s:s[s>0].sum()),sell=("signed",lambda s:-s[s<0].sum()),volume=("notional","sum"),trades=("price","size"))
    return _finish_features(g,window)


def flow_features_from_bars(flow_bars: pd.DataFrame, window=100) -> pd.DataFrame:
    x=flow_bars.copy(); x.index=pd.to_datetime(x.index,utc=True)
    x["buy"]=x["taker_buy_volume"]*x["close"]
    x["sell"]=(x["volume"]-x["taker_buy_volume"]).clip(lower=0)*x["close"]
    x["price"]=x["close"]; x["volume"]=(x.buy+x.sell); x["trades"]=x.get("trades",0)
    return _finish_features(x[["price","buy","sell","volume","trades"]],window)



def cross_exchange_agreement(trades: pd.DataFrame, start, end, direction: str, min_notional: float = 1.0):
    """Weighted agreement. Returns a score in [0,1] and a veto flag."""
    if trades.empty or "exchange" not in trades: return 0.0, {}
    x=trades.copy(); x["time"]=pd.to_datetime(x.time,utc=True); x=x[(x.time>=pd.Timestamp(start))&(x.time<pd.Timestamp(end))]
    if x.empty: return 0.0, {}
    x["signed"]=np.where(x.side.str.lower().eq("buy"),x.price*x.qty,-x.price*x.qty)
    deltas=x.groupby("exchange").signed.sum().to_dict()
    actual={k:float(v) for k,v in deltas.items() if not str(k).endswith("_proxy") and abs(v)>=min_notional}
    if not actual: return 0.0, {}
    desired=1 if direction=="bullish" else -1
    total=sum(abs(v) for v in actual.values())
    agree_notional=sum(abs(v) for v in actual.values() if np.sign(v)==desired)
    score=agree_notional/total if total else 0.0
    return score, actual


def footprint_confirmation(trades, flow_bars, direction, sweep_time, decision_time, window=100, min_score: float = 0.45):
    features=flow_features_from_bars(flow_bars,window) if flow_bars is not None and not flow_bars.empty else orderflow_features(trades,window=window)
    features=features.loc[features.index<=pd.Timestamp(decision_time)]
    if len(features)<20: return False,{"reason":"flow_warmup","bars":len(features),"score":0.0}
    recent=features.loc[features.index>=pd.Timestamp(sweep_time)-pd.Timedelta(minutes=2)]
    current=features.iloc[-1]
    if direction=="bullish":
        extreme=float(((recent.delta_z<-1).sum() + recent.sell_absorption.sum()) / max(len(recent),1))
        reversal=float((current.bullish_delta_reversal or current.delta>0))
    else:
        extreme=float(((recent.delta_z>1).sum() + recent.buy_absorption.sum()) / max(len(recent),1))
        reversal=float((current.bearish_delta_reversal or current.delta<0))
    response_baseline=features.price_response.abs().rolling(20,min_periods=5).median().iloc[-1]
    stalled=float(min(1.0, (recent.price_response.abs()<=response_baseline).sum() / max(len(recent),1))) if np.isfinite(response_baseline) else 0.0
    agreement,deltas=cross_exchange_agreement(trades,pd.Timestamp(decision_time)-pd.Timedelta(minutes=2),decision_time,direction)
    raw=trades.copy(); raw["time"]=pd.to_datetime(raw.time,utc=True)
    raw=raw[(raw.time>=pd.Timestamp(sweep_time))&(raw.time<pd.Timestamp(decision_time))]
    imbalance=0.0
    if not raw.empty:
        footprint=build_footprint(raw)
        ratio=float(footprint.imbalance_ratio.median()) if len(footprint) else 1.0
        if direction=="bullish":
            imbalance=min(1.0, max(0.0, (ratio-1.0)/0.5))
        else:
            imbalance=min(1.0, max(0.0, (1.0/ratio-1.0)/0.5))
    # Weighted score. Weights sum to 1.0.
    weights = {
        "extreme_delta": 0.30,
        "delta_reversal": 0.30,
        "price_response": 0.15,
        "cross_exchange": 0.10,
        "footprint_imbalance": 0.15,
    }
    score = (
        weights["extreme_delta"] * extreme
        + weights["delta_reversal"] * reversal
        + weights["price_response"] * stalled
        + weights["cross_exchange"] * agreement
        + weights["footprint_imbalance"] * imbalance
    )
    confirmed = score >= min_score
    reason = "confirmed" if confirmed else "score_below_threshold"
    return confirmed, {"reason": reason, "score": round(score, 3), "threshold": min_score,
                       "extreme": round(extreme,3), "reversal": round(reversal,3),
                       "stalled_response": round(stalled,3), "agreement": round(agreement,3),
                       "imbalance": round(imbalance,3), "exchange_deltas": deltas,
                       "delta_z": float(current.delta_z) if np.isfinite(current.delta_z) else None,
                       "intensity_z": float(current.intensity_z) if np.isfinite(current.intensity_z) else None}
