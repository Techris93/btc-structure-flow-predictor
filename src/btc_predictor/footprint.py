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

    # Simple absolute price-change / traded-volume heuristic. This is not
    # Kyle's Lambda, which requires a regression on signed order flow.
    price_diff = g.price.diff().abs()
    g["price_impact_ratio"] = price_diff / (g.volume + 1e-6)
    g["low_price_impact_score"] = (1.0 / (1.0 + 1e4 * g.price_impact_ratio)).clip(0.0, 1.0)

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
    """Weighted agreement, requiring real contributions from both venues."""
    if trades.empty or "exchange" not in trades: return 0.0, {}
    x=trades.copy(); x["time"]=pd.to_datetime(x.time,utc=True); x=x[(x.time>=pd.Timestamp(start))&(x.time<pd.Timestamp(end))]
    if x.empty: return 0.0, {}
    x["signed"]=np.where(x.side.str.lower().eq("buy"),x.price*x.qty,-x.price*x.qty)
    deltas=x.groupby("exchange").signed.sum().to_dict()
    actual={k:float(v) for k,v in deltas.items() if not str(k).endswith("_proxy") and abs(v)>=min_notional}
    if not {"binance","bybit"}.issubset(actual): return 0.0, actual
    desired=1 if direction=="bullish" else -1
    total=sum(abs(v) for v in actual.values())
    agree_notional=sum(abs(v) for v in actual.values() if np.sign(v)==desired)
    score=agree_notional/total if total else 0.0
    return score, actual


def _agreement_from_deltas(deltas, direction, min_notional=1.0):
    actual={str(k):float(v) for k,v in (deltas or {}).items() if not str(k).endswith("_proxy") and abs(float(v))>=min_notional}
    if not {"binance","bybit"}.issubset(actual):
        return 0.0,actual
    desired=1 if direction=="bullish" else -1
    total=sum(abs(value) for value in actual.values())
    agreed=sum(abs(value) for value in actual.values() if np.sign(value)==desired)
    return (agreed/total if total else 0.0),actual


def footprint_components_from_aggregates(footprint_bars, direction, sweep_time, decision_time):
    """Causal raw-footprint ratio and venue agreement from closed aggregates."""
    aggregated=footprint_bars.copy() if footprint_bars is not None else pd.DataFrame()
    if aggregated.empty:
        return {"ratio":1.0,"agreement":0.0,"deltas":{},"eligible":False,"exchanges":[]}
    aggregated["bar_close"]=pd.to_datetime(aggregated.bar_close,utc=True)
    sweep_open=pd.Timestamp(sweep_time)-pd.Timedelta("1min")
    aggregated=aggregated.loc[(aggregated.bar_close>sweep_open)&(aggregated.bar_close<=pd.Timestamp(decision_time))]
    if aggregated.empty:
        return {"ratio":1.0,"agreement":0.0,"deltas":{},"eligible":False,"exchanges":[]}
    agreement_window=aggregated.loc[aggregated.bar_close>pd.Timestamp(decision_time)-pd.Timedelta(minutes=5)]
    deltas=(agreement_window.assign(delta=agreement_window.buy-agreement_window.sell).groupby("exchange").delta.sum().to_dict()) if not agreement_window.empty else {}
    agreement,actual=_agreement_from_deltas(deltas,direction)
    footprint=aggregated.groupby("price_level")[["buy","sell"]].sum()
    ratio=float(((footprint.buy+1)/(footprint.sell+1)).median()) if len(footprint) else 1.0
    return {"ratio":ratio,"agreement":float(agreement),"deltas":actual,"eligible":{"binance","bybit"}.issubset(actual),"exchanges":sorted(str(value) for value in aggregated.exchange.dropna().unique())}


def _sweep_bar_trades(trades, sweep_time, decision_time, bar_freq="1min"):
    """Return trades from the confirming bar's open through its close.

    Candle timestamps are close times. Subtracting the bar duration makes the
    lower bound precede trades that occurred inside the sweep candle, while the
    strict decision-time upper bound preserves decision-time causality.
    """
    raw=trades.copy()
    if raw.empty:
        return raw
    raw["time"]=pd.to_datetime(raw.time,utc=True)
    bar_open=pd.Timestamp(sweep_time)-pd.Timedelta(bar_freq)
    return raw[(raw.time>=bar_open)&(raw.time<pd.Timestamp(decision_time))]


def footprint_confirmation(
    trades,
    flow_bars,
    direction,
    sweep_time,
    decision_time,
    window=100,
    min_score: float = 0.40,
    footprint_bars=None,
    price_bucket: float = 25.0,
    full_credit_ratio: float = 1.5,
    market_threshold: float = 0.40,
    raw_threshold: float = 0.40,
    gate_mode: str = "shadow",
):
    features=flow_features_from_bars(flow_bars,window) if flow_bars is not None and not flow_bars.empty else orderflow_features(trades,window=window)
    features=features.loc[features.index<=pd.Timestamp(decision_time)]
    input_exchanges=set()
    if flow_bars is not None and not flow_bars.empty:
        input_exchanges.add("binance")
    elif "exchange" in trades:
        input_exchanges.update(str(value) for value in trades.exchange.dropna().unique())
    if len(features)<20: return False,{"reason":"flow_warmup","bars":len(features),"score":0.0,"threshold":min_score,"contributing_exchanges":sorted(input_exchanges),"market_flow_score":0.0,"market_flow_threshold":market_threshold,"market_flow_confirmed":False,"raw_footprint_score":0.0,"raw_footprint_threshold":raw_threshold,"raw_footprint_confirmed":False,"raw_footprint_eligible":False,"flow_gate_mode":gate_mode}
    recent=features.loc[features.index>=pd.Timestamp(sweep_time)-pd.Timedelta(minutes=2)]
    if recent.empty:
        recent=features.tail(5)
    current=features.iloc[-1]
    recent_tail=recent.tail(5)

    low_impact_score = float(recent.low_price_impact_score.mean()) if not recent.empty else 0.5

    if direction=="bullish":
        extreme=float(((recent.delta_z<-1).sum() + recent.sell_absorption.sum()) / max(len(recent),1))
        has_reversal=recent_tail.bullish_delta_reversal.any() or current.delta>0 or recent_tail.delta.sum()>0
        reversal=1.0 if has_reversal else 0.0
    else:
        extreme=float(((recent.delta_z>1).sum() + recent.buy_absorption.sum()) / max(len(recent),1))
        has_reversal=recent_tail.bearish_delta_reversal.any() or current.delta<0 or recent_tail.delta.sum()<0
        reversal=1.0 if has_reversal else 0.0
    response_baseline=features.price_response.abs().rolling(20,min_periods=5).median().iloc[-1]
    stalled=float(min(1.0, (recent.price_response.abs()<=response_baseline).sum() / max(len(recent),1))) if np.isfinite(response_baseline) else 0.0
    raw=_sweep_bar_trades(trades,sweep_time,decision_time)
    aggregated = footprint_bars.copy() if footprint_bars is not None else pd.DataFrame()
    if not aggregated.empty:
        components=footprint_components_from_aggregates(aggregated,direction,sweep_time,decision_time)
        agreement,deltas=components["agreement"],components["deltas"]
    else:
        agreement,deltas=cross_exchange_agreement(
            trades,
            pd.Timestamp(decision_time)-pd.Timedelta(minutes=5),
            pd.Timestamp(decision_time),
            direction,
        )
    imbalance=0.0
    if not aggregated.empty:
        ratio=components["ratio"]
        footprint=pd.DataFrame({"imbalance_ratio":[ratio]})
    elif not raw.empty:
        footprint=build_footprint(raw,price_bucket=price_bucket)
    else:
        footprint=pd.DataFrame()
    if not footprint.empty:
        ratio=float(footprint.imbalance_ratio.median()) if len(footprint) else 1.0
        ratio_span=max(float(full_credit_ratio)-1.0,1e-6)
        if direction=="bullish":
            imbalance=min(1.0, max(0.0, (ratio-1.0)/ratio_span))
        else:
            imbalance=min(1.0, max(0.0, (1.0/ratio-1.0)/ratio_span))
    weights = {
        "extreme_delta": 0.25,
        "delta_reversal": 0.25,
        "price_response": 0.15,
        "low_price_impact": 0.15,
        "cross_exchange": 0.10,
        "footprint_imbalance": 0.10,
    }
    market_contribution = (
        weights["extreme_delta"] * extreme
        + weights["delta_reversal"] * reversal
        + weights["price_response"] * stalled
        + weights["low_price_impact"] * low_impact_score
    )
    market_flow_score=market_contribution/0.80
    raw_footprint_score=0.75*imbalance+0.25*agreement
    raw_footprint_eligible={"binance","bybit"}.issubset(deltas)
    market_flow_confirmed=market_flow_score>=market_threshold
    raw_footprint_confirmed=raw_footprint_eligible and raw_footprint_score>=raw_threshold
    score = market_contribution + weights["cross_exchange"]*agreement + weights["footprint_imbalance"]*imbalance
    confirmed = (market_flow_confirmed and raw_footprint_confirmed) if gate_mode=="calibrated" else score>=min_score
    reason = "confirmed" if confirmed else "score_below_threshold"
    contributing_exchanges=set(deltas)
    if "exchange" in raw:
        contributing_exchanges.update(str(value) for value in raw.exchange.dropna().unique())
    if flow_bars is not None and not flow_bars.empty:
        contributing_exchanges.add("binance")
    if not aggregated.empty:
        contributing_exchanges.update(str(value) for value in aggregated.exchange.dropna().unique())
    agreement_status="cross_exchange" if {"binance","bybit"}.issubset(deltas) else "single_source" if deltas else "no_exchange_data"
    return confirmed, {"reason": reason, "score": round(score, 3), "threshold": min_score, "bars": len(features),
                       "extreme": round(extreme,3), "reversal": round(reversal,3),
                       "stalled_response": round(stalled,3), "low_price_impact_score": round(low_impact_score, 3),
                       "agreement": round(agreement,3), "imbalance": round(imbalance,3),
                       "exchange_deltas": deltas, "agreement_status": agreement_status,
                       "contributing_exchanges": sorted(contributing_exchanges), "raw_sweep_trades": len(raw),
                       "market_flow_score": round(float(market_flow_score),3), "market_flow_threshold":market_threshold,
                       "market_flow_confirmed":bool(market_flow_confirmed),
                       "raw_footprint_score":round(float(raw_footprint_score),3), "raw_footprint_threshold":raw_threshold,
                       "raw_footprint_confirmed":bool(raw_footprint_confirmed), "raw_footprint_eligible":bool(raw_footprint_eligible),
                       "raw_footprint_ratio":round(float(ratio if not footprint.empty else 1.0),6),
                       "flow_gate_mode":gate_mode,
                       "delta_z": float(current.delta_z) if np.isfinite(current.delta_z) else None,
                       "intensity_z": float(current.intensity_z) if np.isfinite(current.intensity_z) else None}
