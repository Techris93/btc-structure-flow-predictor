from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .footprint import footprint_components_from_aggregates
from .persistence import JsonStore
from .research import proxy_trades
from .strategy import Predictor
from .timeframes import completed_timeframes


PRICE_BUCKETS = (10.0, 25.0, 50.0, 100.0)
FULL_CREDIT_RATIOS = (1.25, 1.5, 2.0, 3.0)
SCORE_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.30, 0.701, 0.05))


@dataclass(frozen=True)
class CalibrationConfig:
    price_bucket: float
    full_credit_ratio: float
    market_threshold: float
    raw_threshold: float


def calibration_grid():
    return [CalibrationConfig(*values) for values in itertools.product(PRICE_BUCKETS, FULL_CREDIT_RATIOS, SCORE_THRESHOLDS, SCORE_THRESHOLDS)]


def walk_forward_periods(start):
    start = pd.Timestamp(start)
    return [
        {"train_start":start,"train_end":start+pd.Timedelta(days=train_days),"validation_start":start+pd.Timedelta(days=train_days),"validation_end":start+pd.Timedelta(days=test_end)}
        for train_days,test_end in ((20,25),(25,30),(30,35),(35,40))
    ]


def _revision():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception:
        return "unknown"


def load_ohlcv(path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["close_time"]).set_index("close_time").sort_index()
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


def aggregate_trade_partitions(root, days, price_bucket, batch_size=250_000) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Calibration requires the research dependency pyarrow") from exc
    output=[]
    for day in days:
        daily=[]
        for exchange in ("binance","bybit"):
            path=Path(root)/exchange/f"date={day}"/"trades.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            parquet=pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=batch_size,columns=["time","price","qty","side","exchange"]):
                frame=batch.to_pandas()
                frame["time"]=pd.to_datetime(frame.time,utc=True)
                frame["bar_close"]=frame.time.dt.ceil("1min")
                frame["notional"]=pd.to_numeric(frame.price)*pd.to_numeric(frame.qty)
                frame["price_level"]=(pd.to_numeric(frame.price)/price_bucket).round()*price_bucket
                frame["buy"]=np.where(frame.side.astype(str).str.lower().eq("buy"),frame.notional,0.0)
                frame["sell"]=np.where(frame.side.astype(str).str.lower().eq("sell"),frame.notional,0.0)
                daily.append(frame.groupby(["bar_close","exchange","price_level"],as_index=False)[["buy","sell"]].sum())
        if daily:
            output.append(pd.concat(daily,ignore_index=True).groupby(["bar_close","exchange","price_level"],as_index=False)[["buy","sell"]].sum())
    if not output:
        return pd.DataFrame(columns=["bar_close","exchange","price_level","buy","sell"])
    return pd.concat(output,ignore_index=True).sort_values(["bar_close","exchange","price_level"])


def _enumerate_signal_range(candles, start, end, evaluation_start, evaluation_end) -> pd.DataFrame:
    """Enumerate one independent decision range with the global causal warmup."""
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    bars=candles.loc[(candles.index>=start-pd.Timedelta(days=7))&(candles.index<end)].copy()
    trades=proxy_trades(bars)
    trade_times=pd.to_datetime(trades.time,utc=True).reset_index(drop=True)
    frames_all=completed_timeframes(bars)
    predictor=Predictor(legacy_orderflow_threshold=0.0,flow_gate_mode="shadow",cache_closed_frames=True)
    events=[]; seen=set()
    first=max(80,int(bars.index.searchsorted(pd.Timestamp(evaluation_start),side="left")))
    last=min(len(bars)-1,int(bars.index.searchsorted(pd.Timestamp(evaluation_end),side="left")))
    frame_cache={name:{"position":None,"frame":None} for name in frames_all}
    for i in range(first,last):
        now=bars.index[i]
        history=bars.iloc[max(0,i-399):i+1]
        trade_end=trade_times.searchsorted(now,side="left")
        known=trades.iloc[max(0,trade_end-3000):trade_end]
        frames={}
        for name,frame in frames_all.items():
            position=int(frame.index.searchsorted(now,side="right"))
            cached=frame_cache[name]
            if cached["position"]!=position:
                cached["position"]=position;cached["frame"]=frame.iloc[max(0,position-400):position]
            frames[name]=cached["frame"]
        out=predictor.predict(history,known,100_000,frames=frames,flow_bars=history,flow_source="historical_binance_kline")
        key=(out.zone,out.sweep_time)
        if out.entry is not None and out.stop is not None and out.target is not None and out.sweep_time and key not in seen:
            seen.add(key)
            events.append({
                "decision_time":now,"bias":out.bias,"zone":out.zone,"sweep_time":pd.Timestamp(out.sweep_time),
                "reclaim_time":pd.Timestamp(out.reclaim_time) if out.reclaim_time else now,
                "entry":float(out.entry),"stop":float(out.stop),"target":float(out.target),
                "market_flow_score":float(out.market_flow_score or 0.0),"setup_type":out.setup_type,
            })
    return pd.DataFrame(events)


def _enumerate_signal_chunk(arguments):
    return _enumerate_signal_range(*arguments)


def enumerate_signal_events(candles, start, end, progress=None, workers=1, chunks=None) -> pd.DataFrame:
    """Enumerate causal production-MTF setups without applying a flow cutoff.

    Decision ranges may run in separate local processes. Every range receives the
    same global seven-day warmup and full closed-bar history, so splitting changes
    throughput only—not the information available at a decision timestamp.
    """
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    total=max(0,int(candles.index.searchsorted(end,side="left")-candles.index.searchsorted(start,side="left")))
    workers=max(1,int(workers))
    if workers==1:
        result=_enumerate_signal_range(candles,start,end,start,end)
        if progress:
            progress(total,total,len(result))
        return result

    chunk_count=max(workers,int(chunks or workers*2))
    boundaries=pd.date_range(start,end,periods=chunk_count+1)
    arguments=[(candles,start,end,boundaries[index],boundaries[index+1]) for index in range(chunk_count)]
    completed=[]; done_bars=0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(_enumerate_signal_chunk,item):item for item in arguments}
        for future in as_completed(futures):
            item=futures[future]
            frame=future.result(); completed.append(frame)
            done_bars+=int(candles.index.searchsorted(item[4],side="left")-candles.index.searchsorted(item[3],side="left"))
            if progress:
                progress(min(total,done_bars),total,sum(len(value) for value in completed))
    populated=[frame for frame in completed if not frame.empty]
    if not populated:
        return pd.DataFrame()
    return (pd.concat(populated,ignore_index=True)
        .sort_values("decision_time")
        .drop_duplicates(["zone","sweep_time"],keep="first")
        .reset_index(drop=True))


def attach_raw_features(events, aggregates_by_bucket):
    rows=[]
    for event in events.to_dict("records"):
        raw={}
        for bucket,aggregates in aggregates_by_bucket.items():
            components=footprint_components_from_aggregates(
                aggregates,event["bias"],event["sweep_time"],event["reclaim_time"]
            )
            raw[str(float(bucket))]=components
        rows.append({**event,"raw_by_bucket":raw})
    return rows


def _raw_score(event, config):
    components=event["raw_by_bucket"][str(float(config.price_bucket))]
    ratio=float(components["ratio"]); span=max(config.full_credit_ratio-1.0,1e-6)
    if event["bias"]=="bullish":
        imbalance=min(1.0,max(0.0,(ratio-1.0)/span))
    else:
        imbalance=min(1.0,max(0.0,(1.0/ratio-1.0)/span))
    return 0.75*imbalance+0.25*float(components["agreement"]),bool(components["eligible"])


def _event_outcome(event,bars,end,fee_bps=5,slippage_bps=2,same_bar_policy="conservative"):
    decision=pd.Timestamp(event["decision_time"]); end=pd.Timestamp(end)
    fill_pos=bars.index.searchsorted(decision,side="right")
    if fill_pos>=len(bars) or bars.index[fill_pos]>=end:
        return None
    side="long" if event["bias"]=="bullish" else "short"; sign=1 if side=="long" else -1
    entry=float(bars.open.iloc[fill_pos])*(1+sign*slippage_bps/10_000)
    stop=float(event["stop"]); target=float(event["target"])
    exit_price=None; exit_time=None; reason=None
    for ts,bar in bars.iloc[fill_pos:].loc[lambda frame:frame.index<end].iterrows():
        hit_stop=float(bar.low)<=stop if side=="long" else float(bar.high)>=stop
        hit_target=float(bar.high)>=target if side=="long" else float(bar.low)<=target
        if hit_stop or hit_target:
            use_stop=(same_bar_policy=="conservative") if hit_stop and hit_target else hit_stop
            raw=stop if use_stop else target; exit_sign=-1 if side=="long" else 1
            exit_price=float(raw)*(1+exit_sign*slippage_bps/10_000)
            exit_time=ts; reason="stop" if use_stop else "target"; break
    if exit_price is None:
        eligible=bars.loc[(bars.index>decision)&(bars.index<end)]
        if eligible.empty:return None
        exit_time=eligible.index[-1]; raw=float(eligible.close.iloc[-1]); exit_sign=-1 if side=="long" else 1
        exit_price=raw*(1+exit_sign*slippage_bps/10_000); reason="fold_end"
    direction=1 if side=="long" else -1
    gross_per_unit=(exit_price-entry)*direction
    fees_per_unit=(entry+exit_price)*fee_bps/10_000
    risk_per_unit=abs(entry-stop)
    if risk_per_unit<=0:return None
    return {"entry_time":bars.index[fill_pos],"exit_time":exit_time,"entry":entry,"exit":exit_price,"risk_per_unit":risk_per_unit,"pnl_per_unit":gross_per_unit-fees_per_unit,"exit_reason":reason}


def _stats(ledger,initial_equity=100_000):
    if not ledger:
        return {"trades":0,"wins":0,"losses":0,"average_r":None,"profit_factor":None,"net_pnl":0.0,"maximum_drawdown":0.0,"profit_concentration":None}
    frame=pd.DataFrame(ledger); wins=frame.loc[frame.pnl>0]; losses=frame.loc[frame.pnl<=0]
    gross_profit=float(wins.pnl.sum()); gross_loss=float(-losses.loc[losses.pnl<0,"pnl"].sum())
    curve=pd.concat([pd.Series([initial_equity]),frame.equity.reset_index(drop=True)]); drawdown=curve.cummax()-curve
    concentration=float(wins.pnl.max()/gross_profit) if gross_profit>0 else None
    return {"trades":len(frame),"wins":len(wins),"losses":len(losses),"average_r":float(frame.r_multiple.mean()),"profit_factor":gross_profit/gross_loss if gross_loss else (math.inf if gross_profit else None),"net_pnl":float(frame.pnl.sum()),"maximum_drawdown":float(drawdown.max()),"profit_concentration":concentration}


def replay_events(events,config,bars,start,end,initial_equity=100_000,risk_fraction=.0025):
    eligible=[]
    for event in events:
        decision=pd.Timestamp(event["decision_time"])
        if not (pd.Timestamp(start)<=decision<pd.Timestamp(end)) or float(event["market_flow_score"])<config.market_threshold:
            continue
        raw_score,complete=_raw_score(event,config)
        if complete and raw_score>=config.raw_threshold:
            eligible.append((event,raw_score))
    eligible.sort(key=lambda item:pd.Timestamp(item[0]["decision_time"]))
    equity=float(initial_equity); ledger=[]; unavailable_until=None
    for event,raw_score in eligible:
        decision=pd.Timestamp(event["decision_time"])
        if unavailable_until is not None and decision<unavailable_until:
            continue
        outcome=_event_outcome(event,bars,end)
        if outcome is None:continue
        size=equity*risk_fraction/outcome["risk_per_unit"]
        pnl=outcome["pnl_per_unit"]*size; equity+=pnl
        ledger.append({**outcome,"decision_time":decision,"zone":event["zone"],"market_flow_score":event["market_flow_score"],"raw_footprint_score":raw_score,"size":size,"pnl":pnl,"equity":equity,"r_multiple":pnl/(outcome["risk_per_unit"]*size)})
        unavailable_until=pd.Timestamp(outcome["exit_time"])
    return ledger,_stats(ledger,initial_equity)


def _adjacent(left,right):
    a=asdict(left); b=asdict(right); changed=[key for key in a if a[key]!=b[key]]
    if len(changed)!=1:return False
    key=changed[0]; grids={"price_bucket":PRICE_BUCKETS,"full_credit_ratio":FULL_CREDIT_RATIOS,"market_threshold":SCORE_THRESHOLDS,"raw_threshold":SCORE_THRESHOLDS}
    values=grids[key]
    return abs(values.index(a[key])-values.index(b[key]))==1


def select_configuration(events,bars,start):
    periods=walk_forward_periods(start); candidates=[]
    for config in calibration_grid():
        fold_stats=[replay_events(events,config,bars,period["validation_start"],period["validation_end"])[1] for period in periods]
        trades=sum(item["trades"] for item in fold_stats)
        positive=sum(1 for item in fold_stats if item["average_r"] is not None and item["average_r"]>0)
        average_values=[item["average_r"] for item in fold_stats if item["average_r"] is not None]
        pf_values=[item["profit_factor"] for item in fold_stats if item["profit_factor"] is not None]
        median_r=float(np.median(average_values)) if average_values else -math.inf
        median_pf=float(np.median(pf_values)) if pf_values else None
        eligible=bool(all(item["trades"]>=5 for item in fold_stats) and trades>=40 and positive>=3 and median_pf is not None and median_pf>1.10 and max(item["maximum_drawdown"] for item in fold_stats)<=5_000)
        candidates.append({"config":config,"folds":fold_stats,"validation_trades":trades,"positive_folds":positive,"median_average_r":median_r,"median_profit_factor":median_pf,"eligible":eligible})
    eligible=[item for item in candidates if item["eligible"]]
    for item in eligible:
        stable_neighbors=[other for other in eligible if other is not item and _adjacent(item["config"],other["config"]) and other["median_average_r"]>=0.9*item["median_average_r"]]
        item["stable_plateau"]=bool(stable_neighbors)
    ranked=sorted((item for item in eligible if item.get("stable_plateau")),key=lambda item:(item["median_average_r"],item["median_profit_factor"] or 0,-max(fold["maximum_drawdown"] for fold in item["folds"])),reverse=True)
    return ranked[:20]


def run_calibration(candles,events,aggregates_by_bucket,manifests,start,end,output_path,progress=None):
    enriched=attach_raw_features(events,aggregates_by_bucket)
    top=select_configuration(enriched,candles,start)
    selected=top[0] if top else None
    holdout=None; passed=False; reasons=[]
    if selected:
        holdout_start=pd.Timestamp(start)+pd.Timedelta(days=40)
        _,holdout=replay_events(enriched,selected["config"],candles,holdout_start,end)
        requirements={
            "minimum_trades":holdout["trades"]>=20,
            "positive_average_r":holdout["average_r"] is not None and holdout["average_r"]>0,
            "profit_factor":holdout["profit_factor"] is not None and holdout["profit_factor"]>=1.20,
            "maximum_drawdown":holdout["maximum_drawdown"]<=5_000,
            "profit_concentration":holdout["profit_concentration"] is not None and holdout["profit_concentration"]<=0.25,
        }
        reasons=[key for key,value in requirements.items() if not value]; passed=all(requirements.values())
        holdout["requirements"]=requirements
    else:
        reasons=["no_stable_development_candidate"]
    identity={"revision":_revision(),"start":pd.Timestamp(start).isoformat(),"end":pd.Timestamp(end).isoformat(),"manifests":[{"exchange":item["exchange"],"date":item["date"],"sha256":item["archive_sha256"]} for item in sorted(manifests,key=lambda value:(value["date"],value["exchange"]))],"grid":{"price_buckets":PRICE_BUCKETS,"full_credit_ratios":FULL_CREDIT_RATIOS,"thresholds":SCORE_THRESHOLDS}}
    run_hash=hashlib.sha256(json.dumps(identity,sort_keys=True,default=str).encode()).hexdigest()
    artifact={"status":"complete","run_hash":run_hash,"identity":identity,"promotion_passed":passed,"promotion_failures":reasons,"selected_config":asdict(selected["config"]) if selected else None,"development":({key:value for key,value in selected.items() if key!="config"} if selected else None),"holdout":holdout,"top_candidates":[{**{key:value for key,value in item.items() if key!="config"},"config":asdict(item["config"])} for item in top]}
    JsonStore(output_path).write(artifact)
    return artifact
