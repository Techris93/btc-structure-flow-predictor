from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from btc_predictor.flow_calibration import (
    PRICE_BUCKETS,
    aggregate_trade_partitions,
    enumerate_signal_events,
    load_ohlcv,
    run_calibration,
)
from btc_predictor.historical_trades import common_complete_days, normalize_range
from btc_predictor.persistence import JsonStore


def main():
    parser=argparse.ArgumentParser(description="Causal 60-day two-venue order-flow calibration")
    parser.add_argument("--ohlcv",default="work/runtime/research-local/binance_1m.csv")
    parser.add_argument("--data-dir",default="work/runtime/flow-calibration")
    parser.add_argument("--end",help="Exclusive UTC end; defaults to the day after the latest complete OHLCV day")
    parser.add_argument("--days",type=int,default=60)
    parser.add_argument("--no-download",action="store_true")
    parser.add_argument("--force-download",action="store_true")
    parser.add_argument("--workers",type=int,default=min(4,os.cpu_count() or 1),help="Local causal-enumeration processes")
    args=parser.parse_args()
    root=Path(args.data_dir); root.mkdir(parents=True,exist_ok=True)
    status=JsonStore(root/"status.json")
    try:
        candles=load_ohlcv(args.ohlcv)
        end=pd.Timestamp(args.end) if args.end else candles.index[-1].floor("D")+pd.Timedelta(days=1)
        end=end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        start=end-pd.Timedelta(days=args.days)
        if args.days!=60:
            raise ValueError("The locked calibration design requires exactly 60 evaluation days")
        if candles.index.min()>start-pd.Timedelta(days=7) or candles.index.max()<end-pd.Timedelta(minutes=1):
            raise ValueError("OHLCV does not cover the 60-day evaluation plus seven-day warmup")
        trade_root=root/"trades"
        manifests=[]
        if not args.no_download:
            def download_progress(done,total,manifest):
                status.write({"status":"running","phase":"downloading_and_normalizing","done":done,"total":total,"last":manifest})
            manifests=normalize_range(start,end,trade_root,force=args.force_download,progress=download_progress)
        else:
            for day in pd.date_range(start,end,freq="D",inclusive="left"):
                for exchange in ("binance","bybit"):
                    path=trade_root/exchange/f"date={day:%Y-%m-%d}"/"manifest.json"
                    manifests.append(json.loads(path.read_text()))
        complete=common_complete_days(trade_root,start,end)
        expected=[day.strftime("%Y-%m-%d") for day in pd.date_range(start,end,freq="D",inclusive="left")]
        missing=sorted(set(expected)-set(complete))
        if missing:
            failure={"status":"failed","phase":"data_validation","error":"incomplete_two_venue_history","missing_days":missing,"complete_days":len(complete)}
            status.write(failure); print(json.dumps(failure,indent=2)); return 2
        status.write({"status":"running","phase":"enumerating_causal_setups","start":start.isoformat(),"end":end.isoformat()})
        def event_progress(done,total,events):
            if done%1000==0 or done==total:
                status.write({"status":"running","phase":"enumerating_causal_setups","bars":done,"total_bars":total,"events":events})
        events=enumerate_signal_events(candles,start,end,progress=event_progress,workers=args.workers)
        events.to_csv(root/"causal_events.csv",index=False)
        aggregates={}
        for index,bucket in enumerate(PRICE_BUCKETS,1):
            status.write({"status":"running","phase":"aggregating_raw_footprint","bucket":bucket,"bucket_index":index,"bucket_total":len(PRICE_BUCKETS)})
            aggregates[bucket]=aggregate_trade_partitions(trade_root,complete,bucket)
        status.write({"status":"running","phase":"walk_forward_selection","events":len(events)})
        artifact=run_calibration(candles,events,aggregates,manifests,start,end,root/"flow_calibration.json")
        status.write(artifact); print(json.dumps(artifact,indent=2,default=str)); return 0
    except Exception as exc:
        failure={"status":"failed","phase":"exception","error":f"{type(exc).__name__}: {exc}"}
        status.write(failure); print(json.dumps(failure,indent=2)); raise


if __name__=="__main__":
    raise SystemExit(main())
