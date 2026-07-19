from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests


BASE = "https://data.binance.vision/data/futures/um"


def read_zip(url):
    response = requests.get(url, timeout=120)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            raw = pd.read_csv(handle, header=None)
    # Newer archives can include a textual header row.
    raw = raw[pd.to_numeric(raw[0], errors="coerce").notna()].copy()
    return pd.DataFrame({
        "close_time":pd.to_datetime(pd.to_numeric(raw[6]),unit="ms",utc=True),
        "open":pd.to_numeric(raw[1]),"high":pd.to_numeric(raw[2]),"low":pd.to_numeric(raw[3]),
        "close":pd.to_numeric(raw[4]),"volume":pd.to_numeric(raw[5]),
        "taker_buy_volume":pd.to_numeric(raw[9]),"trades":pd.to_numeric(raw[8]),
    })


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--start",required=True); parser.add_argument("--end",required=True); parser.add_argument("--output",required=True)
    args=parser.parse_args(); start=pd.Timestamp(args.start); end=pd.Timestamp(args.end)
    start=start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end=end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    pieces=[]
    for month in pd.period_range(start.tz_localize(None).to_period("M")+1, end.tz_localize(None).to_period("M")-1, freq="M"):
        name=f"BTCUSDT-1m-{month}.zip"; frame=read_zip(f"{BASE}/monthly/klines/BTCUSDT/1m/{name}")
        if frame is not None: pieces.append(frame); print(name,len(frame),flush=True)
    # Partial boundary months use daily files, which also makes the exact date range reproducible.
    for day in pd.date_range(start.floor("D"),end.floor("D"),freq="D",inclusive="left"):
        if day.to_period("M") not in {start.tz_localize(None).to_period("M"),end.tz_localize(None).to_period("M")}: continue
        stamp=day.strftime("%Y-%m-%d"); name=f"BTCUSDT-1m-{stamp}.zip"; frame=read_zip(f"{BASE}/daily/klines/BTCUSDT/1m/{name}")
        if frame is not None: pieces.append(frame); print(name,len(frame),flush=True)
    out=pd.concat(pieces,ignore_index=True).drop_duplicates("close_time").sort_values("close_time")
    out=out[(out.close_time>=start)&(out.close_time<=end)]
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output,index=False)
    print(f"saved {len(out)} bars",flush=True)


if __name__ == "__main__": main()
