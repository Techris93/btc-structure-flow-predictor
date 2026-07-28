"""Fetch funding, open interest, and spot klines for feature studies.

Downloads into outputs/features/:
  funding.csv      - 8h funding rates (fapi /fapi/v1/fundingRate)
  open_interest.csv - hourly OI (fapi /futures/data/openInterestHist, ~30d history)
  spot_1m.csv      - spot BTCUSDT 1m closes for perp-spot basis
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs/features"
START = pd.Timestamp("2026-06-18", tz="UTC")
END = pd.Timestamp("2026-07-28 23:59", tz="UTC")
S = requests.Session()


def _get(url, params, retries=8):
    for attempt in range(retries):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(min(5 * (attempt + 1), 30))
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(2 * (attempt + 1), 15))
    return []


def fetch_funding():
    rows, cursor = [], int(START.timestamp() * 1000)
    end_ms = int(END.timestamp() * 1000)
    while cursor < end_ms:
        page = _get("https://fapi.binance.com/fapi/v1/fundingRate",
                    {"symbol": "BTCUSDT", "startTime": cursor, "endTime": end_ms, "limit": 1000})
        if not page:
            break
        rows.extend(page)
        cursor = int(page[-1]["fundingTime"]) + 1
    df = pd.DataFrame(rows)
    if len(df):
        df["time"] = pd.to_datetime(pd.to_numeric(df.fundingTime), unit="ms", utc=True)
        df["funding_rate"] = pd.to_numeric(df.fundingRate)
        df = df[["time", "funding_rate"]].drop_duplicates("time").sort_values("time")
    df.to_csv(OUT / "funding.csv", index=False)
    return len(df)


def fetch_open_interest():
    # futures/data keeps roughly 30 days of history; older startTime returns 400.
    oi_start = max(START, END - pd.Timedelta(days=29))
    rows, cursor = [], int(oi_start.timestamp() * 1000)
    end_ms = int(END.timestamp() * 1000)
    while cursor < end_ms:
        page = _get("https://fapi.binance.com/futures/data/openInterestHist",
                    {"symbol": "BTCUSDT", "period": "1h", "startTime": cursor, "endTime": end_ms, "limit": 500})
        if not page:
            break
        rows.extend(page)
        cursor = int(page[-1]["timestamp"]) + 1
        if len(page) < 500:
            break
    df = pd.DataFrame(rows)
    if len(df):
        df["time"] = pd.to_datetime(pd.to_numeric(df.timestamp), unit="ms", utc=True)
        df["open_interest"] = pd.to_numeric(df.sumOpenInterest)
        df["oi_value"] = pd.to_numeric(df.sumOpenInterestValue)
        df = df[["time", "open_interest", "oi_value"]].drop_duplicates("time").sort_values("time")
    df.to_csv(OUT / "open_interest.csv", index=False)
    return len(df)


def fetch_spot():
    rows, cursor = [], int(START.timestamp() * 1000)
    end_ms = int(END.timestamp() * 1000)
    while cursor < end_ms:
        page = _get("https://api.binance.com/api/v3/klines",
                    {"symbol": "BTCUSDT", "interval": "1m", "startTime": cursor, "endTime": end_ms, "limit": 1500})
        if not page:
            break
        df = pd.DataFrame(page)
        rows.append(pd.DataFrame({
            "close_time": pd.to_datetime(pd.to_numeric(df[6]), unit="ms", utc=True),
            "spot_close": pd.to_numeric(df[4]),
        }))
        cursor = int(df.iloc[-1, 6]) + 1
    df = pd.concat(rows).drop_duplicates("close_time").sort_values("close_time") if rows else pd.DataFrame()
    df.to_csv(OUT / "spot_1m.csv", index=False)
    return len(df)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("funding rows:", fetch_funding())
    print("open interest rows:", fetch_open_interest())
    print("spot rows:", fetch_spot())
