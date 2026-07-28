from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import pandas as pd
import requests

from .backtest import run_event_backtest
from .persistence import JsonStore
from .strategy import Predictor


def _revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def fetch_binance_one_minute(start, end, dataset_path: Path, status: JsonStore):
    """Download close-indexed 1m candles and checkpoint every exchange page."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    existing = pd.read_csv(dataset_path, parse_dates=["close_time"]).set_index("close_time") if dataset_path.exists() else pd.DataFrame()
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)

    def _fetch_pages(from_ms, to_ms, pages):
        frames = []
        cursor = from_ms
        session = requests.Session()
        while cursor < to_ms:
            payload = None
            for attempt in range(10):
                try:
                    response = session.get("https://fapi.binance.com/fapi/v1/klines", params={"symbol":"BTCUSDT","interval":"1m","startTime":cursor,"endTime":to_ms,"limit":1500}, timeout=30)
                    response.raise_for_status()
                    payload = response.json()
                    break
                except Exception as exc:
                    if attempt == 9: raise
                    time.sleep(min(2 * (attempt + 1), 15))
            if payload is None: break
            if not payload: break
            raw = pd.DataFrame(payload)
            page = pd.DataFrame({
                "close_time": pd.to_datetime(pd.to_numeric(raw[6]), unit="ms", utc=True),
                "open": pd.to_numeric(raw[1]), "high": pd.to_numeric(raw[2]), "low": pd.to_numeric(raw[3]),
                "close": pd.to_numeric(raw[4]), "volume": pd.to_numeric(raw[5]),
                "taker_buy_volume": pd.to_numeric(raw[9]), "trades": pd.to_numeric(raw[8]),
            }).set_index("close_time")
            frames.append(page)
            cursor = int(raw.iloc[-1, 6]) + 1; pages += 1
            merged = pd.concat([existing, *frames]) if len(existing) else pd.concat(frames)
            merged = merged.loc[lambda x: ~x.index.duplicated(keep="last")].sort_index()
            merged.to_csv(dataset_path)
            status.write({"status":"running","phase":"fetching","checkpoint":"dataset_page","pages":pages,"bars":len(merged),"last_close":str(merged.index[-1])})
        return (pd.concat(frames) if frames else pd.DataFrame()), pages

    # Unified gap detection: close-indexed 1m bars arrive exactly 60s apart on a
    # 24/7 market, so any wider spacing is missing data — head, tail, or holes
    # left behind by interrupted partial downloads.
    if len(existing):
        ranges = []
        prev = start_ms - 60_000
        for ts in existing.index:
            ts_ms = int(ts.timestamp() * 1000)
            if ts_ms - prev > 90_000:
                ranges.append((prev + 60_000, min(ts_ms - 1, end_ms)))
            prev = ts_ms
        if end_ms - prev > 90_000:
            ranges.append((prev + 60_000, end_ms))
    else:
        ranges = [(start_ms, end_ms)]
    pages = 0
    for from_ms, to_ms in ranges:
        new_rows, pages = _fetch_pages(from_ms, to_ms, pages)
        if len(new_rows):
            existing = pd.concat([existing, new_rows]).loc[lambda x: ~x.index.duplicated(keep="last")].sort_index()
            existing.to_csv(dataset_path)
    return existing.loc[(existing.index >= start) & (existing.index <= end)]


def proxy_trades(one_minute: pd.DataFrame):
    buy = one_minute.taker_buy_volume.clip(lower=0)
    sell = (one_minute.volume - buy).clip(lower=0)
    event_time = one_minute.index - pd.Timedelta(nanoseconds=1)
    a = pd.DataFrame({"time":event_time,"price":one_minute.close.to_numpy(),"qty":buy.to_numpy(),"side":"buy"})
    b = pd.DataFrame({"time":event_time,"price":one_minute.close.to_numpy(),"qty":sell.to_numpy(),"side":"sell"})
    return pd.concat([a, b], ignore_index=True).loc[lambda x: x.qty > 0].sort_values("time")


def run_comparison(data_dir, start, end, config=None):
    config = {"fee_bps":5,"slippage_bps":2,"same_bar_policy":"conservative","decision_stride":1,"warmup_days":10, **(config or {})}
    warmup_days = int(config.get("warmup_days", 10))
    root = Path(data_dir); root.mkdir(parents=True, exist_ok=True)
    status, result_store = JsonStore(root/"status.json"), JsonStore(root/"result.json")
    identity = {"revision":_revision(),"start":str(start),"end":str(end),"config":config}
    run_hash = _hash(identity)
    prior = result_store.read({})
    if prior.get("run_hash") == run_hash and prior.get("status") == "complete": return prior
    status.write({"status":"running","phase":"fetching","run_hash":run_hash,"identity":identity,"started_at":pd.Timestamp.utcnow().isoformat()})
    fetch_start = pd.Timestamp(start) - pd.Timedelta(days=warmup_days)
    bars = fetch_binance_one_minute(fetch_start, end, root/"binance_1m.csv", status)
    dataset_hash = hashlib.sha256(pd.util.hash_pandas_object(bars, index=True).values.tobytes()).hexdigest()
    trades = proxy_trades(bars)
    outputs = {}
    bt_config = {key: value for key, value in config.items() if key != "warmup_days"}
    for mode in ("reactive", "mtf"):
        checkpoint = JsonStore(root/f"{run_hash}-{mode}.json")
        saved = checkpoint.read({})
        if saved.get("complete") and saved.get("dataset_hash") == dataset_hash:
            outputs[mode] = saved["stats"]; continue
        resume = saved.get("resume_state") if saved.get("dataset_hash") == dataset_hash else None
        def save_resume(replay_state):
            checkpoint.write({"complete":False,"dataset_hash":dataset_hash,"resume_state":replay_state,"bars_processed":replay_state["next_i"],"updated_at":pd.Timestamp.utcnow().isoformat()})
        def progress(done, total, ledger):
            status.write({"status":"running","phase":"backtesting","variant":mode,"bars_processed":done,"total_bars":total,"closed_trades":len(ledger),"run_hash":run_hash})
        ledger, stats = run_event_backtest(bars, trades, Predictor(), mode=mode, progress=progress, resume_state=resume, checkpoint=save_resume, decision_start=start, **bt_config)
        ledger.to_csv(root/f"{run_hash}-{mode}-ledger.csv", index=False)
        checkpoint.write({"complete":True,"dataset_hash":dataset_hash,"stats":stats,"ledger":f"{run_hash}-{mode}-ledger.csv"})
        outputs[mode] = stats
    result = {"status":"complete","run_hash":run_hash,"dataset_hash":dataset_hash,"identity":identity,"bars":len(bars),"data_source":"Binance Futures 1m klines with taker-buy volume","variants":outputs,"completed_at":pd.Timestamp.utcnow().isoformat()}
    result_store.write(result); status.write(result)
    return result
