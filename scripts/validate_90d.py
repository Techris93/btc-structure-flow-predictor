"""90-day confirmation of the frozen continuation system (no re-fitting)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from btc_predictor.backtest import run_event_backtest
from btc_predictor.persistence import JsonStore
from btc_predictor.research import fetch_binance_one_minute, proxy_trades
from btc_predictor.strategy import Predictor

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs/validate_90d"
FEATURES = REPO / "outputs/features"
# Parameters are frozen from the 30-day validation; Apr 28 - Jun 17 is fully
# unseen out-of-sample data for every threshold and gate.
DECISION_START = pd.Timestamp("2026-04-28 00:00", tz="UTC")
FETCH_START = DECISION_START - pd.Timedelta(days=10)
END = pd.Timestamp("2026-07-28 23:59", tz="UTC")

CONFIGS = {
    "continuation": dict(reversal_enabled=False),
    "regime_rulebook": dict(),
}


def extend_funding() -> None:
    """Backfill funding.csv to FETCH_START (endpoint has full history)."""
    path = FEATURES / "funding.csv"
    existing = pd.read_csv(path)
    existing["time"] = pd.to_datetime(existing.time, format="ISO8601", utc=True)
    cursor = int(FETCH_START.timestamp() * 1000)
    end_ms = int(existing.time.min().timestamp() * 1000)
    rows, session = [], requests.Session()
    while cursor < end_ms:
        page = None
        for attempt in range(8):
            try:
                r = session.get("https://fapi.binance.com/fapi/v1/fundingRate",
                                params={"symbol": "BTCUSDT", "startTime": cursor,
                                        "endTime": end_ms, "limit": 1000}, timeout=30)
                r.raise_for_status()
                page = r.json()
                break
            except Exception:
                time.sleep(min(2 * (attempt + 1), 15))
        if not page:
            break
        rows.extend(page)
        cursor = int(page[-1]["fundingTime"]) + 1
    if rows:
        add = pd.DataFrame(rows)
        add["time"] = pd.to_datetime(pd.to_numeric(add.fundingTime), unit="ms", utc=True)
        add["funding_rate"] = pd.to_numeric(add.fundingRate)
        existing = pd.concat([existing, add[["time", "funding_rate"]]])
        existing = existing.drop_duplicates("time").sort_values("time")
        existing.to_csv(path, index=False)


def build_features() -> pd.DataFrame:
    funding = pd.read_csv(FEATURES / "funding.csv")
    funding["time"] = pd.to_datetime(funding.time, format="ISO8601", utc=True)
    return funding.set_index("time")[["funding_rate"]].sort_index().ffill()


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "continuation"
    if name not in CONFIGS:
        raise SystemExit(f"unknown config {name!r}; choose from {sorted(CONFIGS)}")
    OUT.mkdir(parents=True, exist_ok=True)
    FEATURES.mkdir(parents=True, exist_ok=True)
    status = JsonStore(OUT / f"{name}-status.json")
    extend_funding()
    bars = fetch_binance_one_minute(FETCH_START, END, OUT / "binance_1m.csv", status)
    trades = proxy_trades(bars)
    features = build_features()
    checkpoint_store = JsonStore(OUT / f"{name}-checkpoint.json")
    saved = checkpoint_store.read({})
    resume = saved.get("resume_state") if saved.get("dataset_start") == str(DECISION_START) else None

    def save_resume(state):
        checkpoint_store.write({
            "complete": False, "dataset_start": str(DECISION_START),
            "resume_state": state, "bars_processed": state["next_i"],
            "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        })

    def progress(done, total, ledger):
        status.write({"status": "running", "config": name, "bars_processed": done,
                      "total_bars": total, "closed_trades": len(ledger)})

    ledger, stats = run_event_backtest(
        bars, trades, Predictor(**CONFIGS[name]), mode="reactive",
        fee_bps=5, slippage_bps=2, same_bar_policy="conservative",
        decision_start=DECISION_START, progress=progress,
        resume_state=resume, checkpoint=save_resume, features=features,
    )
    ledger.to_csv(OUT / f"{name}-ledger.csv", index=False)
    result = {"status": "complete", "config": name, "predictor_config": CONFIGS[name],
              "decision_start": str(DECISION_START), "stats": stats,
              "completed_at": pd.Timestamp.now(tz="UTC").isoformat()}
    (OUT / f"{name}-result.json").write_text(json.dumps(result, indent=2, default=str))
    checkpoint_store.write({"complete": True, "dataset_start": str(DECISION_START)})
    status.write({"status": "complete", "config": name, "stats": stats})
    print(json.dumps({"config": name, "stats": stats}, indent=2, default=str))


if __name__ == "__main__":
    main()
