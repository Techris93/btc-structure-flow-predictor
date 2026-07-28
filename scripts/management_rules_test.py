"""Measure trade-management rules against the continuation baseline.

Three replays of the same 30-day dataset, isolating each rule:
  abort_only  - scratch trades that never reach +0.5R within 8h
  trail_only  - move stop to breakeven once a trade reaches +0.8R
  both        - abort + trail together

Compare against outputs/regime_validation/continuation-result.json (baseline).
Runs checkpoint and resume if interrupted.

Usage: python scripts/management_rules_test.py [abort_only|trail_only|both]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from btc_predictor.backtest import run_event_backtest
from btc_predictor.persistence import JsonStore
from btc_predictor.research import proxy_trades
from btc_predictor.strategy import Predictor

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "outputs/backtest_30d_3cbb8ab/binance_1m.csv"
FEATURES = REPO / "outputs/features"
OUT = REPO / "outputs/management_rules"
DECISION_START = pd.Timestamp("2026-06-28 00:00", tz="UTC")

CONFIGS = {
    "abort_only": {"abort_if_below_r": 0.5, "abort_after_minutes": 480.0},
    "trail_only": {"breakeven_trail_r": 0.8},
    "both": {"abort_if_below_r": 0.5, "abort_after_minutes": 480.0, "breakeven_trail_r": 0.8},
}


def build_features() -> pd.DataFrame:
    funding = pd.read_csv(FEATURES / "funding.csv")
    funding["time"] = pd.to_datetime(funding.time, format="ISO8601", utc=True)
    funding = funding.set_index("time")
    oi = pd.read_csv(FEATURES / "open_interest.csv")
    oi["time"] = pd.to_datetime(oi.time, format="ISO8601", utc=True)
    oi = oi.set_index("time")
    oi["oi_chg_6h"] = oi.open_interest.pct_change(6)
    return funding[["funding_rate"]].join(oi[["oi_chg_6h"]], how="outer").sort_index().ffill()


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "both"
    if name not in CONFIGS:
        raise SystemExit(f"unknown config {name!r}; choose from {sorted(CONFIGS)}")
    OUT.mkdir(parents=True, exist_ok=True)
    bars = pd.read_csv(DATASET, parse_dates=["close_time"]).set_index("close_time")
    bars.index = pd.to_datetime(bars.index, utc=True)
    trades = proxy_trades(bars)
    features = build_features()
    status = JsonStore(OUT / f"{name}-status.json")
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
        bars, trades, Predictor(reversal_enabled=False), mode="reactive",
        fee_bps=5, slippage_bps=2, same_bar_policy="conservative",
        decision_start=DECISION_START, progress=progress,
        resume_state=resume, checkpoint=save_resume, features=features,
        **CONFIGS[name],
    )
    ledger.to_csv(OUT / f"{name}-ledger.csv", index=False)
    result = {"status": "complete", "config": name, "rule_params": CONFIGS[name],
              "decision_start": str(DECISION_START), "stats": stats,
              "completed_at": pd.Timestamp.now(tz="UTC").isoformat()}
    (OUT / f"{name}-result.json").write_text(json.dumps(result, indent=2, default=str))
    checkpoint_store.write({"complete": True, "dataset_start": str(DECISION_START)})
    status.write({"status": "complete", "config": name, "stats": stats})
    print(json.dumps({"config": name, "stats": stats}, indent=2, default=str))


if __name__ == "__main__":
    main()
