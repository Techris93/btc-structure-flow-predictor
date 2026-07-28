"""Regime-rulebook validation on the 30-day dataset with funding/OI features.

Configs:
  continuation    - drift-continuation only (transition regime, 18h time exit)
  regime_rulebook - continuation in transition + reversal in trend-exhaustion,
                    stand down in range (all four research changes combined)

Both replay the full decision window (Jun 28 - Jul 28) with causal funding and
OI context. Runs checkpoint and resume if interrupted.

Usage: python scripts/regime_validation.py [continuation|regime_rulebook]
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
OUT = REPO / "outputs/regime_validation"
DECISION_START = pd.Timestamp("2026-06-28 00:00", tz="UTC")

CONFIGS = {
    "continuation": dict(reversal_enabled=False),
    "regime_rulebook": dict(),
}


def build_features() -> pd.DataFrame:
    funding = pd.read_csv(FEATURES / "funding.csv")
    funding["time"] = pd.to_datetime(funding.time, format="ISO8601", utc=True)
    funding = funding.set_index("time")
    oi = pd.read_csv(FEATURES / "open_interest.csv")
    oi["time"] = pd.to_datetime(oi.time, format="ISO8601", utc=True)
    oi = oi.set_index("time")
    oi["oi_chg_6h"] = oi.open_interest.pct_change(6)
    feats = funding[["funding_rate"]].join(oi[["oi_chg_6h"]], how="outer").sort_index()
    # ffill only carries past observations forward: causal at any decision time.
    return feats.ffill()


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "regime_rulebook"
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
            "complete": False,
            "dataset_start": str(DECISION_START),
            "resume_state": state,
            "bars_processed": state["next_i"],
            "updated_at": pd.Timestamp.utcnow().isoformat(),
        })

    def progress(done, total, ledger):
        status.write({
            "status": "running", "config": name, "bars_processed": done,
            "total_bars": total, "closed_trades": len(ledger),
        })

    ledger, stats = run_event_backtest(
        bars, trades, Predictor(**CONFIGS[name]), mode="reactive",
        fee_bps=5, slippage_bps=2, same_bar_policy="conservative",
        decision_start=DECISION_START, progress=progress,
        resume_state=resume, checkpoint=save_resume, features=features,
    )
    ledger.to_csv(OUT / f"{name}-ledger.csv", index=False)
    result = {
        "status": "complete", "config": name, "predictor_config": CONFIGS[name],
        "decision_start": str(DECISION_START),
        "stats": stats, "completed_at": pd.Timestamp.utcnow().isoformat(),
    }
    (OUT / f"{name}-result.json").write_text(json.dumps(result, indent=2, default=str))
    checkpoint_store.write({"complete": True, "dataset_start": str(DECISION_START)})
    status.write({"status": "complete", "config": name, "stats": stats})
    print(json.dumps({"config": name, "stats": stats}, indent=2, default=str))


if __name__ == "__main__":
    main()
