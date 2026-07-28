"""Out-of-sample validation on the untouched tail of the 30-day dataset.

The stopped 30-day replay analyzed bars 80..39,001 (Jun 28 - Jul 14). Bars from
39,001 onward (Jul 14 - Jul 28) were never seen during analysis, so they form a
natural holdout. This script replays that tail twice: baseline (pre-fix
parameters) and gated (drift gate + cost discipline + zone demotion), writing
comparable stats for each. Both runs checkpoint and resume if interrupted.

Usage: python scripts/oos_validation.py [baseline|gated]
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
OUT = REPO / "outputs/oos_validation"
# First bar of the untouched tail: where the analyzed run was stopped.
TAIL_BAR = 39_001

CONFIGS = {
    # Pre-fix behavior: no drift gate, loose cost gate, no stop floor, no demotion.
    "baseline": dict(
        require_drift_alignment=False,
        max_cost_fraction=0.50,
        min_expectancy_r=0.0,
        min_stop_atr=0.0,
        zone_score_adjustments={},
    ),
    # Post-fix defaults as shipped in strategy.py.
    "gated": dict(),
}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "gated"
    if name not in CONFIGS:
        raise SystemExit(f"unknown config {name!r}; choose from {sorted(CONFIGS)}")
    OUT.mkdir(parents=True, exist_ok=True)
    bars = pd.read_csv(DATASET, parse_dates=["close_time"]).set_index("close_time")
    bars.index = pd.to_datetime(bars.index, utc=True)
    decision_start = bars.index[TAIL_BAR]
    trades = proxy_trades(bars)
    status = JsonStore(OUT / f"{name}-status.json")
    checkpoint_store = JsonStore(OUT / f"{name}-checkpoint.json")
    saved = checkpoint_store.read({})
    resume = saved.get("resume_state") if saved.get("dataset_tail") == str(decision_start) else None

    def save_resume(state):
        checkpoint_store.write({
            "complete": False,
            "dataset_tail": str(decision_start),
            "resume_state": state,
            "bars_processed": state["next_i"],
            "updated_at": pd.Timestamp.utcnow().isoformat(),
        })

    def progress(done, total, ledger):
        status.write({
            "status": "running", "config": name, "bars_processed": done,
            "total_bars": total, "closed_trades": len(ledger),
            "decision_start": str(decision_start),
        })

    ledger, stats = run_event_backtest(
        bars, trades, Predictor(**CONFIGS[name]), mode="reactive",
        fee_bps=5, slippage_bps=2, same_bar_policy="conservative",
        decision_start=decision_start, progress=progress,
        resume_state=resume, checkpoint=save_resume,
    )
    ledger.to_csv(OUT / f"{name}-ledger.csv", index=False)
    result = {
        "status": "complete", "config": name, "predictor_config": CONFIGS[name],
        "decision_start": str(decision_start), "tail_bars": len(bars) - TAIL_BAR,
        "stats": stats, "completed_at": pd.Timestamp.utcnow().isoformat(),
    }
    (OUT / f"{name}-result.json").write_text(json.dumps(result, indent=2, default=str))
    checkpoint_store.write({"complete": True, "dataset_tail": str(decision_start)})
    status.write({"status": "complete", "config": name, "stats": stats})
    print(json.dumps({"config": name, "stats": stats}, indent=2, default=str))


if __name__ == "__main__":
    main()
