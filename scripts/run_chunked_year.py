#!/usr/bin/env python3
"""Full-year MTF backtest in monthly chunks (fast, checkpointable, no thrash).

Each chunk uses a 14-day causal warmup then evaluates a ~30-day window.
Proxy trades from 1m taker-buy (same as research worker). Fee 5 / slip 2 bps.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc_predictor.backtest import run_event_backtest
from btc_predictor.flow_calibration import load_ohlcv
from btc_predictor.persistence import JsonStore
from btc_predictor.research import predictor_for_replay, proxy_trades

OUT = Path("work/runtime/quant-research/full_year_chunked")
OHLCV = Path("work/runtime/research-local/binance_1m.csv")
WARMUP_DAYS = 14
CHUNK_DAYS = 30


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = JsonStore(OUT / "status.json")
    result_path = OUT / "result.json"
    if result_path.exists():
        prior = JsonStore(result_path).read({})
        if prior.get("status") == "complete":
            print("[skip] chunked year already complete", flush=True)
            print(json.dumps(prior.get("summary"), indent=2, default=str), flush=True)
            return 0

    print("[load] ohlcv", flush=True)
    bars = load_ohlcv(OHLCV)
    start = bars.index[0].floor("D")
    end = bars.index[-1].floor("D") + pd.Timedelta(days=1)
    print(f"[load] bars={len(bars)} {start.date()} → {end.date()}", flush=True)

    chunk_starts = pd.date_range(start, end, freq=f"{CHUNK_DAYS}D")
    ledgers = []
    chunk_stats = []
    t_all = time.time()

    for ci, cstart in enumerate(chunk_starts):
        cend = min(cstart + pd.Timedelta(days=CHUNK_DAYS), end)
        if cstart >= end:
            break
        warm_start = cstart - pd.Timedelta(days=WARMUP_DAYS)
        window = bars.loc[(bars.index >= warm_start) & (bars.index < cend)]
        if len(window) < 500:
            continue
        # Only count trades decided inside the evaluation window (not pure warmup).
        eval_start = max(cstart, window.index[0])
        checkpoint = OUT / f"chunk_{ci:02d}_{cstart.date()}.json"
        if checkpoint.exists():
            saved = JsonStore(checkpoint).read({})
            if saved.get("complete"):
                print(f"[skip] chunk {ci} {cstart.date()}→{cend.date()}", flush=True)
                chunk_stats.append(saved["stats"])
                if saved.get("ledger_path") and Path(saved["ledger_path"]).exists():
                    ledgers.append(pd.read_csv(saved["ledger_path"], parse_dates=["decision_time", "entry_time", "exit_time"]))
                continue

        trades = proxy_trades(window)
        print(
            f"[chunk {ci}/{len(chunk_starts)}] bars={len(window)} "
            f"eval={eval_start.date()}→{cend.date()} trades_proxy={len(trades)}",
            flush=True,
        )
        t0 = time.time()
        last = [t0]

        def progress(done, total, records, _ci=ci):
            now = time.time()
            if now - last[0] < 8 and done < total - 1:
                return
            last[0] = now
            rate = done / max(now - t0, 1e-6)
            print(
                f"  chunk{_ci} {done}/{total} ({100*done/total:.0f}%) "
                f"trades={len(records)} rate={rate:.0f}/s",
                flush=True,
            )
            status.write({
                "status": "running",
                "chunk": _ci,
                "bars_processed": done,
                "total_bars": total,
                "rate": round(rate, 1),
            })

        # stride=15: one decision per 15 closed 1m bars when flat. Keeps causality
        # (still close-time / next-open) while avoiding multi-hour thrash on full 1m.
        ledger, stats = run_event_backtest(
            window,
            trades,
            predictor_for_replay(trades),
            mode="mtf",
            fee_bps=5,
            slippage_bps=2,
            decision_stride=15,
            progress=progress,
            checkpoint=None,  # avoid huge JSON mid-chunk
        )
        # Drop warmup decisions
        if len(ledger):
            if "decision_time" in ledger.columns:
                dt = pd.to_datetime(ledger["decision_time"], utc=True)
            else:
                dt = pd.to_datetime(ledger["entry_time"], utc=True)
            ledger = ledger.loc[dt >= eval_start].copy()
        # Recompute simple stats on eval window only
        if len(ledger):
            wins = int((ledger.pnl > 0).sum())
            losses = int((ledger.pnl <= 0).sum())
            gp = float(ledger.loc[ledger.pnl > 0, "pnl"].sum())
            gl = float(-ledger.loc[ledger.pnl < 0, "pnl"].sum())
            stats_eval = {
                "chunk": ci,
                "eval_start": str(eval_start),
                "eval_end": str(cend),
                "bars": len(window),
                "trades": len(ledger),
                "wins": wins,
                "losses": losses,
                "win_rate": wins / len(ledger) if len(ledger) else 0.0,
                "net_pnl": float(ledger.pnl.sum()),
                "average_r": float(ledger.r_multiple.mean()) if "r_multiple" in ledger else None,
                "profit_factor": (gp / gl) if gl else (float("inf") if gp else None),
                "rejection_counts": stats.get("rejection_counts"),
                "elapsed_seconds": round(time.time() - t0, 1),
                "bars_per_sec": round(len(window) / max(time.time() - t0, 1e-6), 1),
            }
        else:
            stats_eval = {
                "chunk": ci,
                "eval_start": str(eval_start),
                "eval_end": str(cend),
                "bars": len(window),
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "net_pnl": 0.0,
                "average_r": None,
                "profit_factor": None,
                "rejection_counts": stats.get("rejection_counts"),
                "elapsed_seconds": round(time.time() - t0, 1),
                "bars_per_sec": round(len(window) / max(time.time() - t0, 1e-6), 1),
            }
        ledger_path = OUT / f"chunk_{ci:02d}_ledger.csv"
        if len(ledger):
            ledger.to_csv(ledger_path, index=False)
            ledgers.append(ledger)
        else:
            ledger_path = None
        JsonStore(checkpoint).write({
            "complete": True,
            "stats": stats_eval,
            "ledger_path": str(ledger_path) if ledger_path else None,
        })
        chunk_stats.append(stats_eval)
        print(
            f"[done chunk {ci}] trades={stats_eval['trades']} net={stats_eval['net_pnl']:.2f} "
            f"rate={stats_eval['bars_per_sec']}/s rejections_top="
            f"{sorted((stats_eval.get('rejection_counts') or {}).items(), key=lambda x: -x[1])[:4]}",
            flush=True,
        )

    # Merge
    if ledgers:
        all_ledger = pd.concat(ledgers, ignore_index=True)
        all_ledger = all_ledger.sort_values("entry_time" if "entry_time" in all_ledger.columns else "decision_time")
        # sequential equity from 100k
        equity = 100_000.0
        eq = []
        for pnl in all_ledger.pnl.fillna(0):
            equity += float(pnl)
            eq.append(equity)
        all_ledger["equity"] = eq
        all_ledger.to_csv(OUT / "year-mtf-ledger.csv", index=False)
        wins = int((all_ledger.pnl > 0).sum())
        losses = int((all_ledger.pnl <= 0).sum())
        gp = float(all_ledger.loc[all_ledger.pnl > 0, "pnl"].sum())
        gl = float(-all_ledger.loc[all_ledger.pnl < 0, "pnl"].sum())
        curve = pd.Series([100_000.0] + eq)
        dd = float((curve.cummax() - curve).max())
        summary = {
            "trades": len(all_ledger),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(all_ledger) if len(all_ledger) else 0.0,
            "net_pnl": float(all_ledger.pnl.sum()),
            "final_equity": equity,
            "average_r": float(all_ledger.r_multiple.mean()) if "r_multiple" in all_ledger else None,
            "profit_factor": (gp / gl) if gl else None,
            "maximum_drawdown": dd,
            "average_hold_minutes": float(all_ledger.hold_minutes.mean()) if "hold_minutes" in all_ledger else None,
        }
    else:
        summary = {"trades": 0, "net_pnl": 0.0, "note": "no_trades_any_chunk"}

    # Aggregate rejections
    rej = {}
    for st in chunk_stats:
        for k, v in (st.get("rejection_counts") or {}).items():
            rej[k] = rej.get(k, 0) + int(v)

    result = {
        "status": "complete",
        "mode": "mtf",
        "method": "monthly_chunks_14d_warmup_decision_stride_15",
        "decision_stride": 15,
        "fee_bps": 5,
        "slippage_bps": 2,
        "data_source": "proxy_trades_from_binance_1m_taker_buy",
        "bars_total": len(bars),
        "start": str(bars.index[0]),
        "end": str(bars.index[-1]),
        "chunks": chunk_stats,
        "summary": summary,
        "rejection_counts": dict(sorted(rej.items(), key=lambda kv: -kv[1])),
        "elapsed_seconds": round(time.time() - t_all, 1),
        "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "mismatches": [
            "Proxy flow ≠ live dual-venue tick flow",
            "Chunk boundaries reset open positions (no cross-chunk holds)",
        ],
    }
    JsonStore(result_path).write(result)
    status.write({"status": "complete", "summary": summary})
    print(json.dumps(result["summary"], indent=2, default=str), flush=True)
    print("[ok] chunked year complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
