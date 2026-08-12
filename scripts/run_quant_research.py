#!/usr/bin/env python3
"""One-shot quant research: smoke → full-year proxy backtest → 60d flow calibration.

Uses the existing local OHLCV CSV (no multi-day download). Designed so a failed
phase does not wipe earlier artifacts. Checkpoints every 1000 bars.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc_predictor.backtest import run_event_backtest
from btc_predictor.flow_calibration import (
    PRICE_BUCKETS,
    aggregate_trade_partitions,
    enumerate_signal_events,
    load_ohlcv,
    run_calibration,
)
from btc_predictor.historical_trades import common_complete_days
from btc_predictor.persistence import JsonStore
from btc_predictor.research import predictor_for_replay, proxy_trades


def _write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    JsonStore(path).write(payload)


def load_bars(csv_path: Path, start=None, end=None) -> pd.DataFrame:
    bars = load_ohlcv(csv_path)
    if start is not None:
        bars = bars.loc[bars.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        bars = bars.loc[bars.index < pd.Timestamp(end, tz="UTC")]
    if bars.empty:
        raise ValueError(f"No bars in range for {csv_path}")
    return bars


def run_backtest_mode(
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    mode: str,
    out_dir: Path,
    *,
    fee_bps=5,
    slippage_bps=2,
    decision_stride=1,
    label="full",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{label}-{mode}-checkpoint.json"
    ledger_path = out_dir / f"{label}-{mode}-ledger.csv"
    stats_path = out_dir / f"{label}-{mode}-stats.json"
    status_path = out_dir / f"{label}-{mode}-status.json"
    checkpoint = JsonStore(checkpoint_path)
    saved = checkpoint.read({})
    resume = saved.get("resume_state") if saved.get("bars") == len(bars) else None
    if saved.get("complete") and saved.get("bars") == len(bars) and stats_path.exists():
        print(f"[skip] {label}/{mode} already complete", flush=True)
        return JsonStore(stats_path).read({})

    t0 = time.time()
    last_log = t0

    def progress(done, total, records):
        nonlocal last_log
        now = time.time()
        if now - last_log < 10 and done < total - 1 and done % 5000 != 0:
            return
        last_log = now
        rate = done / max(now - t0, 1e-6)
        eta = (total - done) / max(rate, 1e-6)
        payload = {
            "status": "running",
            "mode": mode,
            "label": label,
            "bars_processed": done,
            "total_bars": total,
            "closed_trades": len(records),
            "bars_per_sec": round(rate, 2),
            "eta_minutes": round(eta / 60, 1),
        }
        _write(status_path, payload)
        print(
            f"[{label}/{mode}] {done}/{total} bars ({100*done/total:.1f}%) "
            f"trades={len(records)} rate={rate:.1f}/s eta={eta/60:.1f}m",
            flush=True,
        )

    def save_resume(state):
        checkpoint.write({
            "complete": False,
            "bars": len(bars),
            "resume_state": state,
            "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        })

    predictor = predictor_for_replay(trades)
    ledger, stats = run_event_backtest(
        bars,
        trades,
        predictor=predictor,
        mode=mode,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        decision_stride=decision_stride,
        progress=progress,
        resume_state=resume,
        checkpoint=save_resume,
    )
    ledger.to_csv(ledger_path, index=False)
    stats = {
        **stats,
        "mode": mode,
        "label": label,
        "bars": len(bars),
        "decision_stride": decision_stride,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "elapsed_seconds": round(time.time() - t0, 1),
        "start": str(bars.index[0]),
        "end": str(bars.index[-1]),
        "rejection_top": dict(
            sorted((stats.get("rejection_counts") or {}).items(), key=lambda kv: -kv[1])[:15]
        ),
    }
    _write(stats_path, stats)
    checkpoint.write({"complete": True, "bars": len(bars), "stats_path": str(stats_path)})
    _write(status_path, {"status": "complete", **{k: stats[k] for k in ("trades", "net_pnl", "average_r", "profit_factor", "elapsed_seconds") if k in stats}})
    print(f"[done] {label}/{mode} trades={stats.get('trades')} net={stats.get('net_pnl')} avg_r={stats.get('average_r')} pf={stats.get('profit_factor')}", flush=True)
    return stats


def smoke(bars_csv: Path, out_dir: Path, days: int = 7):
    bars_all = load_ohlcv(bars_csv)
    end = bars_all.index[-1].floor("D") + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days)
    # include 14d warmup for structure
    window = bars_all.loc[bars_all.index >= start - pd.Timedelta(days=14)]
    window = window.loc[window.index < end]
    trades = proxy_trades(window)
    print(f"[smoke] bars={len(window)} range={window.index[0]} → {window.index[-1]}", flush=True)
    t0 = time.time()
    stats = run_backtest_mode(window, trades, "mtf", out_dir / "smoke", decision_stride=1, label="smoke")
    elapsed = time.time() - t0
    bars_per_sec = len(window) / max(elapsed, 1e-6)
    full_bars = len(bars_all)
    est_hours = (full_bars / bars_per_sec) / 3600 * 2  # reactive+mtf
    report = {
        "smoke_elapsed_s": round(elapsed, 1),
        "smoke_bars": len(window),
        "bars_per_sec": round(bars_per_sec, 2),
        "full_year_bars": full_bars,
        "est_full_both_modes_hours": round(est_hours, 2),
        "smoke_stats": stats,
    }
    _write(out_dir / "smoke_timing.json", report)
    print(f"[smoke] {bars_per_sec:.1f} bars/s → est full year both modes ~{est_hours:.1f}h", flush=True)
    return report


def full_year(bars_csv: Path, out_dir: Path, modes=("reactive", "mtf"), decision_stride=1):
    bars = load_ohlcv(bars_csv)
    trades = proxy_trades(bars)
    print(f"[full] bars={len(bars)} trades_proxy={len(trades)}", flush=True)
    results = {}
    for mode in modes:
        results[mode] = run_backtest_mode(
            bars, trades, mode, out_dir / "full_year",
            decision_stride=decision_stride, label="year",
        )
    summary = {
        "status": "complete",
        "bars": len(bars),
        "start": str(bars.index[0]),
        "end": str(bars.index[-1]),
        "data_source": "proxy_trades_from_binance_1m_taker_buy",
        "note": "Proxy trades approximate flow from kline taker-buy; not tick-level dual-venue flow.",
        "variants": results,
        "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    _write(out_dir / "full_year" / "result.json", summary)
    return summary


def flow_calibrate(bars_csv: Path, trade_root: Path, out_dir: Path, workers: int = 4):
    """60-day two-venue calibration using existing parquet partitions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    status = JsonStore(out_dir / "status.json")
    artifact_path = out_dir / "flow_calibration.json"
    if artifact_path.exists():
        prior = JsonStore(artifact_path).read({})
        if prior.get("status") == "complete":
            print("[cal] artifact already complete", flush=True)
            return prior

    candles = load_ohlcv(bars_csv)
    # Align to complete trade history window (last 60 complete days).
    complete = common_complete_days(trade_root, candles.index.min(), candles.index.max() + pd.Timedelta(days=1))
    if len(complete) < 60:
        # try listing from partitions
        days = sorted(
            p.name.replace("date=", "")
            for p in (trade_root / "binance").glob("date=*")
            if (trade_root / "binance" / p.name / "trades.parquet").exists()
            and (trade_root / "bybit" / p.name / "trades.parquet").exists()
        )
        complete = days
    if len(complete) < 60:
        raise RuntimeError(f"Need 60 complete dual-venue days, found {len(complete)}")
    complete = complete[-60:]
    start = pd.Timestamp(complete[0], tz="UTC")
    end = pd.Timestamp(complete[-1], tz="UTC") + pd.Timedelta(days=1)
    print(f"[cal] window {start.date()} → {end.date()} days={len(complete)} workers={workers}", flush=True)

    status.write({"status": "running", "phase": "enumerating_causal_setups", "start": start.isoformat(), "end": end.isoformat()})
    events_path = out_dir / "causal_events.csv"
    if events_path.exists() and events_path.stat().st_size > 100:
        events = pd.read_csv(events_path, parse_dates=["decision_time", "sweep_time", "reclaim_time"])
        print(f"[cal] loaded events={len(events)}", flush=True)
    else:
        t0 = time.time()

        def event_progress(done, total, n_events):
            if done % 2000 == 0 or done == total:
                print(f"[cal/events] {done}/{total} bars events={n_events}", flush=True)
                status.write({
                    "status": "running", "phase": "enumerating_causal_setups",
                    "bars": done, "total_bars": total, "events": n_events,
                })

        events = enumerate_signal_events(candles, start, end, progress=event_progress, workers=workers)
        events.to_csv(events_path, index=False)
        print(f"[cal] enumerated events={len(events)} in {time.time()-t0:.0f}s", flush=True)

    aggregates = {}
    for index, bucket in enumerate(PRICE_BUCKETS, 1):
        cache = out_dir / f"agg_bucket_{int(bucket)}.parquet"
        status.write({
            "status": "running", "phase": "aggregating_raw_footprint",
            "bucket": bucket, "bucket_index": index, "bucket_total": len(PRICE_BUCKETS),
        })
        if cache.exists():
            aggregates[bucket] = pd.read_parquet(cache)
            print(f"[cal] loaded aggregates bucket={bucket} rows={len(aggregates[bucket])}", flush=True)
        else:
            t0 = time.time()
            print(f"[cal] aggregating bucket={bucket} …", flush=True)
            aggregates[bucket] = aggregate_trade_partitions(trade_root, complete, bucket)
            aggregates[bucket].to_parquet(cache, index=False)
            print(f"[cal] bucket={bucket} rows={len(aggregates[bucket])} in {time.time()-t0:.0f}s", flush=True)

    # synthetic manifests for identity hash
    manifests = []
    for day in complete:
        for exchange in ("binance", "bybit"):
            manifests.append({
                "exchange": exchange,
                "date": day,
                "archive_sha256": "local-partition",
            })

    status.write({"status": "running", "phase": "walk_forward_selection", "events": len(events)})
    print(f"[cal] walk-forward grid selection on {len(events)} events …", flush=True)
    t0 = time.time()
    artifact = run_calibration(
        candles, events, aggregates, manifests, start, end, artifact_path,
    )
    artifact["elapsed_seconds"] = round(time.time() - t0, 1)
    artifact["complete_days"] = complete
    status.write(artifact)
    print(
        f"[cal] done promotion_passed={artifact.get('promotion_passed')} "
        f"selected={artifact.get('selected_config')} failures={artifact.get('promotion_failures')}",
        flush=True,
    )
    return artifact


def analyze(out_dir: Path) -> dict:
    """Summarize bottlenecks, mismatches, and promotion readiness."""
    report = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "paths": {},
        "full_year": {},
        "calibration": {},
        "mismatches": [],
        "bottlenecks": [],
        "recommendations": [],
    }
    year_result = out_dir / "full_year" / "result.json"
    if year_result.exists():
        year = JsonStore(year_result).read({})
        report["full_year"] = year
        for mode, stats in (year.get("variants") or {}).items():
            rej = stats.get("rejection_top") or stats.get("rejection_counts") or {}
            report["paths"][f"year_{mode}"] = {
                "trades": stats.get("trades"),
                "net_pnl": stats.get("net_pnl"),
                "average_r": stats.get("average_r"),
                "profit_factor": stats.get("profit_factor"),
                "win_rate": stats.get("win_rate"),
                "max_dd": stats.get("maximum_drawdown"),
                "top_rejections": dict(list(rej.items())[:8]) if isinstance(rej, dict) else rej,
            }
    cal_path = out_dir / "flow_calibration" / "flow_calibration.json"
    if cal_path.exists():
        cal = JsonStore(cal_path).read({})
        report["calibration"] = {
            "promotion_passed": cal.get("promotion_passed"),
            "promotion_failures": cal.get("promotion_failures"),
            "selected_config": cal.get("selected_config"),
            "holdout": cal.get("holdout"),
            "top_candidates": len(cal.get("top_candidates") or []),
        }
        if not cal.get("promotion_passed"):
            report["recommendations"].append(
                "Keep FLOW_GATE_MODE=independent; do not promote calibrated gate."
            )
        else:
            report["recommendations"].append(
                "Calibration promotion_passed=true — review holdout then copy artifact to runtime."
            )

    # Known architecture mismatches
    report["mismatches"].extend([
        {
            "id": "proxy_vs_tick_flow",
            "severity": "high",
            "detail": (
                "Full-year research uses proxy trades from 1m taker-buy volume. "
                "Live paper uses dual-venue tick trades + independent flow gate. "
                "Year stats measure structure/RR geometry more than live flow quality."
            ),
        },
        {
            "id": "calibration_window_vs_year",
            "severity": "medium",
            "detail": (
                "Flow calibration is locked to 60 dual-venue days of tick archives. "
                "It does not calibrate the full year and cannot alone prove multi-regime edge."
            ),
        },
        {
            "id": "heuristic_probability",
            "severity": "high",
            "detail": (
                "probability_tp_before_sl remains heuristic unless a reliability model is fit "
                "from a large labeled ledger (year trades or live snapshots)."
            ),
        },
        {
            "id": "soft_filters_unvalidated",
            "severity": "medium",
            "detail": (
                "Live soft filters (RR cap, major magnet, wide breakout) are postmortem heuristics "
                "and are not applied inside the historical research worker unless separately tested."
            ),
        },
    ])
    report["bottlenecks"].extend([
        "Event enumeration (MTF predict per bar) dominates calibration wall time.",
        "Raw footprint aggregation over 60d dual-venue parquet is I/O heavy (cached after first run).",
        "Calibration grid is large (~1296 configs × 4 folds); selection is CPU-bound on events.",
        "Full-year bar loop is O(bars); resume checkpoints every 1000 bars.",
    ])
    _write(out_dir / "quant_research_report.json", report)
    # Markdown summary
    lines = [
        "# Quant research report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Full-year proxy backtest",
    ]
    for mode, path in report["paths"].items():
        lines.append(f"### {mode}")
        lines.append(f"- trades: {path.get('trades')}")
        lines.append(f"- net_pnl: {path.get('net_pnl')}")
        lines.append(f"- average_r: {path.get('average_r')}")
        lines.append(f"- profit_factor: {path.get('profit_factor')}")
        lines.append(f"- win_rate: {path.get('win_rate')}")
        lines.append(f"- max_dd: {path.get('max_dd')}")
        lines.append(f"- top_rejections: `{path.get('top_rejections')}`")
        lines.append("")
    lines.append("## Flow calibration (60d tick)")
    lines.append(f"- promotion_passed: {report['calibration'].get('promotion_passed')}")
    lines.append(f"- failures: {report['calibration'].get('promotion_failures')}")
    lines.append(f"- selected: `{report['calibration'].get('selected_config')}`")
    lines.append(f"- holdout: `{report['calibration'].get('holdout')}`")
    lines.append("")
    lines.append("## Mismatches")
    for m in report["mismatches"]:
        lines.append(f"- **{m['id']}** ({m['severity']}): {m['detail']}")
    lines.append("")
    lines.append("## Bottlenecks")
    for b in report["bottlenecks"]:
        lines.append(f"- {b}")
    lines.append("")
    lines.append("## Recommendations")
    for r in report["recommendations"] or ["See mismatches; keep live governance frozen until promotion criteria clear."]:
        lines.append(f"- {r}")
    (out_dir / "quant_research_report.md").write_text("\n".join(lines) + "\n")
    print((out_dir / "quant_research_report.md").read_text())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv", default="work/runtime/research-local/binance_1m.csv")
    parser.add_argument("--trade-root", default="work/runtime/flow-calibration/trades")
    parser.add_argument("--out", default="work/runtime/quant-research")
    parser.add_argument("--phase", choices=["all", "smoke", "year", "cal", "analyze"], default="all")
    parser.add_argument("--smoke-days", type=int, default=7)
    parser.add_argument("--workers", type=int, default=min(4, __import__("os").cpu_count() or 1))
    parser.add_argument("--decision-stride", type=int, default=1)
    parser.add_argument("--skip-reactive", action="store_true")
    args = parser.parse_args()

    ohlcv = Path(args.ohlcv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trade_root = Path(args.trade_root)

    try:
        if args.phase in ("all", "smoke"):
            smoke(ohlcv, out, days=args.smoke_days)
        if args.phase in ("all", "year"):
            modes = ("mtf",) if args.skip_reactive else ("reactive", "mtf")
            full_year(ohlcv, out, modes=modes, decision_stride=args.decision_stride)
        if args.phase in ("all", "cal"):
            flow_calibrate(ohlcv, trade_root, out / "flow_calibration", workers=args.workers)
        if args.phase in ("all", "analyze"):
            analyze(out)
        print("[ok] phases complete", flush=True)
        return 0
    except Exception as exc:
        err = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        _write(out / "error.json", err)
        print(json.dumps(err, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
