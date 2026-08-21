# BTC Structure Flow Predictor (no liquidation feed)

This repository implements a causal MVP for a Bitcoin trade-setup predictor:

`4h/1h BOS-CHoCH bias → projected price/volume liquidity zones → sweep → 1m order-flow confirmation → structural/ATR risk → walk-forward replay`

It deliberately does not use a liquidation heatmap. Zones are derived from confirmed swings and can be extended with session extremes, volume profile, VWAP, open interest and funding.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Run a small replay:

```python
from btc_predictor.synthetic import make_synthetic
from btc_predictor.backtest import run_event_backtest

ohlc, trades = make_synthetic()
ledger, stats = run_event_backtest(ohlc, trades)
print(stats)
```

## Causality rules

- Pivots become usable only after their right-side confirmation bars.
- Zones carry `created_at`, `available_at`, `expires_at` and `swept_at`.
- Backtest features use only events whose timestamps are at or before the decision bar.
- Walk-forward splits keep test bars after the training window.
- The live connectors are adapters only; raw events should be persisted append-only and replayed deterministically.

This remains research-only: it does not place live exchange orders.

### Live paper governance (no long backtest required)

Paper accounting matches research defaults: **fee 5 bps + slippage 2 bps**. Dashboard and `/api/paper/economics` report **gross vs approx-net** — do not treat gross as alpha.

| Control | Behavior |
|---------|----------|
| Decision snapshots | Full geometry/flow/regime fields on each confirmed setup (`decision_snapshots.json`, closed trades) |
| Fail closed | No new paper entries unless `MARKET_TYPE=linear` and Binance+Bybit futures feeds are fresh (not spot/mixed/stale) |
| Risk | 0.25% risk; max notional **1.0×** equity; daily −2R / weekly −4R; one open unit |
| Exits | **1% stop / 2% target** of Bitcoin price from the fill (2R). Exact percents — no $100 magnet nudge. |
| Fill gate | Next-open fill is cancelled if RR &lt; 1.5 after rebase |
| Time / cooldown | Flatten at **12h** if SL/TP not hit; no same-side re-entry for **8h** after a stop |
| Retrace | Deep sweeps (`RETRACE_ENTRY_ATR=1.2`) use a limit pullback instead of chasing next open |
| Soft filters | Unproven (hero RR, major magnet, same-side cooldown); `untested_breakout` is Book B shadow-only |
| Probability | Heuristic **display/log only** — never used for sizing or hard lifecycle ranking |
| Retune discipline | No parameter changes until ≥40 closed paper trades or 90 days; review metrics only |
| Funnel diary | Weekly counters at `/api/funnel` |
| Shadow book B | Forward-only extra skip (`skip_untested_breakout` by default) at `/api/shadow` |
| Calibration read-only | `/api/calibration` — stay on `independent` unless an artifact already passed promotion |

Policy API: `GET /api/policy`. Seeded three-trade rescore: `GET /api/paper/rescore` and `outputs/seeded_trade_rescore.json`.

## Live feature contract

- 4h and 1h completed candles establish regime; disagreement is neutral.
- A reversal of the held regime requires an opposing confirmed CHoCH.
- 15m completed candles produce swing/equal-level, prior day/week, session, breakout, volume-profile and VWAP zones plus ATR risk.
- ATR-bounded sweeps may reclaim over a declared 60-minute closed-1m window; a 15m zone invalidation does not truncate that already-open reclaim window.
- Live strategy and lifecycle decisions are immutable after the first evaluation of each unique closed 1m bar; late trades are incorporated only in the next bar's decision.
- A durable SQLite buffer receives Binance and Bybit WebSocket trades; 1m taker-buy bars provide flow baselines.
- Market flow and raw trade-level footprint are reported independently and both must clear their configured thresholds for a paper entry. The legacy composite is diagnostic-only. This mode is labelled `independent`, not calibrated, because its 0.40/0.40 thresholds are configured rather than validated.
- Flow is observed provisionally from the closed breach bar and frozen on the closed reclaim bar; provisional observations cannot create entries.
- Session CVD is persisted for the existing Asia, London and New York UTC windows and remains diagnostic-only.
- Every projected zone has immutable identity plus creation, availability, expiry, touch, sweep, and invalidation state.

## Causal research

The Flask app does not execute backtests. Run the durable worker separately:

```bash
python research_worker.py --start 2025-07-19 --end 2026-07-19 --data-dir work/runtime/research
```

The two-venue Phase 2 calibration is also local-only. It downloads official
Binance and Bybit Futures trade archives and validates complete daily coverage.
Its result is retained as research evidence; the configured `independent` gate
does not claim that its thresholds are calibrated:

```bash
python -m pip install -e '.[research]'
python flow_calibration_worker.py \
  --ohlcv work/runtime/research-local/binance_1m.csv \
  --data-dir work/runtime/flow-calibration
```

To use a passed artifact instead, copy `flow_calibration.json` to the configured
runtime path and set `FLOW_GATE_MODE=calibrated`. A missing or failed artifact
requested as `calibrated` falls back to `shadow`; `independent` does not require
an artifact.

It downloads Binance Futures 1m candles (including taker-buy volume), derives completed 15m/1h/4h candles, and compares `reactive` and `mtf` variants over identical dates. Dataset pages, replay state, ledgers, and final results are persisted. Resume identity includes Git revision, configuration, date range, and dataset hash.

Execution is decision-after-close and next-open with explicit fees/slippage. Same-1m-bar stop/target ambiguity defaults to the documented conservative stop-first policy and is counted. Results include trades, wins, losses, win rate, profit factor, net P&L, maximum drawdown, average R, average holding time, forced end-of-data closes, and rejection counts.

## Operations

`BTC_DATA_DIR` must point to persistent storage in production. VAPID private keys, subscriptions, notification deduplication state, and live state are persisted there. An OS file lock permits only one live polling loop. Push tests require a subscription-scoped HMAC token; `/predict` requires `ADMIN_API_TOKEN`; the web backtest POST is disabled.

Run verification with `pytest -q`.

## Local Mac (Binance + Bybit futures)

Both local and Render now default to `MARKET_TYPE=linear` so Binance Futures and Bybit linear stay aligned.
Binance REST remains off by default; live Binance trades come from one long-lived futures WebSocket.
If a temporary futures IP ban reappears on Render, the service stays healthy in degraded mode and retries with backoff.

```bash
cd /path/to/btc-structure-flow-predictor
cp .env.example .env   # already created if missing
./scripts/run_local.sh
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Local defaults:

- `MARKET_TYPE=linear` — Binance Futures + Bybit linear
- `BINANCE_REST_ENABLED=0` — live Binance trades come from WebSocket only
- Bybit REST is used for OHLCV structure candles only
- optional Binance REST backfill is rare and small if you enable it:
  - every `BINANCE_REST_MINUTES` (default 15)
  - at most `BINANCE_TRADE_LIMIT` trades (default 100)
  - at most `BINANCE_FLOW_LIMIT` 1m bars (default 60)

Check health:

```bash
curl -s http://127.0.0.1:8000/healthz | python3 -m json.tool
# Detailed feed, collector, push, and watchdog diagnostics:
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

You want:

- `market_type: "linear"`
- collectors.binance.mode: `"linear"`
- collectors.bybit.mode: `"linear"`

Render uses the same careful futures posture as local:
- `MARKET_TYPE=linear`
- `BINANCE_REST_ENABLED=0`
- one long-lived Binance futures WebSocket with reconnect backoff

If futures endpoints are temporarily blocked, check `/health` for collector errors and wait for the ban to clear rather than enabling REST spam.

### Binance Futures rate-limit policy (local)

Local futures mode is deliberately gentle:

- one long-lived Binance Futures WebSocket (`fstream...@aggTrade`)
- no reconnect storm: quiet periods do not force reconnects
- exponential reconnect backoff (`15s` → max `120s`)
- `BINANCE_REST_ENABLED=0` by default, so `fapi` is not polled
- optional REST backfill, if enabled, is rare and small (`15 min`, `100` trades / `60` bars)

Do not enable frequent Binance REST polling from a cloud IP.

### Binance futures data path (adaptive, within limits)

Budget check (Render + local): `fapi` weight limit is 2,400/min; WS is free up to
300 connects/5 min and 1,024 streams/connection.

- Preferred: one WebSocket with `aggTrade + kline_1m + markPrice@1s` (3 streams on 1 connection).
- When the WS stream is fresh: zero Binance REST calls.
- When the WS stream is stale (>90s): rare REST backfill only.
  - `aggTrades limit=100` → weight ~20 per call, every 3 min ≈ 7 weight/min
  - `klines limit=60` → weight ~2 per call, every 3 min ≈ 0.7 weight/min
  - total ≈ 8 weight/min (~0.3% of the 2,400/min budget)
- On REST error (418/429): cooldown doubles to >=30 min, never retries hot.

Health exposes `binance_data_path`: `websocket` | `rest_backfill` | `stale`.
