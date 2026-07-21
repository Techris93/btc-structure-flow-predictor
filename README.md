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

This remains research-only: it does not place orders, and `probability_tp_before_sl` remains null until out-of-sample calibration exists.

## Live feature contract

- 4h and 1h completed candles establish regime; disagreement is neutral.
- A reversal of the held regime requires an opposing confirmed CHoCH.
- 15m completed candles produce swing/equal-level, prior day/week, session, breakout, volume-profile and VWAP zones plus ATR risk.
- ATR-bounded sweeps may reclaim over multiple closed 1m candles.
- A durable SQLite buffer receives Binance and Bybit WebSocket trades; 1m taker-buy bars provide flow baselines.
- Confirmation requires symmetric delta reversal, absorption/extreme delta, low price response, footprint imbalance, and Binance/Bybit agreement.
- Every projected zone has immutable identity plus creation, availability, expiry, touch, sweep, and invalidation state.

## Causal research

The Flask app does not execute backtests. Run the durable worker separately:

```bash
python research_worker.py --start 2025-07-19 --end 2026-07-19 --data-dir work/runtime/research
```

It downloads Binance Futures 1m candles (including taker-buy volume), derives completed 15m/1h/4h candles, and compares `reactive` and `mtf` variants over identical dates. Dataset pages, replay state, ledgers, and final results are persisted. Resume identity includes Git revision, configuration, date range, and dataset hash.

Execution is decision-after-close and next-open with explicit fees/slippage. Same-1m-bar stop/target ambiguity defaults to the documented conservative stop-first policy and is counted. Results include trades, wins, losses, win rate, profit factor, net P&L, maximum drawdown, average R, average holding time, forced end-of-data closes, and rejection counts.

## Operations

`BTC_DATA_DIR` must point to persistent storage in production. VAPID private keys, subscriptions, notification deduplication state, and live state are persisted there. An OS file lock permits only one live polling loop. Push tests require a subscription-scoped HMAC token; `/predict` requires `ADMIN_API_TOKEN`; the web backtest POST is disabled.

Run verification with `pytest -q`.

## Local Mac (Binance + Bybit futures)

Render cannot reliably reach Binance Futures (`fapi` / `fstream`), so production stays on `MARKET_TYPE=spot`.
On your Mac, run futures mode so both exchanges use linear/futures data.

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
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

You want:

- `market_type: "linear"`
- collectors.binance.mode: `"linear"`
- collectors.bybit.mode: `"linear"`

Keep Render on spot. Do not point a cloud IP at Binance Futures repeatedly.

### Binance Futures rate-limit policy (local)

Local futures mode is deliberately gentle:

- one long-lived Binance Futures WebSocket (`fstream...@aggTrade`)
- no reconnect storm: quiet periods do not force reconnects
- exponential reconnect backoff (`15s` → max `120s`)
- `BINANCE_REST_ENABLED=0` by default, so `fapi` is not polled
- optional REST backfill, if enabled, is rare and small (`15 min`, `100` trades / `60` bars)

Do not enable frequent Binance REST polling from a cloud IP.

