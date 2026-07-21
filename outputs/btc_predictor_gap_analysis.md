# BTC Structure Flow Predictor — Gap & Issue Analysis

**Date:** 2026-07-21  
**Scope:** web service (`btc-structure-flow-predictor`), research worker, strategy, zones, structure, footprint, and operational deployment.  
**Live URL:** https://btc-structure-flow-predictor.onrender.com

---

## 1. Executive Summary

The service is **live and stable** after the recent memory/startup crash fix. However, several functional gaps prevent it from generating tradeable, validated signals consistently:

| Area | Status | Severity |
|------|--------|----------|
| Render OOM / exit 1 | **Fixed** in commit `99394ca` | — |
| Data feed quality | Binance is on **spot fallback**, not futures | High |
| Timeframe alignment | 15m structure is reported but **not enforced** | High |
| Footprint confirmation | Criteria are **too strict and rarely all true** | High |
| Cross-exchange agreement | Fails if either exchange is quiet | Medium |
| Position / P&L tracking | **Not implemented** in live mode | High |
| Probability calibration | Always `null` | Medium |
| Backtest integration | Web app and worker are disconnected | Medium |
| Zone model | Fixed scores, no true recency, no reclaim validity | Medium |
| Sweep detection | 3-minute reclaim window is very tight | Medium |
| Operational monitoring | No stale-feed alert, no degraded-state alert | Medium |

---

## 2. Resolved Crash Issues

### 2.1 OOM at 512 MB

**Root cause:** The live loop was querying **3 hours of tick trades** into a pandas DataFrame every 30–45 seconds, and the WebSocket collectors were inserting trades as **single-row DataFrames** inside an async loop. With Bybit+Binance combined tick rates, this rapidly exceeded the 512 MB Render starter plan.

**Fix (commit `99394ca`):**
- Buffered WebSocket inserts (`_BufferedAppender` in `src/btc_predictor/trade_store.py`).
- Hard trade-store row cap (`TRADE_STORE_MAX_ROWS=80000`) with enforced pruning.
- Shorter live query window (`TRADE_LOOKBACK_MINUTES=90`, `TRADE_QUERY_LIMIT=60000`).
- Reduced gunicorn threads from 4 to 2 and added worker recycling.

**Evidence:** After deploying, the service stayed live for >7 hours, trade counts stabilized around 40,000 per exchange, and Render logs no longer show `Ran out of memory` or `exited with status 1`.

### 2.2 PermissionError on `/var/data` at startup

**Root cause:** Gunicorn imported `app.py` before the Render disk was mounted at `/var/data`, so `runtime_dir()` crashed with `PermissionError`.

**Fix:** `runtime_dir()` now probes the configured directory and falls back to `work/runtime` or a temp directory if it is not writable, then logs a warning.

---

## 3. Functional Gaps & Issues

### 3.1 Binance is on spot data, not futures

**Why it matters:** The predictor was designed around futures order flow and liquidation sweeps. Spot BTCUSDT has different participants, lower leverage, and different tick dynamics. The 15m zone construction, volume profile, and footprint confirmation are being fed with a mix of Bybit futures and Binance spot.

**What happens:** Binance REST and WebSocket are currently hitting `data-api.binance.vision` because `fapi.binance.com` rejects Render's IP with HTTP 418 / SSL issues. The collector automatically rotates to `spot_market_data`.

**Recommended fix:**
- Use a reliable futures market-data proxy for Binance (e.g., Coinbase, OKX, or a paid data provider) and treat it as `binance` order flow.
- Or, if Binance futures cannot be restored, drop the Binance/Bybit cross-exchange requirement and tune Bybit-only confirmation thresholds.
- Document the data mismatch so backtest results are not compared against a live feed that is half spot.

---

### 3.2 15m structure is reported but not enforced

**Code:** `strategy.py` `_regime_bias` only requires 4h and 1h to agree. The 15m signal is computed and stored in `setup_15m`, but it does **not** block a trade.

**Live evidence:**
```json
{
  "regime_4h": "bullish",
  "regime_1h": "bullish",
  "setup_15m": "bearish",
  "no_trade_reason": "orderflow_not_confirmed"
}
```
This means the predictor can emit a bullish setup even when the 15m structure is bearish. That contradicts the stated design: *4h regime → 1h bias → 15m setup → 1m footprint*.

**Recommended fix:** In `Predictor.predict`, require `setup_15m` to match `bias` (or at least not conflict). If they conflict, return `no_trade_reason="timeframe_conflict"`.

---

### 3.3 Footprint confirmation is too strict

**Code:** `footprint.py` `footprint_confirmation` requires **all five** conditions to be true:
1. `extreme_delta` (delta_z < -1 or sell absorption)
2. `delta_reversal` (bullish/bearish reversal flag)
3. `stalled_response` (price response ≤ median)
4. `cross_exchange` (Binance and Bybit deltas agree)
5. `footprint_imbalance` (buy/sell ratio > 1.2 or < 1/1.2)

**Why this is a problem:** In practice, all five rarely align in the same 1-minute window. The live evidence shows a confirmed sweep failing with `orderflow_reason: "extreme_delta"`, meaning the other four were not satisfied. The result is the dashboard almost always shows `UNCONFIRMED`.

**Recommended fix:**
- Use a scoring/weighting system instead of a hard AND gate.
- Make cross-exchange agreement optional or weighted when one feed is spot.
- Allow confirmation over a slightly wider window (e.g., 3–5 minutes after the sweep).
- Log the per-condition hit rates so you can tune thresholds with real data.

---

### 3.4 Cross-exchange agreement is fragile

**Code:** `cross_exchange_agreement` returns `True` only if both Binance and Bybit have non-zero net delta **and** both match the desired direction.

**Why it fails:** If Binance spot is quiet, or if the two markets have a normal 30–60 second lag, one side can be neutral/negative while the other is positive. This single condition kills the entire confirmation.

**Recommended fix:**
- Weight agreement by notional volume, not just sign.
- Allow partial agreement if the dominant exchange (Bybit futures) agrees and the other is neutral.
- Add a minimum-notional threshold so tiny exchanges do not veto the signal.

---

### 3.5 Sweep detection window is too narrow

**Code:** `detect_sweep` in `strategy.py` uses `ohlc.tail(reclaim_bars+1)` and `reclaim_bars=3`. The sweep must breach and reclaim within **3 closed 1m bars**.

**Why this is a problem:** For a 15m zone, a sweep-and-reclaim can take 5–15 minutes. The current logic misses slower reclaims and marks them as `waiting_reclaim` or `expired_reclaim`.

**Recommended fix:**
- Scale `reclaim_bars` with the zone timeframe, e.g., `reclaim_bars = max(3, zone_timeframe_minutes // 2)`.
- Or, for 15m zones, allow up to 15–30 minutes for the reclaim.
- Track the sweep state across polls so a reclaim that happens between polls is not lost.

---

### 3.6 No live position / P&L tracking

**Current behavior:** The predictor emits `entry`, `stop`, `target`, and `position_size`, but never tracks whether a hypothetical trade was filled, stopped, or hit target.

**Why it matters:** You cannot know the win rate, profit factor, or even whether the signal is valid without recording fills.

**Recommended fix:**
- Maintain a `paper_position` state (entry, stop, target, size, open_time).
- On each poll, check the 1m low/high against the stop and target.
- Record closed trades to a `paper_trades` table and expose stats via `/api/performance`.
- This also lets you compare live signal P&L against the research backtest.

---

### 3.7 `probability_tp_before_sl` is always `null`

**Current behavior:** The `PredictorOutput` dataclass has `probability_tp_before_sl` but it is never computed.

**Recommended fix:**
- After the research backtest produces a ledger, fit a simple model (logistic regression or empirical lookup) from signal features to TP-before-SL outcome.
- Use it to filter low-probability setups and size accordingly.

---

### 3.8 Zone model issues

| Issue | Detail |
|-------|--------|
| Fixed scores | Swing, equal level, session, VWAP scores are hard-coded constants. No recency or touch-weighting. |
| Equal-level tolerance | Uses `price * 0.0007`, ~$85 at $120k BTC. This may cluster too many levels. |
| Touch penalty | Penalty is applied once at construction; heavily-touched zones still remain active. |
| No reclaim validity | Once a zone is swept, it is invalidated forever. A clean reclaim-and-hold should make it valid again. |
| Volume profile | Derived from 193 bars of 15m data (~48h). Reasonable, but only updates when the zone is rebuilt. |

**Recommended fix:**
- Introduce a dynamic score that decays with age and touches.
- Separate `swept_at` from `invalidated_at`; allow a zone to be re-validated if price reclaims and holds.
- Use a stricter equal-level tolerance for higher-price assets.

---

### 3.9 Research worker is disconnected from the web app

**Current behavior:** The web app `/api/backtest/one-year` is disabled; the research worker is a separate process that writes to `/var/data/research`.

**Recommended fix:**
- Have the web app read the worker's `status.json` and `result.json` so the dashboard can show the latest backtest stats.
- Add a webhook or cron trigger so the worker runs automatically.

---

## 4. Operational Issues

| Issue | Risk | Recommended fix |
|-------|------|-------------------|
| No stale-feed alert | Collectors may be "connected" but silent | Add a freshness check: if no trade in 60s, mark collector degraded and alert |
| No degraded-state alert | `live_state.status == "degraded"` is only visible in API | Push notification or email on sustained degraded state |
| Push subscriptions in memory | If multiple gunicorn workers were used, lists would diverge | Currently okay (1 worker), but document the limitation |
| Advisory file lock | A crash can leave `live-loop.lock` stale | Add PID to lock file and stale-lock detection |
| Gunicorn worker recycling | `max-requests` recycles worker, but live thread is daemon; state is persisted to disk, which is fine | Ensure `live_state_store` is read on startup if lock is stale |
| No metrics export | No Prometheus/StatsD; hard to observe | Add a `/metrics` endpoint with signal count, collector lag, memory estimate |

---

## 5. Testing Gaps

The existing tests cover basic mechanics but miss:
- Live-loop integration (network, persistence, state machine).
- Multi-day walk-forward causality.
- Footprint confirmation under realistic mixed-exchange conditions.
- Zone expiry and re-validation.
- Sweep detection with 5–15 minute reclaims.

**Recommended:** Add property-based or fixture-driven tests that run the predictor over a week of synthetic data and assert no look-ahead, correct bias transitions, and at least one confirmed setup.

---

## 6. Prioritized Fix Plan

### High priority (will materially change signal quality)
1. Enforce 15m/4h/1h alignment before emitting a setup.
2. Loosen footprint confirmation to a weighted scoring model.
3. Resolve Binance futures feed or document/adjust for spot data.
4. Implement live paper-position tracking and P&L.

### Medium priority (robustness and honesty)
5. Widen sweep reclaim window for 15m zones.
6. Improve zone scoring (recency, touches, reclaim validity).
7. Compute and expose `probability_tp_before_sl` from research ledger.
8. Connect research worker results to the web dashboard.

### Low priority (operational polish)
9. Add stale-feed and degraded-state alerts.
10. Add `/metrics` endpoint and tests for the live loop.

---

## 7. Files Changed for Crash Fix

- `app.py` — reduced candle limits, shorter trade query window, slower poll, runtime env vars.
- `src/btc_predictor/trade_store.py` — buffered inserts, row cap, better WAL handling.
- `src/btc_predictor/persistence.py` — writable directory fallback.
- `render.yaml` — gunicorn threads, worker recycling, environment variables.
- `tests/test_flow_sweeps_store.py` — added max-rows test.

