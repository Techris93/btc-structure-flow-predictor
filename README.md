# BTC Predictor (no liquidation feed)

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

## Production hardening still required

The current package is an MVP research engine. Before execution, add exchange-specific historical loaders, robust order-book/reference-price handling, venue outage tests, partial-fill simulation, calibrated probability training, and paper-trading monitoring. Do not interpret synthetic replay results as evidence of profitability.
