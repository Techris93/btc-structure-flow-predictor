"""Proxy 1m kline replay cannot satisfy the live two-venue independent gate."""

import pandas as pd

from btc_predictor.backtest import run_event_backtest
from btc_predictor.footprint import footprint_confirmation
from btc_predictor.research import predictor_for_replay, proxy_trades, trades_have_two_venues
from btc_predictor.strategy import Predictor


def _klines(n=120):
    idx = pd.date_range("2025-07-19", periods=n, freq="min", tz="UTC")
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 10.0,
            "taker_buy_volume": 6.0,
        },
        index=idx,
    )


def test_proxy_trades_are_not_two_venue():
    trades = proxy_trades(_klines())
    assert "exchange" not in trades.columns
    assert trades_have_two_venues(trades) is False


def test_independent_gate_cannot_confirm_proxy_trades():
    bars = _klines()
    trades = proxy_trades(bars)
    confirmed, details = footprint_confirmation(
        trades,
        bars,
        "bullish",
        bars.index[-5],
        bars.index[-1],
        gate_mode="independent",
    )
    assert confirmed is False
    assert details["reason"] == "two_venue_flow_unavailable"
    assert details["raw_footprint_eligible"] is False


def test_predictor_for_replay_uses_shadow_on_proxy():
    pred = predictor_for_replay(proxy_trades(_klines()))
    assert pred.flow_gate_mode == "shadow"


def test_backtest_adapts_independent_predictor_on_proxy():
    bars = _klines(200)
    trades = proxy_trades(bars)
    predictor = Predictor(flow_gate_mode="independent", cache_closed_frames=True)
    _, stats = run_event_backtest(bars, trades, predictor=predictor, mode="reactive", fee_bps=0, slippage_bps=0)
    assert stats["flow_gate_adapt"] == "shadow_because_two_venue_flow_unavailable"
    assert stats["flow_gate_mode"] == "shadow"
    # Must not silently record only two_venue / orderflow rejects as the whole book.
    assert predictor.flow_gate_mode == "shadow"
