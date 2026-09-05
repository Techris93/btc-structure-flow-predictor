import pandas as pd
import pytest

from btc_predictor.models import Zone
from btc_predictor.strategy import Predictor, LIQUIDITY_SWEEP_ZONES
from btc_predictor.live_policy import evaluate_soft_filters


def _make_test_data(count=100, price=100.0):
    idx = pd.date_range("2026-01-01 00:00", periods=count, freq="min", tz="UTC")
    ohlc = pd.DataFrame(
        {
            "open": price,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price,
            "volume": 10.0,
        },
        index=idx,
    )
    trades = pd.DataFrame(
        {
            "time": idx,
            "price": price,
            "qty": 1.0,
            "side": "buy",
        }
    )
    return ohlc, trades, idx


def test_liquidity_sweep_zones_cannot_be_continuation_breakouts(monkeypatch):
    """Session/range extremes (london_high, pdh, etc.) are strictly excluded from continuation zones."""
    ohlc, trades, idx = _make_test_data(100, 100.0)
    frames = {"15m": ohlc.iloc[::15], "1h": ohlc.iloc[-40:], "4h": ohlc.iloc[-40:]}

    # Zone is a london_high above current price
    lh_zone = Zone("lh1", "london_high", "above", 102.0, 103.0, 1.0, idx[0], idx[1])

    monkeypatch.setattr(Predictor, "_regime_bias", lambda self, frames: "bullish")
    monkeypatch.setattr("btc_predictor.strategy.atr", lambda frame: pd.Series(2.0, index=frame.index))
    # Make sure ADX is high so trend gating passes
    monkeypatch.setattr("btc_predictor.strategy.adx", lambda frame: pd.Series(35.0, index=frame.index))
    monkeypatch.setattr("btc_predictor.strategy.build_projected_zones", lambda frame: [lh_zone])

    predictor = Predictor(min_adx_continuation=25.0)
    # The london_high zone is above price in a bullish bias, which previously was evaluated as a continuation zone.
    # Now it must be rejected because london_high is in LIQUIDITY_SWEEP_ZONES.
    output = predictor.predict(ohlc, trades, frames=frames)
    assert output.no_trade_reason == "no_projected_zone"


def test_reversal_priority_over_continuation(monkeypatch):
    """When both a reversal setup and a continuation setup are confirmed, the engine picks the reversal."""
    ohlc, trades, idx = _make_test_data(100, 100.0)
    frames = {"15m": ohlc.iloc[::15], "1h": ohlc.iloc[-40:], "4h": ohlc.iloc[-40:]}

    rev_zone = Zone("rev_zone", "swing", "below", 95.0, 96.0, 2.0, idx[0], idx[1])
    cont_zone = Zone("cont_zone", "untested_breakout", "above", 101.0, 102.0, 2.0, idx[0], idx[1])
    target_zone = Zone("target_zone", "swing", "above", 115.0, 116.0, 1.0, idx[0], idx[1])

    monkeypatch.setattr(Predictor, "_regime_bias", lambda self, frames: "bullish")
    monkeypatch.setattr("btc_predictor.strategy.atr", lambda frame: pd.Series(2.0, index=frame.index))
    monkeypatch.setattr("btc_predictor.strategy.adx", lambda frame: pd.Series(30.0, index=frame.index))
    monkeypatch.setattr("btc_predictor.strategy.build_projected_zones", lambda frame: [rev_zone, cont_zone, target_zone])

    # Both detect_sweep and detect_continuation return confirmed setups
    monkeypatch.setattr(
        "btc_predictor.strategy.detect_sweep",
        lambda *args, **kwargs: {
            "confirmed": True,
            "status": "confirmed",
            "depth_atr": 1.0,
            "extreme": 94.0,
            "time": idx[-2],
            "reclaim_time": idx[-1],
        },
    )
    monkeypatch.setattr(
        "btc_predictor.strategy.detect_continuation",
        lambda *args, **kwargs: {
            "confirmed": True,
            "status": "confirmed",
            "extreme": 101.5,
            "time": idx[-2],
            "reclaim_time": idx[-1],
        },
    )
    monkeypatch.setattr(
        "btc_predictor.strategy.footprint_confirmation",
        lambda *args, **kwargs: (True, {"reason": "confirmed", "agreement": True, "score": 0.8}),
    )

    predictor = Predictor(use_fixed_pct_exits=False, min_adx_continuation=25.0)
    output = predictor.predict(ohlc, trades, frames=frames)

    assert output.setup_type == "reversal"
    assert output.zone == "rev_zone"


def test_trend_gating_blocks_continuation_in_ranging_market(monkeypatch):
    """In a rangebound market (ADX < 25), continuation setups are ignored."""
    ohlc, trades, idx = _make_test_data(100, 100.0)
    frames = {"15m": ohlc.iloc[::15], "1h": ohlc.iloc[-40:], "4h": ohlc.iloc[-40:]}

    cont_zone = Zone("cont_zone", "untested_breakout", "above", 101.0, 102.0, 2.0, idx[0], idx[1])

    monkeypatch.setattr(Predictor, "_regime_bias", lambda self, frames: "bullish")
    monkeypatch.setattr("btc_predictor.strategy.atr", lambda frame: pd.Series(2.0, index=frame.index))
    # Weak ADX (rangebound)
    monkeypatch.setattr("btc_predictor.strategy.adx", lambda frame: pd.Series(16.0, index=frame.index))
    monkeypatch.setattr("btc_predictor.strategy.build_projected_zones", lambda frame: [cont_zone])

    predictor = Predictor(min_adx_continuation=25.0)
    output = predictor.predict(ohlc, trades, frames=frames)
    assert output.no_trade_reason == "no_projected_zone"


def test_structural_atr_exits_geometry(monkeypatch):
    """Structural ATR exits anchor stop beyond the sweep extreme with min 1.5 ATR buffer and >= 2.0 RR."""
    ohlc, trades, idx = _make_test_data(100, 100.0)
    frames = {"15m": ohlc.iloc[::15], "1h": ohlc.iloc[-40:], "4h": ohlc.iloc[-40:]}

    rev_zone = Zone("rev1", "swing", "below", 95.0, 96.0, 2.0, idx[0], idx[1])
    target_zone = Zone("tgt1", "swing", "above", 116.0, 117.0, 1.0, idx[0], idx[1])

    monkeypatch.setattr(Predictor, "_regime_bias", lambda self, frames: "bullish")
    monkeypatch.setattr("btc_predictor.strategy.atr", lambda frame: pd.Series(2.0, index=frame.index))
    monkeypatch.setattr("btc_predictor.strategy.build_projected_zones", lambda frame: [rev_zone, target_zone])
    monkeypatch.setattr(
        "btc_predictor.strategy.detect_sweep",
        lambda *args, **kwargs: {
            "confirmed": True,
            "status": "confirmed",
            "depth_atr": 1.0,
            "extreme": 93.0,
            "time": idx[-2],
            "reclaim_time": idx[-1],
        },
    )
    monkeypatch.setattr(
        "btc_predictor.strategy.footprint_confirmation",
        lambda *args, **kwargs: (True, {"reason": "confirmed", "agreement": True, "score": 0.8}),
    )

    predictor = Predictor(use_fixed_pct_exits=False, atr_mult=1.5, min_rr=2.0)
    output = predictor.predict(ohlc, trades, frames=frames)

    # Price is 100, ATR is 2.0, sweep extreme is 93.0.
    # stop = min(93.0 - 0.5*2.0, 100 - 1.5*2.0) = min(92.0, 97.0) = 92.0
    assert output.stop == 92.0
    assert output.entry == 100.0
    # Risk = 100 - 92 = 8.0
    # Target zone midpoint is 116.5, RR = (116.5 - 100) / 8 = 2.0625 >= 2.0
    assert output.target == 116.5
    assert output.reward_risk >= 2.0


def test_soft_filters_blocks_continuation_on_liquidity_sweep_zone():
    """live_policy.evaluate_soft_filters hard-skips continuation setups on session/range liquidity zones."""
    for kind in ("london_high", "previous_day_high", "asian_low", "equal_highs", "range_low"):
        snap = {
            "setup_type": "continuation",
            "zone_kind": kind,
            "reward_risk": 2.0,
            "entry": 100.0,
            "stop": 98.0,
            "bias": "bullish",
            "timestamp": "2026-01-01T12:00:00+00:00",
        }
        res = evaluate_soft_filters(snap, enabled=True)
        assert "continuation_on_liquidity_sweep_zone" in res["hard_skips"]
        assert res["allow"] is False
