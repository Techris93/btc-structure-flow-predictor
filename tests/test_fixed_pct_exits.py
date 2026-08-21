"""Fixed 1.0% SL / 2.0% TP geometry, fill rebase, magnet push, time stop."""

import pandas as pd
import pytest

from btc_predictor import live_policy
from btc_predictor.live_policy import (
    fill_min_rr_ok,
    fixed_pct_exits,
    push_stop_beyond_hundred,
    shadow_rule_skip,
)
from btc_predictor.models import PredictorOutput
from btc_predictor.paper_position import PaperLedger
from btc_predictor.strategy import Predictor


def test_fixed_pct_is_two_r():
    geo = fixed_pct_exits(64000.0, "long", push_through_100=False)
    assert geo["stop"] == 64000.0 * 0.99
    assert geo["target"] == 64000.0 * 1.02
    assert abs(geo["reward_risk"] - 2.0) < 1e-9


def test_short_fixed_pct():
    geo = fixed_pct_exits(64000.0, "short", push_through_100=False)
    assert geo["stop"] == 64000.0 * 1.01
    assert geo["target"] == 64000.0 * 0.98
    assert abs(geo["reward_risk"] - 2.0) < 1e-9


def test_push_stop_through_hundred_long():
    # 1% of 64157 ≈ 63515; 63600 sits between stop and entry.
    stop, pushed = push_stop_beyond_hundred(64157.0, 63515.0, "long", buffer=1.0)
    assert pushed is True
    assert stop == 63599.0


def test_fill_min_rr_rejects_thin_reward():
    assert fill_min_rr_ok(100.0, 99.5, 101.0, "long", 1.5) is True
    assert fill_min_rr_ok(100.0, 99.5, 100.4, "long", 1.5) is False


def test_predictor_fixed_pct_on_confirmed_setup(monkeypatch):
    idx = pd.date_range("2025-01-01", periods=100, freq="min", tz="UTC")
    o = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0}, index=idx)
    o.iloc[-1, o.columns.get_loc("low")] = 89
    o.iloc[-1, o.columns.get_loc("close")] = 100
    trades = pd.DataFrame({"time": idx, "price": 100.0, "qty": 1.0, "side": "buy"})
    from btc_predictor.models import Zone
    swept = Zone("swept", "swing", "below", 90, 91, 3, idx[1], idx[2])
    monkeypatch.setattr(Predictor, "_regime_bias", lambda self, frames: "bullish")
    monkeypatch.setattr("btc_predictor.strategy.build_projected_zones", lambda frame: [swept])
    monkeypatch.setattr("btc_predictor.strategy.atr", lambda frame: pd.Series(5.0, index=frame.index))
    monkeypatch.setattr(
        "btc_predictor.strategy.footprint_confirmation",
        lambda *a, **k: (True, {"reason": "confirmed", "agreement": True}),
    )
    monkeypatch.setattr(
        "btc_predictor.strategy.detect_sweep",
        lambda *a, **k: {
            "confirmed": True, "status": "confirmed", "depth_atr": 0.4,
            "extreme": 89.0, "time": idx[-2], "reclaim_time": idx[-1],
        },
    )
    result = Predictor(use_fixed_pct_exits=True, stop_pct=0.01, target_pct=0.02).predict(
        o, trades, frames={"15m": o.iloc[::15], "1h": o.iloc[-40:], "4h": o.iloc[-40:]}
    )
    assert result.entry is not None
    assert result.stop == pytest.approx(result.entry * 0.99)
    assert result.target == pytest.approx(result.entry * 1.02)
    assert (result.target - result.entry) / (result.entry - result.stop) == pytest.approx(2.0)


def test_paper_rebases_sl_tp_on_next_open_fill():
    ledger = PaperLedger(use_fixed_pct_exits=True, soft_filters=False, apply_research_costs=False)
    ledger._closed = []
    ledger._equity = 100_000.0
    pred = PredictorOutput(
        timestamp=pd.Timestamp("2025-01-01", tz="UTC"),
        bias="bullish",
        entry=64000.0,
        stop=63000.0,
        target=66000.0,
        position_size=0.5,
        reward_risk=2.0,
        zone="swing:x",
        zone_kind="swing",
        setup_type="reversal",
        sweep_status="confirmed",
        orderflow_confirmation=True,
    )
    ledger.update(pred)
    fill = pd.DataFrame(
        {"open": [64100.0], "high": [64150.0], "low": [64050.0], "close": [64120.0]},
        index=[pred.timestamp + pd.Timedelta(minutes=1)],
    )
    ledger.update_market(fill)
    assert ledger._open is not None
    assert ledger._open["stop"] == pytest.approx(64100.0 * 0.99)
    assert ledger._open["target"] == pytest.approx(64100.0 * 1.02)
    assert live_policy.fill_min_rr_ok(
        ledger._open["entry"], ledger._open["stop"], ledger._open["target"], "long"
    )


def test_max_hold_closes_overnight_loser():
    ledger = PaperLedger(
        use_fixed_pct_exits=False, soft_filters=False, apply_research_costs=False, max_hold_hours=1
    )
    ledger._closed = []
    ledger._equity = 100_000.0
    pred = PredictorOutput(
        timestamp=pd.Timestamp("2025-01-01", tz="UTC"),
        bias="bullish",
        entry=100.0,
        stop=90.0,
        target=120.0,
        position_size=1.0,
        zone="swing:x",
        setup_type="reversal",
        sweep_status="confirmed",
        orderflow_confirmation=True,
    )
    ledger.update(pred)
    fill = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0]},
        index=[pred.timestamp + pd.Timedelta(minutes=1)],
    )
    ledger.update_market(fill)
    later = pd.DataFrame(
        {"open": [100.5], "high": [101.0], "low": [99.5], "close": [100.2]},
        index=[pred.timestamp + pd.Timedelta(hours=2)],
    )
    status = ledger.update_market(later)
    assert status["closed_trades"] == 1
    assert status["last_closed"]["exit_reason"] == "max_hold"


def test_shadow_skips_untested_breakout():
    decision = shadow_rule_skip(
        {"zone": "untested_breakout:abc", "zone_kind": "untested_breakout", "reward_risk": 2.0},
        "skip_untested_breakout",
    )
    assert decision["skip"] is True
