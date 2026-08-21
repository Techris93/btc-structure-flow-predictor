import pandas as pd
import pytest

from btc_predictor.models import PredictorOutput, Zone
from btc_predictor.strategy import Predictor, detect_continuation, detect_sweep
from btc_predictor.footprint import footprint_confirmation
from btc_predictor.paper_position import PaperLedger
import app as web_app


def _make_ohlc(count=100, base=100.0, trend=0.1):
    idx = pd.date_range("2026-01-01 00:01", periods=count, freq="min", tz="UTC")
    closes = [base + i * trend for i in range(count)]
    return pd.DataFrame({
        "open": closes,
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
        "volume": [10.0] * count,
        "taker_buy_volume": [6.0] * count,
    }, index=idx)


def test_15m_pullback_inside_htf_trend_produces_setup(monkeypatch):
    """Task 1: 15m pullback inside a 4H/1H bullish trend does not veto regime to neutral."""
    idx = pd.date_range("2026-01-01", periods=100, freq="min", tz="UTC")
    ohlc = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0}, index=idx)
    frames = {"15m": ohlc.copy(), "1h": ohlc.copy(), "4h": ohlc.copy()}
    def fake_last(self, frame):
        if frame is frames["15m"]:
            return "bearish", pd.DataFrame(columns=["bias", "event", "level"])
        return "bullish", pd.DataFrame(columns=["bias", "event", "level"])
    monkeypatch.setattr(Predictor, "_last", fake_last)
    predictor = Predictor(require_15m_align=False)
    assert predictor._regime_bias(frames) == "bullish"
    assert predictor.last_regimes["15m"] == "bearish"
    assert predictor.last_regimes["1h"] == "bullish"
    assert predictor.last_regimes["4h"] == "bullish"


def test_continuation_breakout_and_retest_detection():
    """Task 2: Breakout above resistance zone, acceptance, retest from above and bounce."""
    idx = pd.date_range("2026-01-01 00:01", periods=20, freq="min", tz="UTC")
    # Zone is 100.0 to 101.0
    zone = Zone("res1", "untested_breakout", "above", 100.0, 101.0, 1.5, idx[0], idx[1])
    # Bars: 0-3 below zone, 4-7 breakout & accept above 101, 8-12 retest down to 101.2 and hold, 13-19 bounce to 105
    closes = [99.0, 99.5, 99.8, 100.0, 102.5, 103.0, 103.5, 103.0, 101.2, 101.1, 101.5, 102.0, 103.0, 104.0, 104.5, 105.0, 105.5, 105.0, 105.5, 106.0]
    lows = [c - 0.5 for c in closes]
    highs = [c + 0.5 for c in closes]
    ohlc = pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes, "volume": 10.0}, index=idx)
    res = detect_continuation(ohlc, zone, "bullish", setup_atr=2.0)
    assert res["status"] == "confirmed"
    assert res["confirmed"] is True
    assert res["reclaim_time"] is not None
    assert res["extreme"] <= 101.5


def test_regime_aware_order_flow_continuation_vs_reversal():
    """Task 3: Continuation order flow evaluates trend delta, momentum and low opposing absorption."""
    idx = pd.date_range("2026-01-01 00:01", periods=25, freq="min", tz="UTC")
    bars = pd.DataFrame({
        "open": [100.0 + i for i in range(25)],
        "high": [101.0 + i for i in range(25)],
        "low": [99.5 + i for i in range(25)],
        "close": [100.8 + i for i in range(25)],
        "volume": [100.0] * 25,
        "taker_buy_volume": [80.0] * 25,  # Strong aggressive buying
    }, index=idx)
    end = bars.index[-1]
    start = bars.index[-10]
    trades = pd.DataFrame([
        {"time": end - pd.Timedelta(seconds=40), "price": 125, "qty": 10, "side": "buy", "exchange": "binance"},
        {"time": end - pd.Timedelta(seconds=20), "price": 125, "qty": 10, "side": "buy", "exchange": "bybit"},
    ])
    # Test continuation confirmation
    confirmed_cont, details_cont = footprint_confirmation(
        trades, bars, "bullish", start, end, setup_type="continuation",
        market_threshold=0.40, raw_threshold=0.40, gate_mode="independent"
    )
    assert details_cont["market_flow_confirmed"] is True
    assert details_cont["raw_footprint_confirmed"] is True
    assert confirmed_cont is True


def test_superseded_trade_exit_notification_and_new_trade(monkeypatch, tmp_path):
    """Task 4: Superseded trade produces a distinct paper exit notification and clean replacement."""
    exit_store = web_app.JsonStore(tmp_path / "paper_exit_push.json")
    monkeypatch.setattr(web_app, "paper_exit_push_store", exit_store)
    sent_pushes = []
    monkeypatch.setattr(web_app, "_send_push", lambda payload, **k: (sent_pushes.append(payload) or 1, 0))
    monkeypatch.setattr(web_app, "_latest_delivery_summary", lambda *a: {"attempted": 1, "accepted": 1, "failed": 0, "batch_id": "b1"})
    
    trade_superseded = {
        "entry_time": "2026-08-10T13:00:00Z",
        "exit_time": "2026-08-10T14:00:00Z",
        "side": "short",
        "entry": 64800.0,
        "exit": 64700.0,
        "stop": 65100.0,
        "target": 64000.0,
        "size": 1.0,
        "pnl": 100.0,
        "r_multiple": 0.33,
        "exit_reason": "superseded_by_confirmed_setup",
        "setup_type": "reversal",
    }
    
    # Notification should now be dispatched for superseded setup
    accepted = web_app._notify_paper_exits([trade_superseded])
    assert accepted == 1
    assert len(sent_pushes) == 1
    assert "Setup replaced" in sent_pushes[0]["title"]
    assert "P&L +$100.00" in sent_pushes[0]["body"]


def test_paper_position_records_and_segregates_setup_type(tmp_path):
    """Task 4: PaperLedger records setup_type and reports segregated statistics."""
    ledger = PaperLedger(tmp_path / "ledger.json", use_fixed_pct_exits=False)
    pred_cont = PredictorOutput(
        timestamp=pd.Timestamp("2026-01-01 00:00", tz="UTC"),
        bias="bullish",
        setup_type="continuation",
        entry=100.0,
        stop=95.0,
        target=110.0,
        position_size=1.0,
        zone="zone_cont_1",
    )
    event = {
        "event_id": "e1",
        "event_type": "setup_confirmed",
        "signal_id": "sig-cont-1",
        "created_at": "2026-01-01T00:00:00Z",
        "snapshot": dict(pred_cont.__dict__, timestamp="2026-01-01T00:00:00Z"),
    }
    status = ledger.apply_lifecycle([event])
    # Fill at next open
    fill_bar = pd.DataFrame({"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0]}, index=[pd.Timestamp("2026-01-01 00:01", tz="UTC")])
    status = ledger.update_market(fill_bar)
    assert status["open_position"] is not None
    assert status["open_position"]["setup_type"] == "continuation"
    
    # Hit target
    target_bar = pd.DataFrame({"open": [105.0], "high": [111.0], "low": [104.0], "close": [110.5]}, index=[pd.Timestamp("2026-01-01 00:02", tz="UTC")])
    status = ledger.update_market(target_bar)
    assert status["open_position"] is None
    assert status["closed_trades"] == 4  # 3 seeded historical + 1 new closed trade
    assert status["last_closed"]["setup_type"] == "continuation"
    assert status["setup_type_stats"]["continuation"]["trades"] == 1
    assert status["setup_type_stats"]["continuation"]["wins"] == 1
    assert status["setup_type_stats"]["reversal"]["trades"] == 3


def test_strict_single_position_same_direction_rejected_opposite_flips(tmp_path):
    """Option A: While Short is active, 2nd Short is rejected. Opposite Long closes Short as signal_flipped."""
    ledger = PaperLedger(tmp_path / "ledger.json", use_fixed_pct_exits=False)

    # 1. Open Trade #1 (Short)
    pred_short1 = PredictorOutput(
        timestamp=pd.Timestamp("2026-08-10 13:00", tz="UTC"),
        bias="bearish",
        setup_type="reversal",
        entry=64862.80,
        stop=65268.25,
        target=64112.00,
        position_size=1.0,
        zone="zone_short_1",
    )
    event_s1 = {
        "event_id": "e_s1",
        "event_type": "setup_confirmed",
        "signal_id": "sig-short-1",
        "created_at": "2026-08-10T13:00:00Z",
        "snapshot": dict(pred_short1.__dict__, timestamp="2026-08-10T13:00:00Z"),
    }
    ledger.apply_lifecycle([event_s1])
    fill_bar1 = pd.DataFrame(
        {"open": [64862.80], "high": [64900.0], "low": [64800.0], "close": [64850.0]},
        index=[pd.Timestamp("2026-08-10 13:01", tz="UTC")],
    )
    status = ledger.update_market(fill_bar1)
    assert status["open_position"] is not None
    assert status["open_position"]["signal_id"] == "sig-short-1"
    assert status["open_position"]["entry"] == 64862.80
    assert status["open_position"]["stop"] == 65268.25
    assert status["open_position"]["target"] == 64112.00
    closed_before = status["closed_trades"]

    # 2. Second Short arrives while Trade #1 is active -> must be rejected without mutating Trade #1
    pred_short2 = PredictorOutput(
        timestamp=pd.Timestamp("2026-08-10 14:00", tz="UTC"),
        bias="bearish",
        setup_type="reversal",
        entry=64787.60,
        stop=64982.23,
        target=64472.50,
        position_size=1.2,
        zone="zone_short_2",
    )
    event_s2 = {
        "event_id": "e_s2",
        "event_type": "setup_confirmed",
        "signal_id": "sig-short-2",
        "created_at": "2026-08-10T14:00:00Z",
        "snapshot": dict(pred_short2.__dict__, timestamp="2026-08-10T14:00:00Z"),
    }
    status2 = ledger.apply_lifecycle([event_s2])
    # Assert Trade #1 is completely untouched
    assert status2["closed_trades"] == closed_before
    assert status2["open_position"] is not None
    assert status2["open_position"]["signal_id"] == "sig-short-1"
    assert status2["open_position"]["entry"] == 64862.80
    assert status2["open_position"]["stop"] == 65268.25
    assert status2["open_position"]["target"] == 64112.00
    assert status2["open_position"]["size"] == status["open_position"]["size"]
    assert status2["pending_order"] is None

    # 3. Third setup in OPPOSITE direction (Bullish Long) arrives -> closes Short #1 as signal_flipped and opens Long
    pred_long1 = PredictorOutput(
        timestamp=pd.Timestamp("2026-08-10 15:00", tz="UTC"),
        bias="bullish",
        setup_type="continuation",
        entry=65100.0,
        stop=64700.0,
        target=65800.0,
        position_size=0.8,
        zone="zone_long_1",
    )
    event_l1 = {
        "event_id": "e_l1",
        "event_type": "setup_confirmed",
        "signal_id": "sig-long-1",
        "created_at": "2026-08-10T15:00:00Z",
        "snapshot": dict(pred_long1.__dict__, timestamp="2026-08-10T15:00:00Z"),
    }
    flip_bar = pd.DataFrame(
        {"open": [64950.0], "high": [65000.0], "low": [64900.0], "close": [64950.0]},
        index=[pd.Timestamp("2026-08-10 15:00", tz="UTC")],
    )
    status3 = ledger.apply_lifecycle([event_l1], flip_bar)
    assert status3["closed_trades"] == closed_before + 1
    assert status3["last_closed"]["signal_id"] == "sig-short-1"
    assert status3["last_closed"]["exit_reason"] == "signal_flipped"
    assert status3["pending_order"] is not None
    assert status3["pending_order"]["signal_id"] == "sig-long-1"
    assert status3["pending_order"]["side"] == "long"


def test_flat_ledger_allows_subsequent_setup_confirmation_after_skipped_signal():
    """When paper ledger is flat (previous signal skipped or closed), new setup confirms and notifies."""
    engine = web_app.SignalLifecycle(confirm_observations=2)
    state = engine.initial_state()

    # 1. Signal 1 confirms in lifecycle
    pred1 = PredictorOutput(
        timestamp=pd.Timestamp("2026-08-19 10:00", tz="UTC"),
        bias="bullish",
        setup_type="continuation",
        entry=64332.80,
        stop=64011.14,
        target=64976.13,
        position_size=0.77,
        zone="zone1",
        sweep_status="confirmed",
        sweep_time="2026-08-19T09:58:00+00:00",
        orderflow_confirmation=True,
    )
    state, _ = engine.evaluate(state, pred1, {}, pd.Timestamp("2026-08-19 10:00:30Z"))
    pred1_b = PredictorOutput(**{**pred1.__dict__, "timestamp": pd.Timestamp("2026-08-19 10:01", tz="UTC")})
    state, events = engine.evaluate(state, pred1_b, {}, pd.Timestamp("2026-08-19 10:01:30Z"))
    assert len(events) == 1
    assert events[0]["event_type"] == "setup_confirmed"
    assert state["active"] is not None

    # 2. Paper status is passed and shows account is flat (open_position=None, pending_order=None, closed_trades=28)
    flat_paper_status = {
        "open_position": None,
        "pending_order": None,
        "closed_trades": 28,
        "equity": 96410.14,
    }

    # 3. Subsequent Signal 2 (e.g. afternoon breakout at 64,775) arrives while flat
    pred2 = PredictorOutput(
        timestamp=pd.Timestamp("2026-08-19 13:42", tz="UTC"),
        bias="bullish",
        setup_type="continuation",
        entry=64775.00,
        stop=64451.13,
        target=65422.75,
        position_size=0.77,
        zone="zone2",
        sweep_status="confirmed",
        sweep_time="2026-08-19T13:40:00+00:00",
        orderflow_confirmation=True,
    )
    state, events2_a = engine.evaluate(state, pred2, flat_paper_status, pd.Timestamp("2026-08-19 13:42:30Z"))
    assert events2_a == []
    pred2_b = PredictorOutput(**{**pred2.__dict__, "timestamp": pd.Timestamp("2026-08-19 13:43", tz="UTC")})
    state, events2_b = engine.evaluate(state, pred2_b, flat_paper_status, pd.Timestamp("2026-08-19 13:43:30Z"))

    # Signal 2 must confirm and notify because the account was flat
    assert len(events2_b) == 1
    assert events2_b[0]["event_type"] == "setup_confirmed"
    assert events2_b[0]["snapshot"]["entry"] == 64775.00


def test_continuation_rearm_refractory_period_prevents_rapid_reconfirmation():
    """Verify that same-direction continuation setups within 30m / 1.0 ATR are throttled."""
    engine = web_app.SignalLifecycle(confirm_observations=2, continuation_rearm_seconds=1800, continuation_rearm_atr=1.0)
    state = engine.initial_state()
    flat_status = {"open_position": None, "pending_order": None, "closed_trades": 0}

    # 1. Setup #1 confirmed at 10:00 UTC (Entry $70,000, ATR 250)
    p1 = PredictorOutput(
        timestamp=pd.Timestamp("2026-08-21 10:00", tz="UTC"),
        bias="bullish",
        setup_type="continuation",
        entry=70000.0,
        stop=69300.0,
        target=71400.0,
        position_size=1.0,
        zone="zone_1",
        sweep_status="confirmed",
        sweep_time="2026-08-21T09:58:00+00:00",
        orderflow_confirmation=True,
        setup_atr=250.0,
    )
    state, _ = engine.evaluate(state, p1, flat_status, pd.Timestamp("2026-08-21 10:00:30Z"))
    p1_next = PredictorOutput(**{**p1.__dict__, "timestamp": pd.Timestamp("2026-08-21 10:01", tz="UTC")})
    state, events1 = engine.evaluate(state, p1_next, flat_status, pd.Timestamp("2026-08-21 10:01:30Z"))
    assert len(events1) == 1
    assert events1[0]["event_type"] == "setup_confirmed"

    # 2. Setup #2 arrives 5 minutes later at 10:06 with Entry $70,100 (diff is $100 < 1.0 ATR ($250))
    p2 = PredictorOutput(
        timestamp=pd.Timestamp("2026-08-21 10:06", tz="UTC"),
        bias="bullish",
        setup_type="continuation",
        entry=70100.0,
        stop=69399.0,
        target=71502.0,
        position_size=1.0,
        zone="zone_2",
        sweep_status="confirmed",
        sweep_time="2026-08-21T10:05:00+00:00",
        orderflow_confirmation=True,
        setup_atr=250.0,
    )
    state, events2_a = engine.evaluate(state, p2, flat_status, pd.Timestamp("2026-08-21 10:06:30Z"))
    p2_next = PredictorOutput(**{**p2.__dict__, "timestamp": pd.Timestamp("2026-08-21 10:07", tz="UTC")})
    state, events2_b = engine.evaluate(state, p2_next, flat_status, pd.Timestamp("2026-08-21 10:07:30Z"))
    # Should be throttled by refractory period (no new notification)
    assert events2_a == []
    assert events2_b == []

    # 3. Setup #3 arrives after 35 minutes (elapsed > 30m) -> Confirms normally
    p3 = PredictorOutput(
        timestamp=pd.Timestamp("2026-08-21 10:38", tz="UTC"),
        bias="bullish",
        setup_type="continuation",
        entry=70800.0,
        stop=70092.0,
        target=72216.0,
        position_size=1.0,
        zone="zone_3",
        sweep_status="confirmed",
        sweep_time="2026-08-21T10:36:00+00:00",
        orderflow_confirmation=True,
        setup_atr=250.0,
    )
    state, events3_a = engine.evaluate(state, p3, flat_status, pd.Timestamp("2026-08-21 10:38:30Z"))
    p3_next = PredictorOutput(**{**p3.__dict__, "timestamp": pd.Timestamp("2026-08-21 10:39", tz="UTC")})
    state, events3_b = engine.evaluate(state, p3_next, flat_status, pd.Timestamp("2026-08-21 10:39:30Z"))
    assert len(events3_b) == 1
    assert events3_b[0]["event_type"] == "setup_confirmed"
    assert events3_b[0]["snapshot"]["entry"] == 70800.0
