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
