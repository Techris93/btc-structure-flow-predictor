import pandas as pd
import pytest

from btc_predictor.models import PredictorOutput, Zone
from btc_predictor.strategy import Predictor

from btc_predictor.paper_position import PaperLedger


def confirmed_event(prediction, signal_id="signal-1", created_at=None):
    snapshot = dict(prediction.__dict__)
    snapshot["timestamp"] = pd.Timestamp(prediction.timestamp).isoformat()
    return {
        "event_id": f"confirmed-{signal_id}",
        "event_type": "setup_confirmed",
        "signal_id": signal_id,
        "created_at": pd.Timestamp(created_at or prediction.timestamp).isoformat(),
        "snapshot": snapshot,
    }

def test_strategy_uses_setup_atr_checks_all_zones_and_leaves_probability_uncalibrated(monkeypatch):
    idx=pd.date_range("2025-01-01",periods=100,freq="min",tz="UTC")
    o=pd.DataFrame({"open":100.,"high":101.,"low":99.,"close":100.,"volume":10.},index=idx)
    o.iloc[-1,o.columns.get_loc("low")]=89; o.iloc[-1,o.columns.get_loc("close")]=92
    setup=o.iloc[::15].copy()
    frames={"15m":setup,"1h":o.iloc[-40:].copy(),"4h":o.iloc[-40:].copy()}
    trades=pd.DataFrame({"time":idx,"price":100.,"qty":1.,"side":"buy"})
    near=Zone("near","swing","below",95,96,2,idx[1],idx[2])
    swept=Zone("swept","swing","below",90,91,3,idx[1],idx[2])
    target=Zone("target","swing","above",110,111,1,idx[1],idx[2])
    monkeypatch.setattr(Predictor,"_regime_bias",lambda self,frames:"bullish")
    monkeypatch.setattr("btc_predictor.strategy.build_projected_zones",lambda frame:[near,swept,target])
    monkeypatch.setattr("btc_predictor.strategy.atr",lambda frame:pd.Series(5.,index=frame.index))
    monkeypatch.setattr("btc_predictor.strategy.footprint_confirmation",lambda *a,**k:(True,{"reason":"confirmed","agreement":True}))
    result=Predictor(min_rr=.1).predict(o,trades,frames=frames)
    assert result.zone == "swept"
    assert result.stop <= result.entry - 1.5*5
    assert result.probability_tp_before_sl is not None and 0 <= result.probability_tp_before_sl <= 1
    assert result.sweep_evaluation_status == "evaluated"
    assert result.orderflow_evaluation_status == "evaluated"


def test_predictor_marks_orderflow_not_evaluated_until_sweep_confirms(monkeypatch):
    idx = pd.date_range("2025-01-01", periods=100, freq="min", tz="UTC")
    ohlc = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0},
        index=idx,
    )
    trades = pd.DataFrame({"time": idx, "price": 100.0, "qty": 1.0, "side": "buy"})
    frames = {"15m": ohlc.iloc[::15], "1h": ohlc.iloc[-40:], "4h": ohlc.iloc[-40:]}
    zone = Zone("waiting", "swing", "below", 95, 96, 2, idx[1], idx[2])
    monkeypatch.setattr(Predictor, "_regime_bias", lambda self, frames: "bullish")
    monkeypatch.setattr("btc_predictor.strategy.build_projected_zones", lambda frame: [zone])
    monkeypatch.setattr("btc_predictor.strategy.atr", lambda frame: pd.Series(5.0, index=frame.index))
    monkeypatch.setattr(
        "btc_predictor.strategy.detect_sweep",
        lambda *args, **kwargs: {"confirmed": False, "status": "none"},
    )

    result = Predictor().predict(ohlc, trades, frames=frames, flow_source="rest_backfill")

    assert result.sweep_evaluation_status == "evaluated"
    assert result.orderflow_evaluation_status == "not_evaluated"
    assert result.orderflow_confirmation is False
    assert result.orderflow_reason == "awaiting_confirmed_sweep"


def test_predictor_requires_15m_alignment_with_higher_timeframes(monkeypatch):
    idx=pd.date_range("2025-01-01",periods=100,freq="min",tz="UTC")
    o=pd.DataFrame({"open":100.,"high":101.,"low":99.,"close":100.,"volume":10.},index=idx)
    frames={"15m":o.copy(),"1h":o.copy(),"4h":o.copy()}
    def fake_last(self, frame):
        if frame is frames["15m"]:
            return "bearish", pd.DataFrame(columns=["bias","event","level"])
        return "bullish", pd.DataFrame(columns=["bias","event","level"])
    monkeypatch.setattr(Predictor,"_last",fake_last)
    predictor=Predictor()
    assert predictor._regime_bias(frames)=="neutral"


def test_paper_ledger_keeps_position_when_bias_remains_on_same_side():
    ledger=PaperLedger()
    pred=PredictorOutput(
        timestamp=pd.Timestamp("2025-01-01",tz="UTC"),
        bias="bullish",
        entry=100.0,
        stop=95.0,
        target=110.0,
        position_size=1.0,
    )
    ledger.update(pred)
    assert ledger._pending is not None and ledger._open is None
    fill_bar=pd.DataFrame(
        {"open":[100.0],"high":[101.0],"low":[99.0],"close":[100.0]},
        index=[pred.timestamp+pd.Timedelta(minutes=1)],
    )
    ledger.update_market(fill_bar)
    assert ledger._open is not None and ledger._open["side"] == "long"
    assert ledger._closed == []
    # Still bullish should not flip the open position
    ledger.update(PredictorOutput(timestamp=pd.Timestamp("2025-01-01 00:01",tz="UTC"),bias="bullish"))
    assert ledger._open is not None
    assert ledger._closed == []
    # A raw bearish output cannot flip the position; lifecycle owns exits.
    ledger.update(PredictorOutput(timestamp=pd.Timestamp("2025-01-01 00:02",tz="UTC"),bias="bearish"))
    assert ledger._open is not None
    invalidated = {
        "event_type": "setup_invalidated",
        "signal_id": "signal-1",
        "created_at": "2025-01-01T00:02:00+00:00",
        "reason": "signal_flipped",
    }
    ledger.apply_lifecycle([invalidated])
    assert ledger._open is None
    assert len(ledger._closed) == 1
    assert ledger._closed[0]["exit_reason"] == "signal_flipped"


def test_paper_ledger_ignores_bars_before_entry():
    idx = pd.date_range("2025-01-01", periods=5, freq="min", tz="UTC")
    ohlc = pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [90.0, 99.0, 99.0, 99.0, 99.0],  # historical low before entry
            "close": [100.0] * 5,
            "volume": [1.0] * 5,
        },
        index=idx,
    )
    ledger = PaperLedger()
    pred = PredictorOutput(
        timestamp=idx[-1],
        bias="bullish",
        entry=100.0,
        stop=95.0,
        target=110.0,
        position_size=1.0,
    )
    status = ledger.update(pred, ohlc)
    assert ledger._open is None and ledger._pending is not None
    assert status["closed_trades"] == 0
    # After entry, a real stop should close the position.
    later = pd.DataFrame(
        {
            "open": [100.0],
            "high": [100.5],
            "low": [94.0],
            "close": [95.0],
            "volume": [1.0],
        },
        index=pd.DatetimeIndex([idx[-1] + pd.Timedelta(minutes=1)]),
    )
    status = ledger.update(PredictorOutput(timestamp=later.index[0], bias="bullish"), later)
    assert ledger._open is None
    assert status["closed_trades"] == 1
    assert status["last_closed"]["exit_reason"] == "stop"
    assert len(status["newly_closed"]) == 1
    assert status["newly_closed"][0]["exit_reason"] == "stop"


def test_paper_ledger_atomic_persistence_and_superseded_setups(tmp_path):
    path = tmp_path / "paper_ledger.json"
    ledger = PaperLedger(path)
    # Since path didn't exist, it seeds the 3 historical trades
    status = ledger._status()
    assert status["closed_trades"] == 3
    assert status["equity"] > 100000.0

    # Test opening a setup
    pred1 = PredictorOutput(
        timestamp=pd.Timestamp("2026-07-25 10:00", tz="UTC"),
        bias="bearish",
        entry=65000.0,
        stop=65200.0,
        target=64500.0,
        position_size=1.0,
        zone="zone1",
    )
    ledger.apply_lifecycle([confirmed_event(pred1, "signal-1")])
    fill1=pd.DataFrame(
        {"open":[65000.0],"high":[65050.0],"low":[64950.0],"close":[65000.0],"volume":[1.0]},
        index=[pd.Timestamp("2026-07-25 10:01",tz="UTC")],
    )
    ledger.update_market(fill1)
    assert ledger._open is not None and ledger._open["zone"] == "zone1"

    # Test superseded by a new setup on a different zone
    pred2 = PredictorOutput(
        timestamp=pd.Timestamp("2026-07-25 10:15", tz="UTC"),
        bias="bearish",
        entry=64800.0,
        stop=65000.0,
        target=64200.0,
        position_size=1.0,
        zone="zone2",
    )
    ohlc = pd.DataFrame(
        {"open": [64900.0], "high": [64950.0], "low": [64850.0], "close": [64900.0], "volume": [1.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-25 10:15", tz="UTC")]),
    )
    # Raw output is inert; only a confirmed replacement may supersede.
    status = ledger.update(pred2, ohlc)
    assert status["closed_trades"] == 3
    assert ledger._open is not None and ledger._open["zone"] == "zone1"
    status = ledger.apply_lifecycle([confirmed_event(pred2, "signal-2")], ohlc)
    assert status["closed_trades"] == 4
    assert status["last_closed"]["exit_reason"] == "superseded_by_confirmed_setup"
    assert ledger._open is None and ledger._pending is not None and ledger._pending["zone"] == "zone2"


def test_paper_ledger_no_churn_on_same_setup_reemission(tmp_path):
    """A re-emitted signal for the same setup keeps its original trade."""
    ledger = PaperLedger(tmp_path / "ledger.json")
    base = dict(
        bias="bearish",
        stop=65200.0,
        target=64500.0,
        position_size=1.0,
        zone="zone1",
        sweep_time="2026-07-25 09:58:00+00:00",
    )
    pred1 = PredictorOutput(
        timestamp=pd.Timestamp("2026-07-25 10:00", tz="UTC"),
        entry=65000.0,
        **base,
    )
    ledger.apply_lifecycle([confirmed_event(pred1, "signal-1")])
    fill=pd.DataFrame(
        {"open":[65000.0],"high":[65050.0],"low":[64950.0],"close":[65000.0]},
        index=[pd.Timestamp("2026-07-25 10:01",tz="UTC")],
    )
    ledger.update_market(fill)
    assert ledger._open is not None and ledger._open["entry"] == 65000.0

    # Same setup re-emitted each minute with drifting prices: no churn.
    for minute in (1, 2, 3):
        pred = PredictorOutput(
            timestamp=pd.Timestamp(f"2026-07-25 10:0{minute}", tz="UTC"),
            entry=65000.0 - 10 * minute,
            **base,
        )
        status = ledger.update(pred)
        assert status["closed_trades"] == 3
        assert ledger._open is not None and ledger._open["entry"] == 65000.0

    # A raw new sweep remains inert until lifecycle confirmation.
    pred_new_sweep = PredictorOutput(
        timestamp=pd.Timestamp("2026-07-25 10:05", tz="UTC"),
        entry=64900.0,
        **{**base, "sweep_time": "2026-07-25 10:03:00+00:00"},
    )
    status = ledger.update(pred_new_sweep)
    assert status["closed_trades"] == 3
    assert ledger._open is not None and ledger._open["entry"] == 65000.0
    status = ledger.apply_lifecycle([confirmed_event(pred_new_sweep, "signal-2")])
    assert status["closed_trades"] == 4
    assert status["last_closed"]["exit_reason"] == "superseded_by_confirmed_setup"
    assert ledger._open is None and ledger._pending is not None and ledger._pending["entry"] == 64900.0


def test_deep_sweep_retrace_entry_is_directional_and_recomputes_rr(monkeypatch):
    idx = pd.date_range("2026-01-01", periods=100, freq="min", tz="UTC")
    ohlc = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0},
        index=idx,
    )
    trades = pd.DataFrame({"time": idx, "price": 100.0, "qty": 1.0, "side": "buy"})
    frames = {"15m": ohlc.iloc[::15], "1h": ohlc.iloc[-40:], "4h": ohlc.iloc[-40:]}
    sweep_zone = Zone("sweep", "swing", "below", 95, 96, 3, idx[1], idx[2])
    target_zone = Zone("target", "swing", "above", 120, 121, 1, idx[1], idx[2])

    monkeypatch.setattr(Predictor, "_regime_bias", lambda self, frames: "bullish")
    monkeypatch.setattr("btc_predictor.strategy.atr", lambda frame: pd.Series(5.0, index=frame.index))
    monkeypatch.setattr(
        "btc_predictor.strategy.footprint_confirmation",
        lambda *args, **kwargs: (True, {"reason": "confirmed", "agreement": True, "score": 0.8}),
    )
    monkeypatch.setattr(
        "btc_predictor.strategy.detect_sweep",
        lambda *args, **kwargs: {
            "confirmed": True,
            "status": "confirmed",
            "depth_atr": 2.0,
            "extreme": 90.0,
            "time": idx[-2],
            "reclaim_time": idx[-1],
        },
    )
    monkeypatch.setattr("btc_predictor.strategy.build_projected_zones", lambda frame: [sweep_zone, target_zone])

    result = Predictor(retrace_entry_atr=1.2).predict(ohlc, trades, frames=frames)

    assert result.entry_type == "limit"
    assert result.entry < 100.0
    assert result.reward_risk >= 1.5


def test_pending_retrace_order_waits_for_a_real_limit_touch(tmp_path):
    t0 = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    prediction = PredictorOutput(
        timestamp=t0,
        bias="bullish",
        entry=95.0,
        stop=87.5,
        target=120.0,
        position_size=1.0,
        zone="zone1",
        sweep_time="2026-01-01 00:00:00+00:00",
        entry_type="limit",
    )
    ledger = PaperLedger(tmp_path / "ledger.json")

    signal_bar = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0]},
        index=[t0],
    )
    status = ledger.update(prediction, signal_bar)
    assert status["pending_order"] is not None
    assert status["open_position"] is None

    no_touch_time = t0 + pd.Timedelta(minutes=1)
    no_touch = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [98.0], "close": [100.0]},
        index=[no_touch_time],
    )
    status = ledger.update(
        PredictorOutput(timestamp=no_touch_time, bias="bullish", zone="zone1", sweep_time=prediction.sweep_time),
        no_touch,
    )
    assert status["pending_order"] is not None
    assert status["open_position"] is None

    fill_time = t0 + pd.Timedelta(minutes=2)
    touch = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [94.0], "close": [95.0]},
        index=[fill_time],
    )
    status = ledger.update(
        PredictorOutput(timestamp=fill_time, bias="bullish", zone="zone1", sweep_time=prediction.sweep_time),
        touch,
    )
    assert status["pending_order"] is None
    assert status["open_position"] is not None
    assert status["open_position"]["entry"] == 95.0


def test_paper_ledger_lifecycle_neutral_exit_and_unrealized_pnl(tmp_path):
    """Lifecycle neutralization closes a dead thesis; raw neutral is inert."""
    ledger = PaperLedger(tmp_path / "ledger.json", neutral_exit_observations=3)
    pred_open = PredictorOutput(
        timestamp=pd.Timestamp("2026-07-25 10:00", tz="UTC"),
        bias="bearish",
        entry=65000.0,
        stop=65200.0,
        target=64500.0,
        position_size=1.0,
        zone="zone1",
        sweep_time="2026-07-25 09:58:00+00:00",
    )
    ledger.apply_lifecycle([confirmed_event(pred_open, "signal-1")])

    ohlc = pd.DataFrame(
        {"open": [65000.0], "high": [65050.0], "low": [64850.0], "close": [64900.0], "volume": [1.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-25 10:01", tz="UTC")]),
    )
    neutral = lambda minute: PredictorOutput(
        timestamp=pd.Timestamp(f"2026-07-25 10:0{minute}", tz="UTC"),
        bias="neutral",
    )

    status = ledger.update(neutral(1), ohlc)
    assert ledger._open is not None
    assert status["open_unrealized_pnl"] == 100.0
    assert status["mark_to_market_equity"] == round(status["equity"] + 100.0, 2)

    ledger.update(neutral(2), ohlc)
    ledger.update(neutral(3), ohlc)
    assert ledger._open is not None
    status = ledger.apply_lifecycle([{
        "event_type": "setup_invalidated",
        "signal_id": "signal-1",
        "created_at": "2026-07-25T10:03:00+00:00",
        "reason": "signal_neutralized",
    }], ohlc)
    assert ledger._open is None
    assert status["last_closed"]["exit_reason"] == "signal_neutralized"
    assert status["last_closed"]["exit"] == 64900.0

    # A directional re-confirmation resets the neutral grace counter.
    ledger2 = PaperLedger(tmp_path / "ledger2.json", neutral_exit_observations=3)
    ledger2.apply_lifecycle([confirmed_event(pred_open, "signal-1")])
    ledger2.update(neutral(1), ohlc)
    ledger2.update(
        PredictorOutput(
            timestamp=pd.Timestamp("2026-07-25 10:02", tz="UTC"),
            bias="bearish",
            entry=65000.0,
            stop=65200.0,
            target=64500.0,
            position_size=1.0,
            zone="zone1",
            sweep_time="2026-07-25 09:58:00+00:00",
        ),
        ohlc,
    )
    ledger2.update(neutral(3), ohlc)
    ledger2.update(neutral(4), ohlc)
    assert ledger2._open is not None


def test_legacy_position_signal_binding_is_persisted_without_close(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = PaperLedger(path)
    pred = PredictorOutput(
        timestamp=pd.Timestamp("2026-07-25 10:00", tz="UTC"),
        bias="bullish",
        entry=65000.0,
        stop=64750.0,
        target=65500.0,
        position_size=.4,
        zone="zone1",
    )
    ledger.update(pred)
    closed_before = len(ledger._closed)

    assert ledger.bind_active_signal("adopted-signal") is True
    assert ledger._pending["signal_id"] == "adopted-signal"
    assert len(ledger._closed) == closed_before
    reloaded = PaperLedger(path)
    assert reloaded._pending["signal_id"] == "adopted-signal"


def test_market_signal_fills_next_open_and_preserves_planned_risk(tmp_path):
    ledger=PaperLedger(tmp_path/"ledger.json")
    decision=pd.Timestamp("2026-01-01 00:00",tz="UTC")
    prediction=PredictorOutput(
        timestamp=decision,bias="bullish",entry=100.0,stop=95.0,target=110.0,
        position_size=2.0,zone="zone1",
    )
    ledger.apply_lifecycle([confirmed_event(prediction,"signal-next-open")])
    assert ledger._open is None and ledger._pending["entry_type"]=="market_next_open"
    next_bar=pd.DataFrame(
        {"open":[102.0],"high":[103.0],"low":[101.0],"close":[102.5]},
        index=[decision+pd.Timedelta(minutes=1)],
    )
    status=ledger.update_market(next_bar)
    assert status["pending_order"] is None
    assert status["open_position"]["entry"]==102.0
    assert status["open_position"]["entry_time"]==(decision+pd.Timedelta(minutes=1)).isoformat()
    assert status["open_position"]["size"]==pytest.approx(10.0/7.0)


def test_pending_retrace_order_cancels_only_after_lifecycle_neutralization(tmp_path):
    ledger = PaperLedger(tmp_path / "ledger.json", neutral_exit_observations=2)
    t0 = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    pending = PredictorOutput(
        timestamp=t0,
        bias="bullish",
        entry=95.0,
        stop=87.5,
        target=120.0,
        position_size=1.0,
        zone="zone1",
        sweep_time="2026-01-01 00:00:00+00:00",
        entry_type="limit",
    )
    ledger.apply_lifecycle([confirmed_event(pending, "signal-1")])
    ledger.update(PredictorOutput(timestamp=t0 + pd.Timedelta(minutes=1), bias="neutral"))
    status = ledger.update(PredictorOutput(timestamp=t0 + pd.Timedelta(minutes=2), bias="neutral"))
    assert status["pending_order"] is not None
    status = ledger.apply_lifecycle([{
        "event_type": "setup_invalidated",
        "signal_id": "signal-1",
        "created_at": (t0 + pd.Timedelta(minutes=2)).isoformat(),
        "reason": "signal_neutralized",
    }])
    assert status["pending_order"] is None
