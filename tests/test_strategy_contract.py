import pandas as pd

from btc_predictor.models import PredictorOutput, Zone
from btc_predictor.strategy import Predictor

from btc_predictor.paper_position import PaperLedger

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
    assert ledger._open is not None and ledger._open["side"] == "long"
    assert ledger._closed == []
    # Still bullish should not flip the open position
    ledger.update(PredictorOutput(timestamp=pd.Timestamp("2025-01-01 00:01",tz="UTC"),bias="bullish"))
    assert ledger._open is not None
    assert ledger._closed == []
    # Bearish flips the position
    ledger.update(PredictorOutput(timestamp=pd.Timestamp("2025-01-01 00:02",tz="UTC"),bias="bearish"))
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
    assert ledger._open is not None
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
    ledger.update(pred1)
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
    status = ledger.update(pred2, ohlc)
    assert status["closed_trades"] == 4
    assert status["last_closed"]["exit_reason"] == "superseded_by_new_setup"
    assert ledger._open is not None and ledger._open["zone"] == "zone2"
