import pandas as pd
import pytest

from btc_predictor.backtest import run_event_backtest
from btc_predictor.models import PredictorOutput


def bars(count=83):
    index = pd.date_range("2025-01-01 00:01", periods=count, freq="min", tz="UTC")
    return pd.DataFrame({"open":100.0,"high":100.5,"low":99.5,"close":100.0,"volume":10.0,"taker_buy_volume":5.0}, index=index)


class RecordingPredictor:
    def __init__(self, stop=90, target=110): self.calls=[]; self.stop=stop; self.target=target
    def predict(self, history, trades, equity, frames=None):
        self.calls.append((history.index[-1], trades.time.max(), frames))
        return PredictorOutput(history.index[-1], "bullish", "reversal", "z", "confirmed", True, 100, self.stop, self.target, 2, None, 1)


def test_decision_uses_closed_history_and_fills_next_open():
    ohlc = bars(); future = ohlc.index[80] + pd.Timedelta(seconds=30)
    known = ohlc.index[80] - pd.Timedelta(seconds=1)
    trades = pd.DataFrame({"time":[known, future],"price":[100,999],"qty":[1,1],"side":["buy","buy"]})
    predictor = RecordingPredictor()
    ledger, stats = run_event_backtest(ohlc, trades, predictor=predictor, slippage_bps=10)
    assert predictor.calls[0][0] == ohlc.index[80]
    assert predictor.calls[0][1] == known
    assert ledger.iloc[0].decision_time == ohlc.index[80]
    assert ledger.iloc[0].entry_time == ohlc.index[81]
    assert ledger.iloc[0].entry == pytest.approx(float(ohlc.open.iloc[81]) * 1.001)
    assert stats["causality"].startswith("close-time decision")


def test_same_bar_collision_policy_and_end_close_metrics():
    ohlc = bars(); ohlc.iloc[81, ohlc.columns.get_loc("high")] = 102; ohlc.iloc[81, ohlc.columns.get_loc("low")] = 98
    trades = pd.DataFrame({"time":ohlc.index,"price":100.0,"qty":1.0,"side":"buy"})
    ledger, stats = run_event_backtest(ohlc, trades, predictor=RecordingPredictor(99,101), fee_bps=0, slippage_bps=0)
    assert ledger.iloc[0].exit_reason == "stop"
    assert stats["same_bar_collisions"] == 1
    assert stats["average_r"] is not None and stats["average_hold_minutes"] == 0
    assert stats["wins"] + stats["losses"] == stats["trades"]


def test_open_position_is_force_closed_or_reported():
    ohlc=bars(82); trades=pd.DataFrame({"time":ohlc.index,"price":100.,"qty":1.,"side":"buy"})
    ledger, stats = run_event_backtest(ohlc,trades,predictor=RecordingPredictor(),fee_bps=0,slippage_bps=0,force_close=True)
    assert ledger.iloc[-1].exit_reason == "end_of_data"
    ledger, stats = run_event_backtest(ohlc,trades,predictor=RecordingPredictor(),force_close=False)
    assert stats["open_position"] is not None
