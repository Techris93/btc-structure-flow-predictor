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


class LimitPredictor:
    def __init__(self, limit=99.0, stop=90, target=110, valid_bars=3):
        self.limit=limit; self.stop=stop; self.target=target; self.valid_bars=valid_bars
    def predict(self, history, trades, equity, frames=None):
        now=history.index[-1]
        return PredictorOutput(now, "bullish", "reversal", "z", "confirmed", True,
                               self.limit, self.stop, self.target, 2, None, 1,
                               entry_type="limit",
                               entry_expires_at=(now+pd.Timedelta(minutes=self.valid_bars)).isoformat())


def test_limit_order_fills_only_on_later_touch():
    ohlc=bars()
    # First bar after the decision does not touch the limit; the next one does.
    ohlc.iloc[81, ohlc.columns.get_loc("low")] = 99.4
    ohlc.iloc[82, ohlc.columns.get_loc("low")] = 98.5
    trades=pd.DataFrame({"time":ohlc.index,"price":100.,"qty":1.,"side":"buy"})
    ledger, stats = run_event_backtest(ohlc, trades, predictor=LimitPredictor(), fee_bps=0, slippage_bps=0)
    assert stats["trades"] >= 1
    assert ledger.iloc[0].entry_time == ohlc.index[82]
    # Touch fills at the limit, not at the open.
    assert ledger.iloc[0].entry == pytest.approx(99.0)


def test_limit_order_expires_unfilled():
    ohlc=bars(88)
    trades=pd.DataFrame({"time":ohlc.index,"price":100.,"qty":1.,"side":"buy"})
    ledger, stats = run_event_backtest(ohlc, trades, predictor=LimitPredictor(limit=95.0, valid_bars=2), fee_bps=0, slippage_bps=0)
    assert stats["rejection_counts"].get("limit_expired", 0) >= 1


class RotatingLimitPredictor:
    """Alternates zones so each new setup supersedes the working order."""
    def __init__(self):
        self.calls = 0
    def predict(self, history, trades, equity, frames=None):
        self.calls += 1
        now = history.index[-1]
        zone = "z-a" if self.calls % 2 else "z-b"
        return PredictorOutput(now, "bullish", "reversal", zone, "confirmed", True,
                               95.0, 90.0, 110.0, 2, None, 1,
                               entry_type="limit",
                               entry_expires_at=(now+pd.Timedelta(minutes=60)).isoformat())


def test_new_setup_supersedes_working_limit_order():
    ohlc=bars(90)
    trades=pd.DataFrame({"time":ohlc.index,"price":100.,"qty":1.,"side":"buy"})
    predictor = RotatingLimitPredictor()
    ledger, stats = run_event_backtest(ohlc, trades, predictor=predictor, fee_bps=0, slippage_bps=0)
    # Limit at 95 never touched by 99.5-100.5 bars: no fills, but supersession counted.
    assert stats["rejection_counts"].get("pending_superseded", 0) >= 1
    assert stats["rejection_counts"].get("limit_expired", 0) == 0
