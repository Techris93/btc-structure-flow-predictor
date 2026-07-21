import pandas as pd

from btc_predictor.footprint import cross_exchange_agreement, orderflow_features
from btc_predictor.models import Zone
from btc_predictor.strategy import detect_sweep
from btc_predictor.trade_store import TradeStore


def test_sweep_is_atr_bounded_and_can_reclaim_over_multiple_bars():
    idx=pd.date_range("2025-01-01",periods=5,freq="min",tz="UTC")
    x=pd.DataFrame({"open":100.,"high":[101]*5,"low":[99,99,94,95,99],"close":[100,100,95,98,101],"volume":1.},index=idx)
    z=Zone("z","swing","below",96,97,1,idx[0],idx[0])
    result=detect_sweep(x,z,"bullish",10,.05,2,3)
    assert result["confirmed"] and result["reclaim_time"]==idx[-2]
    assert result["depth_atr"]==.2
    assert not detect_sweep(x,z,"bullish",1,.05,1,3)["confirmed"]


def test_orderflow_reversals_are_symmetric_and_exchange_agreement_is_required():
    idx=pd.date_range("2025-01-01",periods=3,freq="min",tz="UTC")
    trades=[]
    for exchange in ("binance","bybit"):
        trades += [{"time":idx[0]-pd.Timedelta(seconds=10),"price":100,"qty":2,"side":"buy","exchange":exchange},
                   {"time":idx[1]-pd.Timedelta(seconds=10),"price":99,"qty":3,"side":"sell","exchange":exchange}]
    t=pd.DataFrame(trades); f=orderflow_features(t,window=2)
    assert f.bearish_delta_reversal.iloc[-1]
    agrees,deltas=cross_exchange_agreement(t,idx[0]-pd.Timedelta(minutes=1),idx[1],"bearish")
    assert agrees and set(deltas)=={"binance","bybit"}


def test_trade_store_deduplicates_and_persists(tmp_path):
    path=tmp_path/"trades.sqlite3"; store=TradeStore(path); now=pd.Timestamp("2025-01-01",tz="UTC")
    row=pd.DataFrame([{"time":now,"price":100,"qty":1,"side":"buy","exchange":"binance","trade_id":"1"}])
    assert store.append(row)==1 and store.append(row)==0
    reopened=TradeStore(path); out=reopened.query(now-pd.Timedelta(seconds=1),now+pd.Timedelta(seconds=1))
    assert len(out)==1 and out.iloc[0].exchange=="binance"
    assert reopened.stats()["binance"]["trades"]==1
    reopened.set_collector_status("binance",connected=True,mode="spot_market_data")
    assert reopened.collector_status()["binance"]=={"connected":True,"mode":"spot_market_data"}


def test_trade_store_enforces_max_rows(tmp_path):
    path=tmp_path/"cap.sqlite3"; store=TradeStore(path, max_rows=5); now=pd.Timestamp("2025-01-01",tz="UTC")
    rows=[{"time":now+pd.Timedelta(seconds=i),"price":100+i,"qty":1,"side":"buy","exchange":"binance","trade_id":str(i)} for i in range(12)]
    assert store.append_rows(rows)==12
    store.prune()
    assert store.stats()["binance"]["trades"]==5
    out=store.query(now, now+pd.Timedelta(minutes=1), limit=3)
    assert len(out)==3



def test_detect_sweep_default_reclaim_window_is_fifteen_bars():
    idx = pd.date_range("2025-01-01", periods=20, freq="min", tz="UTC")
    lows = [99.0] * 20
    closes = [100.0] * 20
    highs = [101.0] * 20
    # Breach inside the default 15-bar lookback; keep closes unreclaimed until bar 17.
    lows[8] = 94.0
    for i in range(8, 17):
        closes[i] = 95.0
    closes[17] = 101.0
    x = pd.DataFrame({"open": 100.0, "high": highs, "low": lows, "close": closes, "volume": 1.0}, index=idx)
    z = Zone("z", "swing", "below", 96, 97, 1, idx[0], idx[0])
    result = detect_sweep(x, z, "bullish", 10)
    assert result["confirmed"] and result["reclaim_time"] == idx[17]
    # Explicit 3-bar window should miss this delayed reclaim.
    assert not detect_sweep(x, z, "bullish", 10, .05, 2, 3)["confirmed"]


def test_collector_status_marks_stale_feeds(tmp_path):
    store = TradeStore(tmp_path / "fresh.sqlite3")
    now = pd.Timestamp("2025-01-01T00:10:00Z")
    store.set_collector_status("binance", connected=True, mode="linear", latest="2025-01-01T00:00:00+00:00")
    store.set_collector_status("bybit", connected=True, mode="linear", latest="2025-01-01T00:09:50+00:00")
    status = store.collector_status(now=now, stale_after_seconds=90)
    assert status["binance"]["stale"] is True
    assert status["binance"]["fresh"] is False
    assert status["bybit"]["stale"] is False
    assert status["bybit"]["fresh"] is True


def test_collector_liveness_uses_last_message_at_not_only_trades(tmp_path):
    """Heartbeat streams (markPrice/kline) keep the feed fresh in quiet markets."""
    store = TradeStore(tmp_path / "heartbeat.sqlite3")
    now = pd.Timestamp("2025-01-01T00:10:00Z")
    # Trade event-time is old, but a message arrived 2s ago.
    store.set_collector_status(
        "binance",
        connected=True,
        mode="linear",
        latest="2025-01-01T00:05:00+00:00",
        last_message_at="2025-01-01T00:09:58+00:00",
    )
    status = store.collector_status(now=now, stale_after_seconds=90)
    assert status["binance"]["stale"] is False
    assert status["binance"]["fresh"] is True
    assert status["binance"]["lag_seconds"] <= 5


def test_flow_kline_buffer_returns_rest_shaped_frame(tmp_path):
    store = TradeStore(tmp_path / "klines.sqlite3")
    base = pd.Timestamp("2025-01-01T00:00:00Z")
    for i in range(3):
        open_ms = int((base + pd.Timedelta(minutes=i)).timestamp() * 1000)
        close_ms = int((base + pd.Timedelta(minutes=i + 1)).timestamp() * 1000)
        candle = {
            "open_time": open_ms,
            "close_time": close_ms,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.5 + i,
            "volume": 10.0 + i,
            "trades": 5 + i,
            "taker_buy_volume": 6.0 + i,
        }
        store.add_flow_kline("binance", candle, closed=i < 2)
    closed = store.flow_bars_df("binance", limit=10)
    assert len(closed) == 2  # current candle excluded by default
    assert list(closed.columns) == ["open", "high", "low", "close", "volume", "trades", "taker_buy_volume"]
    assert closed.index.name == "close_time" and closed.index.tz is not None
    assert closed["taker_buy_volume"].iloc[-1] == 7.0
    with_current = store.flow_bars_df("binance", limit=10, include_current=True)
    assert len(with_current) == 3
    assert with_current["taker_buy_volume"].iloc[-1] == 8.0
