import pandas as pd

from btc_predictor.footprint import (
    _sweep_window_trades,
    cross_exchange_agreement,
    footprint_confirmation,
    orderflow_features,
)
from btc_predictor.models import Zone
from btc_predictor.strategy import detect_sweep, zone_reclaim_eligible
from btc_predictor.trade_store import TradeStore


def test_sweep_is_atr_bounded_and_can_reclaim_over_multiple_bars():
    idx=pd.date_range("2025-01-01",periods=5,freq="min",tz="UTC")
    x=pd.DataFrame({"open":100.,"high":[101]*5,"low":[99,99,94,95,99],"close":[100,100,95,98,101],"volume":1.},index=idx)
    z=Zone("z","swing","below",96,97,1,idx[0],idx[0])
    result=detect_sweep(x,z,"bullish",10,.05,2,3)
    assert result["confirmed"] and result["reclaim_time"]==idx[-2]
    assert result["depth_atr"]==.2
    assert not detect_sweep(x,z,"bullish",1,.05,1,3)["confirmed"]


def test_continuous_deeper_breach_keeps_original_sweep_identity():
    idx = pd.date_range("2026-01-01", periods=4, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [98.0] * 4,
            "high": [99.0] * 4,
            "low": [95.0, 94.0, 93.0, 98.0],
            "close": [95.5, 95.0, 98.0, 99.0],
            "volume": 1.0,
        },
        index=idx,
    )
    zone = Zone("stable", "swing", "below", 96, 97, 1, idx[0], idx[0])

    result = detect_sweep(frame, zone, "bullish", 10, reclaim_bars=15)

    assert result["confirmed"] is True
    assert result["time"] == idx[0]
    assert result["extreme"] == 93.0
    assert result["reclaim_time"] == idx[2]


def test_sweep_cannot_precede_zone_availability():
    idx = pd.date_range("2026-01-01 00:01", periods=5, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": [99.0, 94.0, 98.0, 99.0, 99.0],
            "close": [100.0, 95.0, 98.0, 100.0, 100.0],
            "volume": 1.0,
        },
        index=idx,
    )
    zone = Zone("future", "swing", "below", 96, 97, 1, idx[0], idx[3])

    result = detect_sweep(frame, zone, "bullish", 10, reclaim_bars=15)

    assert result == {"status": "none", "confirmed": False}


def test_sweep_bar_open_includes_trades_before_close_timestamp():
    decision_time = pd.Timestamp("2026-01-01 00:05:00Z")
    trades = pd.DataFrame(
        [
            {"time": decision_time - pd.Timedelta(seconds=50), "price": 100, "qty": 1, "side": "sell", "exchange": "binance"},
            {"time": decision_time - pd.Timedelta(seconds=10), "price": 101, "qty": 1, "side": "buy", "exchange": "bybit"},
        ]
    )

    retained = _sweep_window_trades(trades, decision_time, decision_time)

    assert len(retained) == 2


def test_invalidated_zone_keeps_full_declared_reclaim_window():
    idx = pd.date_range("2026-01-01", periods=70, freq="min", tz="UTC")
    zone = Zone(
        "grace",
        "swing",
        "below",
        96,
        97,
        1,
        idx[0],
        idx[0],
        swept_at=idx[2],
        invalidated_at=idx[2],
    )

    assert zone.is_active(idx[10]) is False
    assert zone_reclaim_eligible(zone, idx[10], reclaim_bars=60) is True
    assert zone_reclaim_eligible(zone, idx[63], reclaim_bars=60) is False
    frame = pd.DataFrame(
        {
            "open": 98.0,
            "high": 99.0,
            "low": [99.0, 94.0] + [95.0] * 8 + [98.0],
            "close": [98.0, 95.0] + [95.0] * 8 + [98.0],
            "volume": 1.0,
        },
        index=idx[:11],
    )
    sweep = detect_sweep(frame, zone, "bullish", 10, reclaim_bars=60)
    assert sweep["confirmed"] is True
    assert sweep["reclaim_time"] == idx[10]


def test_sweep_identity_rearms_only_after_three_clear_bars_and_atr_distance():
    idx = pd.date_range("2026-01-01", periods=6, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [98.0] * 6,
            "high": [99.0, 104.0, 104.0, 104.0, 99.0, 99.0],
            "low": [94.0, 98.0, 98.0, 98.0, 93.0, 98.0],
            "close": [98.0, 103.0, 103.0, 103.0, 98.0, 99.0],
            "volume": 1.0,
        },
        index=idx,
    )
    zone = Zone("rearmed", "swing", "below", 96, 97, 1, idx[0], idx[0])

    result = detect_sweep(frame, zone, "bullish", 10, reclaim_bars=15, rearm_bars=3, rearm_atr=.5)

    assert result["confirmed"] is True
    assert result["time"] == idx[4]


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


def test_cross_exchange_agreement_rejects_single_venue_window():
    end = pd.Timestamp("2026-01-01 00:05:00Z")
    trades = pd.DataFrame([
        {"time": end - pd.Timedelta(seconds=10), "price": 100, "qty": 2, "side": "buy", "exchange": "binance"}
    ])

    score, deltas = cross_exchange_agreement(trades, end - pd.Timedelta(minutes=1), end, "bullish")

    assert score == 0.0
    assert set(deltas) == {"binance"}


def test_price_impact_metric_is_honestly_named():
    idx = pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC")
    trades = pd.DataFrame([
        {"time": ts - pd.Timedelta(seconds=1), "price": 100 + i, "qty": 1, "side": "buy"}
        for i, ts in enumerate(idx)
    ])

    features = orderflow_features(trades, window=2)

    assert "price_impact_ratio" in features
    assert "low_price_impact_score" in features
    assert "kyle_lambda" not in features
    assert "kyle_absorption" not in features


def test_footprint_confirmation_passes_complete_agreement_window():
    idx = pd.date_range("2025-01-01", periods=30, freq="min", tz="UTC")
    trades = []
    for minute, timestamp in enumerate(idx):
        for exchange in ("binance", "bybit"):
            trades.append({
                "time": timestamp - pd.Timedelta(seconds=10),
                "price": 100.0 - minute * 0.1,
                "qty": 1.0,
                "side": "sell" if minute % 2 else "buy",
                "exchange": exchange,
            })
    confirmed, details = footprint_confirmation(
        pd.DataFrame(trades),
        None,
        "bearish",
        idx[-5],
        idx[-1],
    )

    assert isinstance(confirmed, bool)
    assert 0.0 <= details["agreement"] <= 1.0
    assert set(details["exchange_deltas"]) == {"binance", "bybit"}
    assert set(details["contributing_exchanges"]) == {"binance", "bybit"}


def test_trade_store_deduplicates_and_persists(tmp_path):
    path=tmp_path/"trades.sqlite3"; store=TradeStore(path); now=pd.Timestamp("2025-01-01",tz="UTC")
    row=pd.DataFrame([{"time":now,"price":100,"qty":1,"side":"buy","exchange":"binance","trade_id":"1"}])
    assert store.append(row)==1 and store.append(row)==0
    reopened=TradeStore(path); out=reopened.query(now-pd.Timedelta(seconds=1),now+pd.Timedelta(seconds=1))
    assert len(out)==1 and out.iloc[0].exchange=="binance"
    compact=reopened.query(now-pd.Timedelta(seconds=1),now+pd.Timedelta(seconds=1),include_trade_id=False)
    assert list(compact.columns)==["price","qty","side","exchange","time"]
    assert reopened.stats()["binance"]["trades"]==1
    reopened.set_collector_status("binance",connected=True,mode="spot_market_data")
    assert reopened.collector_status()["binance"]=={"connected":True,"mode":"spot_market_data"}


def test_trade_store_reads_do_not_wait_for_python_write_lock(tmp_path):
    store = TradeStore(tmp_path / "concurrent.sqlite3")
    now = pd.Timestamp("2025-01-01", tz="UTC")
    store.append_rows([
        {"time": now, "price": 100.0, "qty": 1.0, "side": "buy", "exchange": "bybit"}
    ])

    # A collector preparing another write must not block a WAL reader at the
    # Python layer. This guards the starvation that stopped booted polls.
    store.db_write_lock.acquire()
    try:
        out = store.query(now - pd.Timedelta(seconds=1), now + pd.Timedelta(seconds=1))
    finally:
        store.db_write_lock.release()
    assert len(out) == 1


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
