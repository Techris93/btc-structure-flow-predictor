import asyncio
import json

import pytest

from btc_predictor.binance_backfill import BinanceBackfillController
from btc_predictor.binance_rest import (
    BinanceRateLimited,
    BinanceRestConfig,
    BinanceRestDeferred,
    BinanceRestLimiter,
)
import btc_predictor.trade_store as trade_store_module


class FakeClock:
    def __init__(self, value=0.0):
        self.value = float(value)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


def test_rest_token_bucket_stays_below_configured_weight_cap():
    monotonic = FakeClock()
    wall = FakeClock(1_700_000_000)
    limiter = BinanceRestLimiter(
        BinanceRestConfig(capacity_weight=100, window_seconds=60),
        monotonic_fn=monotonic,
        wall_time_fn=wall,
    )

    limiter.acquire(80)
    with pytest.raises(BinanceRestDeferred) as deferred:
        limiter.acquire(30)
    assert deferred.value.reason == "binance_weight_budget"
    assert deferred.value.retry_after_seconds == pytest.approx(6.0)

    monotonic.advance(6)
    wall.advance(6)
    limiter.acquire(30)
    assert limiter.snapshot()["requests_attempted"] == 2


def test_rest_weight_headers_and_real_429_start_exact_cooldown():
    monotonic = FakeClock()
    wall = FakeClock(1_700_000_000)
    limiter = BinanceRestLimiter(
        BinanceRestConfig(capacity_weight=1200, window_seconds=60),
        monotonic_fn=monotonic,
        wall_time_fn=wall,
    )

    limiter.acquire(20)
    limiter.observe_response(200, {"X-MBX-USED-WEIGHT-1M": "321"})
    assert limiter.snapshot()["server_used_weight_1m"] == 321

    limiter.acquire(2)
    with pytest.raises(BinanceRateLimited) as limited:
        limiter.observe_response(429, {"Retry-After": "7"})
    assert limited.value.retry_after_seconds == 7
    assert limiter.snapshot()["cooldown_remaining_seconds"] == 7

    with pytest.raises(BinanceRestDeferred) as deferred:
        limiter.acquire(1)
    assert deferred.value.retry_after_seconds == 7
    monotonic.advance(7)
    wall.advance(7)
    limiter.acquire(1)


def test_418_uses_extended_default_only_after_real_response():
    monotonic = FakeClock()
    wall = FakeClock(1_700_000_000)
    limiter = BinanceRestLimiter(
        BinanceRestConfig(
            capacity_weight=1200,
            window_seconds=60,
            default_418_cooldown_seconds=600,
        ),
        monotonic_fn=monotonic,
        wall_time_fn=wall,
    )
    assert limiter.snapshot()["cooldown"] is False
    limiter.acquire(20)
    with pytest.raises(BinanceRateLimited) as limited:
        limiter.observe_response(418, {})
    assert limited.value.retry_after_seconds == 600
    assert limiter.snapshot()["cooldown_remaining_seconds"] == 600


def test_backfill_runs_once_at_startup_and_once_per_detected_gap():
    controller = BinanceBackfillController(gap_seconds=45)
    start = "2026-08-08T12:00:00Z"

    assert controller.decide(start, websocket_fresh=False) == "startup"
    controller.mark_success("startup", start, websocket_fresh=False)
    assert controller.decide("2026-08-08T12:10:00Z", websocket_fresh=False) is None

    assert controller.decide("2026-08-08T12:11:00Z", websocket_fresh=True) is None
    assert controller.decide("2026-08-08T12:12:00Z", websocket_fresh=False) is None
    assert controller.decide("2026-08-08T12:12:44Z", websocket_fresh=False) is None
    assert controller.decide("2026-08-08T12:12:45Z", websocket_fresh=False) == "gap_recovery"
    controller.mark_success(
        "gap_recovery", "2026-08-08T12:12:45Z", websocket_fresh=False
    )
    assert controller.decide("2026-08-08T13:00:00Z", websocket_fresh=False) is None


def test_reconnect_backoff_is_exponential_jittered_and_max_bounded(monkeypatch):
    monkeypatch.setattr(trade_store_module, "BINANCE_WS_RECONNECT_BASE_SECONDS", 5.0)
    monkeypatch.setattr(trade_store_module, "BINANCE_WS_RECONNECT_MAX_SECONDS", 120.0)
    monkeypatch.setattr(trade_store_module, "BINANCE_WS_RECONNECT_JITTER_MIN", 0.8)
    monkeypatch.setattr(trade_store_module, "BINANCE_WS_RECONNECT_JITTER_MAX", 1.2)

    assert trade_store_module._binance_reconnect_delay(0, lambda low, high: low) == 4.0
    assert trade_store_module._binance_reconnect_delay(1, lambda low, high: high) == 12.0
    assert trade_store_module._binance_reconnect_delay(20, lambda low, high: high) == 120.0


class FakeStore:
    def __init__(self):
        self.status = {}

    def set_collector_status(self, name, **values):
        self.status[name] = {**self.status.get(name, {}), **values}

    def append_rows(self, _rows):
        return 0

    def add_flow_kline(self, *_args, **_kwargs):
        return None


def test_silent_binance_websocket_forces_reconnect_within_backoff(monkeypatch):
    import websockets

    store = FakeStore()
    connections = []

    class FakeWebSocket:
        def __init__(self, silent):
            self.silent = silent
            self.closed = False

        async def recv(self):
            if self.silent:
                await asyncio.sleep(10)
            await asyncio.sleep(0.001)
            return json.dumps(
                {"data": {"e": "markPriceUpdate", "p": "65000.0"}}
            )

        async def close(self):
            self.closed = True

    class FakeConnection:
        def __init__(self, websocket):
            self.websocket = websocket

        async def __aenter__(self):
            return self.websocket

        async def __aexit__(self, *_args):
            return False

    def fake_connect(url, **kwargs):
        ws = FakeWebSocket(silent=not connections)
        connections.append((url, kwargs, ws))
        return FakeConnection(ws)

    monkeypatch.setenv("MARKET_TYPE", "linear")
    for key in (
        "BINANCE_WS_PROXY_URL",
        "QUOTAGUARDSTATIC_URL",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(websockets, "connect", fake_connect)
    monkeypatch.setattr(
        trade_store_module, "BINANCE_WS_HEARTBEAT_TIMEOUT_SECONDS", 0.01
    )
    monkeypatch.setattr(
        trade_store_module, "BINANCE_WS_RECONNECT_BASE_SECONDS", 0.01
    )
    monkeypatch.setattr(
        trade_store_module, "BINANCE_WS_RECONNECT_MAX_SECONDS", 0.02
    )
    monkeypatch.setattr(trade_store_module, "BINANCE_WS_RECONNECT_JITTER_MIN", 1.0)
    monkeypatch.setattr(trade_store_module, "BINANCE_WS_RECONNECT_JITTER_MAX", 1.0)

    async def scenario():
        task = asyncio.create_task(trade_store_module._binance(store))
        try:
            for _ in range(100):
                if len(connections) >= 2 and store.status.get("binance", {}).get(
                    "last_message_at"
                ):
                    break
                await asyncio.sleep(0.002)
            assert len(connections) >= 2
            assert connections[0][2].closed is True
            assert connections[0][1]["proxy"] is None
            assert "aggTrade" in connections[0][0]
            assert "kline_1m" in connections[0][0]
            assert "markPrice@1s" in connections[0][0]
            assert store.status["binance"]["transport"] == "direct"
            assert store.status["binance"]["proxy_required"] is False
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


def test_bybit_supervisor_continues_when_binance_collector_dies():
    calls = {"binance": 0, "bybit": 0}

    async def dying_binance(_store):
        calls["binance"] += 1
        raise RuntimeError("deliberate Binance failure")

    async def healthy_bybit(_store):
        while True:
            calls["bybit"] += 1
            await asyncio.sleep(0.001)

    async def scenario():
        binance_task = asyncio.create_task(
            trade_store_module._supervise_collector(
                "binance", dying_binance, object(), 0.002
            )
        )
        bybit_task = asyncio.create_task(
            trade_store_module._supervise_collector(
                "bybit", healthy_bybit, object(), 0.002
            )
        )
        deadline = asyncio.get_running_loop().time() + 0.2
        while (calls["binance"] <= 1 or calls["bybit"] <= 5) and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.002)
        assert calls["binance"] > 1
        assert calls["bybit"] > 5
        for task in (binance_task, bybit_task):
            task.cancel()
        await asyncio.gather(binance_task, bybit_task, return_exceptions=True)

    asyncio.run(scenario())
