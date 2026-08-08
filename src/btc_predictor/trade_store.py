from __future__ import annotations

import asyncio
from collections import deque
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sqlite3
import threading
import time

import pandas as pd


logger = logging.getLogger("btc_predictor.trade_store")

BINANCE_WS_HEARTBEAT_TIMEOUT_SECONDS = max(
    10.0, float(os.getenv("BINANCE_WS_HEARTBEAT_TIMEOUT_SECONDS", "35"))
)
BINANCE_WS_RECONNECT_BASE_SECONDS = max(
    1.0,
    float(
        os.getenv(
            "BINANCE_WS_RECONNECT_BASE_SECONDS",
            os.getenv("BINANCE_WS_RECONNECT_SECONDS", "5"),
        )
    ),
)
BINANCE_WS_RECONNECT_MAX_SECONDS = max(
    BINANCE_WS_RECONNECT_BASE_SECONDS,
    float(os.getenv("BINANCE_WS_RECONNECT_MAX_SECONDS", "120")),
)
BINANCE_WS_RECONNECT_JITTER_MIN = max(
    0.0, float(os.getenv("BINANCE_WS_RECONNECT_JITTER_MIN", "0.80"))
)
BINANCE_WS_RECONNECT_JITTER_MAX = max(
    BINANCE_WS_RECONNECT_JITTER_MIN,
    float(os.getenv("BINANCE_WS_RECONNECT_JITTER_MAX", "1.20")),
)


def _configured_binance_ws_proxy():
    """Return the explicit Binance WebSocket proxy, if configured."""
    for key in (
        "BINANCE_WS_PROXY_URL",
        "QUOTAGUARDSTATIC_URL",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
    ):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return None


def _configured_bybit_ws_proxy():
    """Keep Bybit routing independent from Binance-specific proxy settings."""
    for key in ("BYBIT_WS_PROXY_URL", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return None


def _binance_reconnect_delay(attempt: int, uniform_fn=None) -> float:
    """Exponential reconnect delay with an explicit multiplicative jitter range."""
    core = min(
        BINANCE_WS_RECONNECT_MAX_SECONDS,
        BINANCE_WS_RECONNECT_BASE_SECONDS * (2 ** max(0, int(attempt))),
    )
    uniform = uniform_fn or random.uniform
    lower = min(
        BINANCE_WS_RECONNECT_MAX_SECONDS,
        core * BINANCE_WS_RECONNECT_JITTER_MIN,
    )
    upper = min(
        BINANCE_WS_RECONNECT_MAX_SECONDS,
        core * BINANCE_WS_RECONNECT_JITTER_MAX,
    )
    return float(
        uniform(lower, max(lower, upper))
    )


class TradeStore:
    """Small durable rolling store shared by REST bootstrap and WebSocket collectors."""

    def __init__(self, path, max_rows: int = 120_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # WAL allows readers to proceed while a collector commits. Keep
        # database writes and in-memory feed state on separate locks so busy
        # trade streams cannot starve predictor/status reads.
        self.db_write_lock = threading.Lock()
        self.state_lock = threading.RLock()
        self.max_rows = int(max_rows)
        self._collector_status = {}
        # Recent 1m klines collected from the Binance WebSocket, so footprint
        # baselines never need REST. Keyed by exchange.
        self._flow_bars = {}
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS trades ("
                "event_key TEXT PRIMARY KEY,"
                "time_us INTEGER NOT NULL,"
                "price REAL NOT NULL,"
                "qty REAL NOT NULL,"
                "side TEXT NOT NULL,"
                "exchange TEXT NOT NULL)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS trades_time ON trades(time_us)")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=20, check_same_thread=False)

    def append_rows(self, rows):
        if not rows:
            return 0
        prepared = []
        for row in rows:
            exchange = str(row.get("exchange", "unknown"))
            trade_id = row.get("trade_id")
            ts = pd.Timestamp(row["time"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            price = float(row["price"])
            qty = float(row["qty"])
            side = str(row["side"]).lower()
            raw = f"{exchange}|{trade_id or ''}|{ts.value}|{price}|{qty}|{side}"
            key = f"{exchange}:{trade_id}" if trade_id not in (None, "") else hashlib.sha1(raw.encode()).hexdigest()
            prepared.append((key, int(ts.value // 1000), price, qty, side, exchange))
        with self.db_write_lock, self._connect() as db:
            before = db.total_changes
            db.executemany("INSERT OR IGNORE INTO trades VALUES (?,?,?,?,?,?)", prepared)
            return db.total_changes - before

    def append(self, frame):
        if frame is None or getattr(frame, "empty", True):
            return 0
        rows = []
        for row in frame.itertuples(index=False):
            rows.append(
                {
                    "time": getattr(row, "time"),
                    "price": getattr(row, "price"),
                    "qty": getattr(row, "qty"),
                    "side": getattr(row, "side"),
                    "exchange": getattr(row, "exchange", "unknown"),
                    "trade_id": getattr(row, "trade_id", None),
                }
            )
        return self.append_rows(rows)

    def query(self, start, end, limit: int | None = None, include_trade_id: bool = True):
        start_us = int(pd.Timestamp(start).value // 1000)
        end_us = int(pd.Timestamp(end).value // 1000)
        selected = "time_us,price,qty,side,exchange,event_key" if include_trade_id else "time_us,price,qty,side,exchange"
        sql = f"SELECT {selected} FROM trades WHERE time_us>=? AND time_us<? ORDER BY time_us"
        params = [start_us, end_us]
        if limit is not None:
            sql += " DESC LIMIT ?"
            params.append(int(limit))
        with self._connect() as db:
            rows = db.execute(sql, params).fetchall()
        if not rows:
            columns = ["time", "price", "qty", "side", "exchange"]
            if include_trade_id:
                columns.append("trade_id")
            return pd.DataFrame(columns=columns)
        if limit is not None:
            rows = list(reversed(rows))
        columns = ["time_us", "price", "qty", "side", "exchange"]
        if include_trade_id:
            columns.append("trade_id")
        frame = pd.DataFrame(rows, columns=columns)
        frame["time"] = pd.to_datetime(frame.pop("time_us"), unit="us", utc=True)
        return frame

    def prune(self, before=None, max_rows: int | None = None):
        max_rows = self.max_rows if max_rows is None else int(max_rows)
        with self.db_write_lock, self._connect() as db:
            if before is not None:
                db.execute(
                    "DELETE FROM trades WHERE time_us<?",
                    (int(pd.Timestamp(before).value // 1000),),
                )
            count = db.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            if count > max_rows:
                cutoff = db.execute(
                    "SELECT time_us FROM trades ORDER BY time_us DESC LIMIT 1 OFFSET ?",
                    (max_rows - 1,),
                ).fetchone()
                if cutoff:
                    db.execute("DELETE FROM trades WHERE time_us<?", (cutoff[0],))
            try:
                db.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.OperationalError:
                pass

    def stats(self):
        with self._connect() as db:
            rows = db.execute(
                "SELECT exchange,COUNT(*),MIN(time_us),MAX(time_us) FROM trades GROUP BY exchange"
            ).fetchall()
        return {
            exchange: {
                "trades": count,
                "oldest": pd.to_datetime(oldest, unit="us", utc=True).isoformat(),
                "latest": pd.to_datetime(latest, unit="us", utc=True).isoformat(),
            }
            for exchange, count, oldest, latest in rows
        }

    def set_collector_status(self, name, **values):
        with self.state_lock:
            self._collector_status[name] = {**self._collector_status.get(name, {}), **values}

    def collector_status(self, now=None, stale_after_seconds: int | None = None):
        """Return collector status, optionally marking stale feeds.

        A collector can report connected=True while no trades arrive. Callers
        should treat freshness as part of health, not just the socket flag.
        """
        now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        with self.state_lock:
            out = {name: dict(values) for name, values in self._collector_status.items()}
        if stale_after_seconds is None:
            return out
        for name, values in out.items():
            # Liveness comes from the most recent message of ANY stream, not
            # only trades: markPrice/kline heartbeats keep the feed "fresh"
            # even in quiet markets. Event-time `latest` stays for data views.
            latest = values.get("last_message_at") or values.get("latest")
            lag_seconds = None
            if latest:
                latest_ts = pd.Timestamp(latest)
                if latest_ts.tzinfo is None:
                    latest_ts = latest_ts.tz_localize("UTC")
                lag_seconds = float((now - latest_ts).total_seconds())
            values["lag_seconds"] = None if lag_seconds is None else round(lag_seconds, 1)
            values["stale"] = lag_seconds is None or lag_seconds > float(stale_after_seconds)
            if values.get("connected") and values["stale"]:
                values["fresh"] = False
            else:
                values["fresh"] = bool(values.get("connected")) and not values["stale"]
        return out

    def exchange_latest(self, exchange: str):
        """Latest trade timestamp for an exchange from durable store stats."""
        stats = self.stats().get(exchange) or {}
        latest = stats.get("latest")
        return pd.Timestamp(latest) if latest else None

    # --- 1m kline buffer (WebSocket-derived flow baseline) -------------------

    def add_flow_kline(self, exchange: str, candle: dict, closed: bool):
        """Store a 1m kline update from the WebSocket.

        candle keys: open_time (ms), close_time (ms), open, high, low, close,
        volume, trades, taker_buy_volume (all floats except ms ints).
        When `closed` is True the candle is finalized into the recent deque.
        """
        exchange = str(exchange)
        with self.state_lock:
            slot = self._flow_bars.setdefault(exchange, {"current": None, "closed": deque(maxlen=240)})
            if closed:
                slot["closed"].append(candle)
                slot["current"] = None
            else:
                slot["current"] = candle

    def flow_bars_df(self, exchange: str = "binance", limit: int = 180, include_current: bool = False):
        """Recent 1m klines in the same shape as the REST klines frame.

        Index: close_time (UTC). Columns: open, high, low, close, volume,
        trades, taker_buy_volume. Closed candles only by default.
        """
        with self.state_lock:
            slot = self._flow_bars.get(exchange) or {"current": None, "closed": deque()}
            candles = list(slot["closed"])
            if include_current and slot["current"] is not None:
                candles.append(slot["current"])
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "trades", "taker_buy_volume"],
                                index=pd.DatetimeIndex([], name="close_time", tz="UTC"))
        frame = pd.DataFrame(candles)
        frame.index = pd.to_datetime(pd.to_numeric(frame.pop("close_time")), unit="ms", utc=True)
        frame.index.name = "close_time"
        frame = frame[["open", "high", "low", "close", "volume", "trades", "taker_buy_volume"]].astype(float)
        return frame.sort_index().tail(limit)


class _BufferedAppender:
    def __init__(self, store, flush_every=25, flush_seconds=1.0):
        self.store = store
        self.flush_every = flush_every
        self.flush_seconds = flush_seconds
        self.buffer = []
        self.last_flush = time.monotonic()

    def add(self, row):
        self.buffer.append(row)
        if len(self.buffer) >= self.flush_every or (time.monotonic() - self.last_flush) >= self.flush_seconds:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        rows = self.buffer
        self.buffer = []
        self.last_flush = time.monotonic()
        self.store.append_rows(rows)


async def _binance(store):
    import websockets
    import os

    # Binance futures WS limits: 300 connection attempts / 5 min / IP, and
    # 5 inbound messages/sec/connection. Keep one long-lived stream and avoid
    # reconnect storms (no REST spam; REST is separate and off by default).
    market_type = os.getenv("MARKET_TYPE", "linear").lower()
    if market_type == "linear":
        # One connection, three streams (limit: 1024 streams / connection):
        # - aggTrade: all trades, for order-flow deltas
        # - kline_1m: 1m OHLCV + taker-buy volume, for footprint baselines (no REST needed)
        # - markPrice@1s: heartbeat so liveness never depends on trade frequency
        url = (
            "wss://fstream.binance.com/market/stream?streams="
            "btcusdt@aggTrade/btcusdt@kline_1m/btcusdt@markPrice@1s"
        )
        mode = "linear"
    else:
        url = (
            "wss://data-stream.binance.vision/stream?streams="
            "btcusdt@aggTrade/btcusdt@kline_1m"
        )
        mode = "spot"
    # Preserve at-most-one-second latency while reducing SQLite transaction
    # churn during high-volume bursts.
    buffer = _BufferedAppender(store, flush_every=200, flush_seconds=1.0)
    proxy = _configured_binance_ws_proxy()
    reconnect_attempt = 0
    while True:
        received_frame = False
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=30,
                open_timeout=10,
                proxy=proxy,
            ) as ws:
                store.set_collector_status(
                    "binance",
                    connected=True,
                    mode=mode,
                    transport="proxy" if proxy else "direct",
                    proxy_required=False,
                    proxy_configured=bool(proxy),
                    error=None,
                )
                while True:
                    try:
                        message = await asyncio.wait_for(
                            ws.recv(),
                            timeout=BINANCE_WS_HEARTBEAT_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        buffer.flush()
                        error = (
                            "heartbeat_timeout_no_frames_"
                            f"{BINANCE_WS_HEARTBEAT_TIMEOUT_SECONDS:g}s"
                        )
                        logger.error("Binance %s WebSocket stale: %s", mode, error)
                        store.set_collector_status(
                            "binance",
                            connected=False,
                            mode=mode,
                            transport="proxy" if proxy else "direct",
                            proxy_required=False,
                            proxy_configured=bool(proxy),
                            error=error,
                        )
                        await ws.close()
                        raise TimeoutError(error)
                    if not received_frame:
                        received_frame = True
                        reconnect_attempt = 0
                    received_at = pd.Timestamp.now(tz="UTC")
                    x = json.loads(message)
                    envelope = x.get("data", x) if isinstance(x, dict) else {}
                    event = envelope.get("e")
                    store.set_collector_status(
                        "binance",
                        connected=True,
                        mode=mode,
                        transport="proxy" if proxy else "direct",
                        proxy_required=False,
                        proxy_configured=bool(proxy),
                        error=None,
                        last_message_at=received_at.isoformat(),
                    )
                    if event == "aggTrade":
                        ts = pd.to_datetime(envelope["T"], unit="ms", utc=True)
                        buffer.add(
                            {
                                "time": ts,
                                "price": float(envelope["p"]),
                                "qty": float(envelope["q"]),
                                "side": "sell" if envelope["m"] else "buy",
                                "exchange": "binance",
                                "trade_id": f"{mode}:{envelope['a']}",
                            }
                        )
                        store.set_collector_status(
                            "binance",
                            connected=True,
                            mode=mode,
                            transport="proxy" if proxy else "direct",
                            proxy_required=False,
                            proxy_configured=bool(proxy),
                            error=None,
                            latest=ts.isoformat(),
                        )
                    elif event == "kline":
                        k = envelope["k"]
                        store.add_flow_kline(
                            "binance",
                            {
                                "open_time": int(k["t"]),
                                "close_time": int(k["T"]),
                                "open": float(k["o"]),
                                "high": float(k["h"]),
                                "low": float(k["l"]),
                                "close": float(k["c"]),
                                "volume": float(k["v"]),
                                "trades": int(k["n"]),
                                "taker_buy_volume": float(k["V"]),
                            },
                            closed=bool(k.get("x")),
                        )
                        store.set_collector_status(
                            "binance",
                            connected=True,
                            mode=mode,
                            transport="proxy" if proxy else "direct",
                            proxy_required=False,
                            proxy_configured=bool(proxy),
                            error=None,
                            latest_kline_at=received_at.isoformat(),
                        )
                    elif event == "markPriceUpdate":
                        store.set_collector_status(
                            "binance",
                            connected=True,
                            mode=mode,
                            transport="proxy" if proxy else "direct",
                            proxy_required=False,
                            proxy_configured=bool(proxy),
                            error=None,
                            latest_mark_price=float(envelope["p"]),
                            latest_mark_price_at=received_at.isoformat(),
                        )
        except Exception as exc:
            buffer.flush()
            delay = _binance_reconnect_delay(reconnect_attempt)
            reconnect_attempt += 1
            logger.warning(
                "Binance %s WebSocket failed; reconnecting in %.2fs "
                "(attempt=%s jitter=%.2f..%.2f): %s",
                mode,
                delay,
                reconnect_attempt,
                BINANCE_WS_RECONNECT_JITTER_MIN,
                BINANCE_WS_RECONNECT_JITTER_MAX,
                exc,
            )
            store.set_collector_status(
                "binance",
                connected=False,
                mode=mode,
                transport="proxy" if proxy else "direct",
                proxy_required=False,
                proxy_configured=bool(proxy),
                reconnect_attempt=reconnect_attempt,
                reconnect_in_seconds=round(delay, 3),
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
            )
            await asyncio.sleep(delay)


async def _bybit(store):
    import websockets
    import os

    market_type = os.getenv("MARKET_TYPE", "linear").lower()
    url = (
        "wss://stream.bybit.com/v5/public/spot"
        if market_type == "spot"
        else "wss://stream.bybit.com/v5/public/linear"
    )
    mode = market_type if market_type in ("spot", "linear") else "spot"
    buffer = _BufferedAppender(store, flush_every=200, flush_seconds=1.0)
    while True:
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=30,
                proxy=_configured_bybit_ws_proxy() or True,
            ) as ws:
                store.set_collector_status("bybit", connected=True, mode=mode, error=None)
                await ws.send(json.dumps({"op": "subscribe", "args": ["publicTrade.BTCUSDT"]}))
                async for message in ws:
                    received_at = pd.Timestamp.now(tz="UTC")
                    payload = json.loads(message)
                    store.set_collector_status(
                        "bybit",
                        connected=True,
                        mode=mode,
                        error=None,
                        last_message_at=received_at.isoformat(),
                    )
                    data = payload.get("data", [])
                    if not data:
                        continue
                    latest = None
                    for x in data:
                        ts = pd.to_datetime(int(x["T"]), unit="ms", utc=True)
                        latest = ts if latest is None or ts > latest else latest
                        buffer.add(
                            {
                                "time": ts,
                                "price": float(x["p"]),
                                "qty": float(x["v"]),
                                "side": x["S"].lower(),
                                "exchange": "bybit",
                                "trade_id": str(x.get("i", "")),
                            }
                        )
                    if latest is not None:
                        store.set_collector_status(
                            "bybit",
                            connected=True,
                            mode=mode,
                            error=None,
                            latest=latest.isoformat(),
                        )
        except Exception as exc:
            buffer.flush()
            logger.warning("Bybit WebSocket failed; reconnecting: %s", exc)
            store.set_collector_status(
                "bybit",
                connected=False,
                mode=mode,
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
            )
            await asyncio.sleep(max(2, int(os.getenv("BYBIT_WS_RECONNECT_SECONDS", "5"))))


async def _supervise_collector(name, collector, store, restart_seconds):
    """Restart one exchange collector without terminating the other exchange."""
    while True:
        try:
            await collector(store)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("%s collector exited unexpectedly; restarting", name)
            await asyncio.sleep(restart_seconds)


def start_collectors(store):

    async def collect():
        await asyncio.gather(
            _supervise_collector(
                "binance", _binance, store, BINANCE_WS_RECONNECT_BASE_SECONDS
            ),
            _supervise_collector(
                "bybit",
                _bybit,
                store,
                max(2.0, float(os.getenv("BYBIT_WS_RECONNECT_SECONDS", "5"))),
            ),
        )

    def run():
        asyncio.run(collect())

    thread = threading.Thread(target=run, name="trade-websockets", daemon=True)
    thread.start()
    return thread
