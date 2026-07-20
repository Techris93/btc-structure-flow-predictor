from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time

import pandas as pd


logger = logging.getLogger("btc_predictor.trade_store")


class TradeStore:
    """Small durable rolling store shared by REST bootstrap and WebSocket collectors."""

    def __init__(self, path, max_rows: int = 120_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.max_rows = int(max_rows)
        self._collector_status = {}
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
        with self.lock, self._connect() as db:
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

    def query(self, start, end, limit: int | None = None):
        start_us = int(pd.Timestamp(start).value // 1000)
        end_us = int(pd.Timestamp(end).value // 1000)
        sql = (
            "SELECT time_us,price,qty,side,exchange,event_key FROM trades "
            "WHERE time_us>=? AND time_us<? ORDER BY time_us"
        )
        params = [start_us, end_us]
        if limit is not None:
            sql += " DESC LIMIT ?"
            params.append(int(limit))
        with self.lock, self._connect() as db:
            rows = db.execute(sql, params).fetchall()
        if not rows:
            return pd.DataFrame(columns=["time", "price", "qty", "side", "exchange", "trade_id"])
        if limit is not None:
            rows = list(reversed(rows))
        frame = pd.DataFrame(rows, columns=["time_us", "price", "qty", "side", "exchange", "trade_id"])
        frame["time"] = pd.to_datetime(frame.pop("time_us"), unit="us", utc=True)
        return frame

    def prune(self, before=None, max_rows: int | None = None):
        max_rows = self.max_rows if max_rows is None else int(max_rows)
        with self.lock, self._connect() as db:
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
        with self.lock, self._connect() as db:
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
        with self.lock:
            self._collector_status[name] = {**self._collector_status.get(name, {}), **values}

    def collector_status(self):
        with self.lock:
            return {name: dict(values) for name, values in self._collector_status.items()}


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

    endpoints = [
        ("futures", "wss://fstream.binance.com/ws/btcusdt@aggTrade"),
        ("spot_market_data", "wss://data-stream.binance.vision/ws/btcusdt@aggTrade"),
    ]
    endpoint_index = 0
    buffer = _BufferedAppender(store, flush_every=40, flush_seconds=1.0)
    while True:
        mode, url = endpoints[endpoint_index]
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=30, open_timeout=10) as ws:
                store.set_collector_status("binance", connected=True, mode=mode, error=None)
                while True:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=15)
                    except asyncio.TimeoutError:
                        buffer.flush()
                        raise
                    x = json.loads(message)
                    ts = pd.to_datetime(x["T"], unit="ms", utc=True)
                    buffer.add(
                        {
                            "time": ts,
                            "price": float(x["p"]),
                            "qty": float(x["q"]),
                            "side": "sell" if x["m"] else "buy",
                            "exchange": "binance",
                            "trade_id": f"{mode}:{x['a']}",
                        }
                    )
                    store.set_collector_status(
                        "binance",
                        connected=True,
                        mode=mode,
                        error=None,
                        latest=ts.isoformat(),
                    )
        except Exception as exc:
            buffer.flush()
            logger.warning("Binance %s WebSocket failed; rotating endpoint: %s", mode, exc)
            store.set_collector_status(
                "binance",
                connected=False,
                mode=mode,
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
            )
            endpoint_index = (endpoint_index + 1) % len(endpoints)
            await asyncio.sleep(2)


async def _bybit(store):
    import websockets

    url = "wss://stream.bybit.com/v5/public/linear"
    buffer = _BufferedAppender(store, flush_every=40, flush_seconds=1.0)
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=30) as ws:
                store.set_collector_status("bybit", connected=True, mode="linear", error=None)
                await ws.send(json.dumps({"op": "subscribe", "args": ["publicTrade.BTCUSDT"]}))
                async for message in ws:
                    payload = json.loads(message)
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
                            mode="linear",
                            error=None,
                            latest=latest.isoformat(),
                        )
        except Exception as exc:
            buffer.flush()
            logger.warning("Bybit WebSocket failed; reconnecting: %s", exc)
            store.set_collector_status(
                "bybit",
                connected=False,
                mode="linear",
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
            )
            await asyncio.sleep(2)


def start_collectors(store):
    async def collect():
        await asyncio.gather(_binance(store), _bybit(store))

    def run():
        asyncio.run(collect())

    thread = threading.Thread(target=run, name="trade-websockets", daemon=True)
    thread.start()
    return thread
