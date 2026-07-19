"""Public-feed adapters. Network collection is intentionally separated from features."""
from __future__ import annotations
import json, asyncio
from datetime import datetime, timezone
from .models import TradeEvent

def _utc(ms: int) -> datetime: return datetime.fromtimestamp(ms/1000, tz=timezone.utc)

def normalize_binance_trade(payload: dict) -> TradeEvent:
    return TradeEvent(_utc(payload["T"]), "binance", payload["s"], float(payload["p"]), float(payload["q"]), "sell" if payload["m"] else "buy", str(payload.get("a")))

def normalize_bybit_trade(payload: dict) -> TradeEvent:
    return TradeEvent(_utc(int(payload["T"])), "bybit", payload["s"], float(payload["p"]), float(payload["v"]), payload["S"].lower(), str(payload.get("i")))

async def websocket_trade_stream(url: str, parser, queue: asyncio.Queue, subscribe: dict | None = None):
    try:
        import websockets
    except ImportError as e: raise RuntimeError("Install websockets to enable live collection") from e
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=60) as ws:
                if subscribe: await ws.send(json.dumps(subscribe))
                async for message in ws: await queue.put(parser(json.loads(message)))
        except Exception:
            await asyncio.sleep(2)
