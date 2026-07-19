from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sqlite3
import threading

import pandas as pd


class TradeStore:
    """Small durable rolling store shared by REST bootstrap and WebSocket collectors."""
    def __init__(self,path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.lock=threading.RLock()
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS trades (event_key TEXT PRIMARY KEY,time_us INTEGER NOT NULL,price REAL NOT NULL,qty REAL NOT NULL,side TEXT NOT NULL,exchange TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS trades_time ON trades(time_us)")

    def _connect(self): return sqlite3.connect(self.path,timeout=20)

    def append(self,frame):
        if frame is None or frame.empty:return 0
        x=frame.copy(); x["time"]=pd.to_datetime(x.time,utc=True)
        rows=[]
        for row in x.itertuples(index=False):
            exchange=str(getattr(row,"exchange","unknown")); trade_id=getattr(row,"trade_id",None)
            raw=f"{exchange}|{trade_id or ''}|{row.time.value}|{row.price}|{row.qty}|{row.side}"
            key=f"{exchange}:{trade_id}" if trade_id not in (None,"") else hashlib.sha1(raw.encode()).hexdigest()
            rows.append((key,int(row.time.value//1000),float(row.price),float(row.qty),str(row.side).lower(),exchange))
        with self.lock,self._connect() as db:
            before=db.total_changes; db.executemany("INSERT OR IGNORE INTO trades VALUES (?,?,?,?,?,?)",rows); return db.total_changes-before

    def query(self,start,end):
        start_us=int(pd.Timestamp(start).value//1000); end_us=int(pd.Timestamp(end).value//1000)
        with self.lock,self._connect() as db:
            rows=db.execute("SELECT time_us,price,qty,side,exchange,event_key FROM trades WHERE time_us>=? AND time_us<? ORDER BY time_us",(start_us,end_us)).fetchall()
        return pd.DataFrame(rows,columns=["time_us","price","qty","side","exchange","trade_id"]).assign(time=lambda x:pd.to_datetime(x.pop("time_us"),unit="us",utc=True)) if rows else pd.DataFrame(columns=["time","price","qty","side","exchange","trade_id"])

    def prune(self,before):
        with self.lock,self._connect() as db: db.execute("DELETE FROM trades WHERE time_us<?",(int(pd.Timestamp(before).value//1000),))


async def _binance(store):
    import websockets
    url="wss://fstream.binance.com/ws/btcusdt@aggTrade"
    while True:
        try:
            async with websockets.connect(url,ping_interval=20,ping_timeout=30) as ws:
                async for message in ws:
                    x=json.loads(message)
                    store.append(pd.DataFrame([{"time":pd.to_datetime(x["T"],unit="ms",utc=True),"price":float(x["p"]),"qty":float(x["q"]),"side":"sell" if x["m"] else "buy","exchange":"binance","trade_id":str(x["a"])}]))
        except Exception: await asyncio.sleep(2)


async def _bybit(store):
    import websockets
    url="wss://stream.bybit.com/v5/public/linear"
    while True:
        try:
            async with websockets.connect(url,ping_interval=20,ping_timeout=30) as ws:
                await ws.send(json.dumps({"op":"subscribe","args":["publicTrade.BTCUSDT"]}))
                async for message in ws:
                    payload=json.loads(message); data=payload.get("data",[])
                    if data:
                        store.append(pd.DataFrame([{"time":pd.to_datetime(int(x["T"]),unit="ms",utc=True),"price":float(x["p"]),"qty":float(x["v"]),"side":x["S"].lower(),"exchange":"bybit","trade_id":str(x.get("i",""))} for x in data]))
        except Exception: await asyncio.sleep(2)


def start_collectors(store):
    async def collect(): await asyncio.gather(_binance(store),_bybit(store))
    def run(): asyncio.run(collect())
    thread=threading.Thread(target=run,name="trade-websockets",daemon=True); thread.start(); return thread
