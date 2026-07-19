from __future__ import annotations

import os
import threading
import time
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request

from btc_predictor.strategy import Predictor
from btc_predictor.synthetic import make_synthetic

app = Flask(__name__)
predictor = Predictor()
live_lock = threading.Lock()
live_state = {"status": "starting", "source": None, "prediction": None, "updated_at": None, "error": None}
live_thread_started = False


def _bybit_data():
    base = "https://api.bybit.com/v5/market"
    k = requests.get(f"{base}/kline", params={"category":"linear","symbol":"BTCUSDT","interval":"1","limit":"300"}, timeout=10).json()
    rows = list(reversed(k["result"]["list"]))
    ohlc = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","turnover"])
    ohlc["timestamp"] = pd.to_datetime(pd.to_numeric(ohlc.timestamp), unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]: ohlc[c] = pd.to_numeric(ohlc[c])
    ohlc = ohlc.set_index("timestamp")
    tr = requests.get(f"{base}/recent-trade", params={"category":"linear","symbol":"BTCUSDT","limit":"1000"}, timeout=10).json()["result"]["list"]
    trades = pd.DataFrame({"time":pd.to_datetime([int(x["time"]) for x in tr],unit="ms",utc=True),"price":[float(x["price"]) for x in tr],"qty":[float(x["size"]) for x in tr],"side":[x["side"].lower() for x in tr],"exchange":"bybit"})
    return ohlc, trades


def _binance_trades():
    tr = requests.get("https://fapi.binance.com/fapi/v1/aggTrades", params={"symbol":"BTCUSDT","limit":1000}, timeout=10).json()
    return pd.DataFrame({"time":pd.to_datetime([x["T"] for x in tr],unit="ms",utc=True),"price":[float(x["p"]) for x in tr],"qty":[float(x["q"]) for x in tr],"side":["sell" if x["m"] else "buy" for x in tr],"exchange":"binance"})


def _live_loop():
    global live_state
    while True:
        try:
            ohlc, trades = _bybit_data()
            sources = "bybit"
            try:
                trades = pd.concat([trades, _binance_trades()], ignore_index=True)
                sources = "bybit+binance"
            except Exception:
                pass
            result = predictor.predict(ohlc, trades, 100000)
            with live_lock:
                live_state = {"status":"live", "source":sources, "prediction":dict(result.__dict__), "updated_at":pd.Timestamp.utcnow().isoformat(), "error":None}
        except Exception as exc:
            with live_lock:
                live_state.update({"status":"degraded", "error":str(exc), "updated_at":pd.Timestamp.utcnow().isoformat()})
        time.sleep(max(15, int(os.getenv("LIVE_POLL_SECONDS", "30"))))


def start_live_loop():
    global live_thread_started
    if not live_thread_started:
        live_thread_started = True
        threading.Thread(target=_live_loop, name="live-predictor", daemon=True).start()


start_live_loop()


@app.get("/dashboard")
@app.get("/")
def index():
    return render_template("dashboard.html")


@app.get("/health")
def health():
    with live_lock: state = dict(live_state)
    return jsonify({"status": "ok", "service": "btc-structure-flow-predictor", "paper_only": True, "market_feed": state["status"]})


@app.get("/api/live")
def api_live():
    with live_lock: state = dict(live_state)
    if state.get("prediction"):
        state["prediction"] = dict(state["prediction"])
        state["prediction"]["timestamp"] = str(state["prediction"]["timestamp"])
    return jsonify({"paper_only": True, **state})




@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        ohlc = pd.DataFrame(payload["ohlc"])
        trades = pd.DataFrame(payload["trades"])
        if "timestamp" in ohlc:
            ohlc.index = pd.to_datetime(ohlc.pop("timestamp"), utc=True)
        elif "time" in ohlc:
            ohlc.index = pd.to_datetime(ohlc.pop("time"), utc=True)
        else:
            raise ValueError("ohlc requires timestamp or time")
        required_ohlc = {"open", "high", "low", "close", "volume"}
        required_trades = {"time", "price", "qty", "side"}
        if not required_ohlc.issubset(ohlc) or not required_trades.issubset(trades):
            raise ValueError("missing required OHLC or trade columns")
        result = predictor.predict(ohlc, trades, float(payload.get("equity", 100000)))
        output = dict(result.__dict__)
        output["timestamp"] = str(output["timestamp"])
        return jsonify({"paper_only": True, **output})
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
