from __future__ import annotations

import os
import pandas as pd
from flask import Flask, jsonify, render_template, request

from btc_predictor.strategy import Predictor
from btc_predictor.synthetic import make_synthetic

app = Flask(__name__)
predictor = Predictor()


@app.get("/dashboard")
@app.get("/")
def index():
    return render_template("dashboard.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "btc-structure-flow-predictor", "paper_only": True})


@app.get("/demo")
def demo():
    ohlc, trades = make_synthetic(days=1)
    ohlc = ohlc.iloc[-240:]
    trades = trades[trades.time >= ohlc.index[0]]
    result = predictor.predict(ohlc, trades, 100000)
    output = dict(result.__dict__)
    output["timestamp"] = str(output["timestamp"])
    return jsonify({"paper_only": True, "source": "synthetic_demo", **output})


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
