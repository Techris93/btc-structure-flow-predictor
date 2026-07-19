from __future__ import annotations

import base64
import fcntl
import hashlib
import hmac
import json
import os
import logging
import threading
import time

import pandas as pd
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from flask import Flask, jsonify, render_template, request

try:
    from pywebpush import webpush
except ImportError:
    webpush = None

from btc_predictor.persistence import JsonStore, runtime_dir
from btc_predictor.strategy import Predictor

app = Flask(__name__)
logger = logging.getLogger("btc_predictor")
predictor = Predictor()
data_dir = runtime_dir()
live_lock = threading.Lock()
push_lock = threading.Lock()
live_state = {"status":"starting","source":None,"prediction":None,"updated_at":None,"error":None}
live_thread_started = False
live_thread = None
_live_lock_handle = None

subscription_store = JsonStore(data_dir / "push_subscriptions.json")
push_state_store = JsonStore(data_dir / "push_state.json")
live_state_store = JsonStore(data_dir / "live_state.json")
research_status_store = JsonStore(os.getenv("BTC_RESEARCH_STATUS", str(data_dir / "research/status.json")))
push_subscriptions = subscription_store.read([])

vapid_path = data_dir / "vapid_private.pem"
if vapid_path.exists():
    _vapid_key = serialization.load_pem_private_key(vapid_path.read_bytes(), password=None)
else:
    _vapid_key = ec.generate_private_key(ec.SECP256R1())
    vapid_path.write_bytes(_vapid_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    os.chmod(vapid_path, 0o600)
_vapid_private_pem = _vapid_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
_vapid_public_key = base64.urlsafe_b64encode(_vapid_key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)).rstrip(b"=").decode()
_vapid_subject = os.getenv("VAPID_SUBJECT", "mailto:onyedikachristopher.agada@st.uskudar.edu.tr")
secret_path = data_dir / "push_test_secret"
if not secret_path.exists():
    secret_path.write_bytes(os.urandom(32)); os.chmod(secret_path, 0o600)
_push_secret = secret_path.read_bytes()


def _bybit_data():
    base = "https://api.bybit.com/v5/market"
    def candles(interval, limit="300"):
        response = requests.get(f"{base}/kline", params={"category":"linear","symbol":"BTCUSDT","interval":interval,"limit":limit}, timeout=10)
        response.raise_for_status(); rows = list(reversed(response.json()["result"]["list"]))
        frame = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","turnover"])
        duration = {"1":"1min","15":"15min","60":"1h","240":"4h"}[interval]
        frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame.timestamp), unit="ms", utc=True) + pd.Timedelta(duration)
        for column in ["open","high","low","close","volume"]: frame[column] = pd.to_numeric(frame[column])
        frame = frame.set_index("timestamp")
        return frame.loc[frame.index <= pd.Timestamp.now(tz="UTC")]
    ohlc, frames = candles("1"), {"15m":candles("15"),"1h":candles("60"),"4h":candles("240")}
    response = requests.get(f"{base}/recent-trade", params={"category":"linear","symbol":"BTCUSDT","limit":"1000"}, timeout=10)
    response.raise_for_status(); raw = response.json()["result"]["list"]
    trades = pd.DataFrame({"time":pd.to_datetime([int(x["time"]) for x in raw],unit="ms",utc=True),"price":[float(x["price"]) for x in raw],"qty":[float(x["size"]) for x in raw],"side":[x["side"].lower() for x in raw],"exchange":"bybit"})
    return ohlc, trades, frames


def _binance_trades():
    response = requests.get("https://fapi.binance.com/fapi/v1/aggTrades", params={"symbol":"BTCUSDT","limit":1000}, timeout=10)
    response.raise_for_status(); raw = response.json()
    return pd.DataFrame({"time":pd.to_datetime([x["T"] for x in raw],unit="ms",utc=True),"price":[float(x["p"]) for x in raw],"qty":[float(x["q"]) for x in raw],"side":["sell" if x["m"] else "buy" for x in raw],"exchange":"binance"})


def _persist_subscriptions():
    subscription_store.write(push_subscriptions)


def _send_push(payload, subscriptions=None):
    if webpush is None: return 0, 0
    with push_lock: targets = list(subscriptions if subscriptions is not None else push_subscriptions)
    sent, failed, stale = 0, 0, []
    for sub in targets:
        try:
            webpush(subscription_info=sub, data=json.dumps(payload), vapid_private_key=_vapid_private_pem, vapid_claims={"sub":_vapid_subject})
            sent += 1
        except Exception as exc:
            failed += 1
            if "404" in str(exc) or "410" in str(exc): stale.append(sub.get("endpoint"))
    if stale:
        with push_lock:
            push_subscriptions[:] = [s for s in push_subscriptions if s.get("endpoint") not in stale]
            _persist_subscriptions()
    return sent, failed


def _live_loop():
    global live_state
    persisted = push_state_store.read({})
    previous_key = persisted.get("key")
    last_sent = pd.Timestamp(persisted["sent_at"]) if persisted.get("sent_at") else None
    while True:
        try:
            with live_lock: live_state.update({"status":"polling","updated_at":pd.Timestamp.now(tz="UTC").isoformat()})
            ohlc, trades, frames = _bybit_data(); sources = "bybit"
            try:
                trades = pd.concat([trades, _binance_trades()], ignore_index=True); sources = "bybit+binance"
            except Exception: pass
            result = predictor.predict(ohlc, trades, 100_000, frames=frames)
            key = "|".join(str(getattr(result, k, None)) for k in ("bias","setup_type","zone","sweep_status","entry","stop","target"))
            now = pd.Timestamp.now(tz="UTC"); cooldown = pd.Timedelta(seconds=int(os.getenv("PUSH_COOLDOWN_SECONDS", "60")))
            if previous_key is not None and key != previous_key and (last_sent is None or now-last_sent >= cooldown):
                _send_push({"title":"BTC Predictor update","body":f"{result.bias.upper()} · {result.setup_type or result.no_trade_reason or result.sweep_status}"})
                last_sent = now
            previous_key = key
            push_state_store.write({"key":key,"sent_at":last_sent.isoformat() if last_sent is not None else None})
            next_state = {"status":"live","source":sources,"prediction":dict(result.__dict__),"updated_at":now.isoformat(),"error":None}
            live_state_store.write(next_state)
            with live_lock: live_state = next_state
        except Exception as exc:
            logger.exception("Live market poll failed")
            with live_lock: live_state.update({"status":"degraded","error":str(exc),"updated_at":pd.Timestamp.now(tz="UTC").isoformat()})
        time.sleep(max(15, int(os.getenv("LIVE_POLL_SECONDS", "30"))))


def start_live_loop():
    global live_thread_started, live_thread, _live_lock_handle
    if live_thread_started and live_thread is not None and live_thread.is_alive(): return True
    try:
        _live_lock_handle = open(data_dir / "live-loop.lock", "w")
        fcntl.flock(_live_lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError): return False
    live_thread_started = True
    live_thread = threading.Thread(target=_live_loop, name="live-predictor", daemon=True); live_thread.start()
    return True


@app.get("/")
@app.get("/dashboard")
def index(): return render_template("dashboard.html")


@app.get("/health")
def health():
    start_live_loop()
    with live_lock: state = dict(live_state)
    return jsonify({"status":"ok","service":"btc-structure-flow-predictor","paper_only":True,"market_feed":state["status"],"live_loop_owner":live_thread_started,"live_thread_alive":bool(live_thread and live_thread.is_alive())})


@app.get("/api/live")
def api_live():
    start_live_loop()
    with live_lock: state = dict(live_state)
    if not live_thread_started: state = live_state_store.read(state)
    if state.get("prediction"):
        state["prediction"] = dict(state["prediction"]); state["prediction"]["timestamp"] = str(state["prediction"]["timestamp"])
    return jsonify({"paper_only":True,**state})


@app.get("/api/backtest/one-year")
def backtest_status(): return jsonify(research_status_store.read({"status":"idle","note":"Run the separate research worker."}))


@app.post("/api/backtest/one-year")
def no_web_backtest(): return jsonify({"error":"Research is disabled in the web process; use the authenticated worker job."}), 409


@app.get("/sw.js")
def service_worker():
    script = "self.addEventListener('push',e=>{let d=e.data?e.data.json():{};e.waitUntil(self.registration.showNotification(d.title||'BTC Predictor',{body:d.body||'Prediction update',icon:'/favicon.ico',tag:'btc-predictor'}));});self.addEventListener('notificationclick',e=>{e.notification.close();e.waitUntil(clients.openWindow('/'));});"
    return script, 200, {"Content-Type":"application/javascript","Service-Worker-Allowed":"/","Cache-Control":"no-store"}


@app.get("/push/config")
def push_config(): return jsonify({"supported":webpush is not None,"vapid_public_key":_vapid_public_key})


@app.post("/push/subscribe")
def push_subscribe():
    data = request.get_json(silent=True) or {}
    if not data.get("endpoint") or not data.get("keys"): return jsonify({"error":"invalid subscription"}), 400
    with push_lock:
        if not any(x.get("endpoint") == data["endpoint"] for x in push_subscriptions): push_subscriptions.append(data)
        _persist_subscriptions()
    token = hmac.new(_push_secret, data["endpoint"].encode(), hashlib.sha256).hexdigest()
    return jsonify({"ok":True,"subscriptions":len(push_subscriptions),"test_token":token})


@app.post("/push/test")
def push_test():
    data = request.get_json(silent=True) or {}; endpoint, token = data.get("endpoint", ""), data.get("test_token", "")
    expected = hmac.new(_push_secret, endpoint.encode(), hashlib.sha256).hexdigest()
    if not endpoint or not hmac.compare_digest(token, expected): return jsonify({"error":"unauthorized"}), 401
    if webpush is None: return jsonify({"error":"pywebpush unavailable"}), 503
    with push_lock: target = [s for s in push_subscriptions if s.get("endpoint") == endpoint]
    sent, failed = _send_push({"title":"BTC Predictor test","body":"Web Push is connected and delivering notifications."}, target)
    return jsonify({"ok":sent > 0,"sent":sent,"failed":failed})


@app.post("/predict")
def predict():
    admin_token = os.getenv("ADMIN_API_TOKEN")
    supplied = request.headers.get("Authorization", "").removeprefix("Bearer ")
    if not admin_token or not hmac.compare_digest(supplied, admin_token): return jsonify({"error":"unauthorized"}), 401
    payload = request.get_json(silent=True) or {}
    try:
        ohlc, trades = pd.DataFrame(payload["ohlc"]), pd.DataFrame(payload["trades"])
        timestamp = "timestamp" if "timestamp" in ohlc else "time"
        ohlc.index = pd.to_datetime(ohlc.pop(timestamp), utc=True)
        result = predictor.predict(ohlc, trades, float(payload.get("equity", 100_000)))
        output = dict(result.__dict__); output["timestamp"] = str(output["timestamp"])
        return jsonify({"paper_only":True,**output})
    except (KeyError, TypeError, ValueError, IndexError) as exc: return jsonify({"error":str(exc)}), 400


@app.after_request
def no_cache(response):
    if request.path in ("/","/dashboard","/sw.js"): response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


if __name__ == "__main__": app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
