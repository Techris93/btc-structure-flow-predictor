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
from btc_predictor.trade_store import TradeStore, start_collectors
from btc_predictor.paper_position import PaperLedger

app = Flask(__name__)
logger = logging.getLogger("btc_predictor")
predictor = Predictor()
data_dir = runtime_dir()
live_lock = threading.Lock()
push_lock = threading.Lock()
live_state = {"status":"starting","source":None,"prediction":None,"updated_at":None,"error":None}
live_thread_started = False
live_thread = None
collector_thread = None
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
_vapid_public_key = base64.urlsafe_b64encode(_vapid_key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)).rstrip(b"=").decode()
_vapid_subject = os.getenv("VAPID_SUBJECT", "mailto:onyedikachristopher.agada@st.uskudar.edu.tr")
secret_path = data_dir / "push_test_secret"
if not secret_path.exists():
    secret_path.write_bytes(os.urandom(32)); os.chmod(secret_path, 0o600)
_push_secret = secret_path.read_bytes()
trade_store = TradeStore(data_dir / "live_trades.sqlite3", max_rows=int(os.getenv("TRADE_STORE_MAX_ROWS", "80000")))
paper_ledger = PaperLedger(os.getenv("PAPER_LEDGER_PATH", str(data_dir / "paper_ledger.json")))

MARKET_TYPE = os.getenv("MARKET_TYPE", "linear").lower()
if MARKET_TYPE not in ("spot", "linear"):
    MARKET_TYPE = "linear"
BINANCE_REST_ENABLED = os.getenv("BINANCE_REST_ENABLED", "0" if MARKET_TYPE == "linear" else "1").lower() in ("1", "true", "yes", "on")
BINANCE_REST_MINUTES = max(5, int(os.getenv("BINANCE_REST_MINUTES", "15")))
BINANCE_TRADE_LIMIT = max(50, min(200, int(os.getenv("BINANCE_TRADE_LIMIT", "100"))))
BINANCE_FLOW_LIMIT = max(30, min(180, int(os.getenv("BINANCE_FLOW_LIMIT", "60"))))
BINANCE_USER_AGENT = os.getenv("BINANCE_USER_AGENT", "btc-structure-flow-predictor/0.1 (+local-research)")
COLLECTOR_STALE_SECONDS = max(30, int(os.getenv("COLLECTOR_STALE_SECONDS", "90")))
# Rare REST backfill only when the WebSocket trade stream is stale. Still off by default
# for normal operation; set BINANCE_REST_ON_STALE=1 to enable emergency recovery.
BINANCE_REST_ON_STALE = os.getenv("BINANCE_REST_ON_STALE", "1").lower() in ("1", "true", "yes", "on")
BINANCE_STALE_REST_MINUTES = max(5, int(os.getenv("BINANCE_STALE_REST_MINUTES", "10")))


def _http_get(url, params=None, timeout=10):
    headers = {"User-Agent": BINANCE_USER_AGENT, "Accept": "application/json"}
    response = requests.get(url, params=params or {}, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


def _bybit_data():
    base = "https://api.bybit.com/v5/market"
    def candles(interval, limit="300"):
        response = _http_get(f"{base}/kline", params={"category":MARKET_TYPE,"symbol":"BTCUSDT","interval":interval,"limit":limit}, timeout=10)
        response.raise_for_status(); rows = list(reversed(response.json()["result"]["list"]))
        frame = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","turnover"])
        duration = {"1":"1min","15":"15min","60":"1h","240":"4h"}[interval]
        frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame.timestamp), unit="ms", utc=True) + pd.Timedelta(duration)
        for column in ["open","high","low","close","volume"]: frame[column] = pd.to_numeric(frame[column])
        frame = frame.set_index("timestamp")
        return frame.loc[frame.index <= pd.Timestamp.now(tz="UTC")]
    ohlc, frames = candles("1","180"), {"15m":candles("15","400"),"1h":candles("60","150"),"4h":candles("240","120")}
    response = _http_get(f"{base}/recent-trade", params={"category":MARKET_TYPE,"symbol":"BTCUSDT","limit":"1000"}, timeout=10)
    response.raise_for_status(); raw = response.json()["result"]["list"]
    trades = pd.DataFrame({"time":pd.to_datetime([int(x["time"]) for x in raw],unit="ms",utc=True),"price":[float(x["price"]) for x in raw],"qty":[float(x["size"]) for x in raw],"side":[x["side"].lower() for x in raw],"exchange":"bybit","trade_id":[str(x.get("execId",x.get("i",""))) for x in raw]})
    return ohlc, trades, frames


def _binance_trades():
    """Sparse REST backfill only. Prefer WebSocket collectors for live flow."""
    if MARKET_TYPE == "linear":
        url = "https://fapi.binance.com/fapi/v1/aggTrades"
        mode = "linear"
    else:
        url = "https://data-api.binance.vision/api/v3/aggTrades"
        mode = "spot"
    response = _http_get(url, params={"symbol":"BTCUSDT","limit":BINANCE_TRADE_LIMIT}, timeout=10)
    raw = response.json()
    return pd.DataFrame({
        "time":pd.to_datetime([x["T"] for x in raw],unit="ms",utc=True),
        "price":[float(x["p"]) for x in raw],
        "qty":[float(x["q"]) for x in raw],
        "side":["sell" if x["m"] else "buy" for x in raw],
        "exchange":"binance",
        "trade_id":[f"{mode}:{x['a']}" for x in raw],
    })


def _binance_flow_bars(limit=None):
    limit = BINANCE_FLOW_LIMIT if limit is None else limit
    if MARKET_TYPE == "linear":
        url = "https://fapi.binance.com/fapi/v1/klines"
    else:
        url = "https://data-api.binance.vision/api/v3/klines"
    response = _http_get(url, params={"symbol":"BTCUSDT","interval":"1m","limit":limit}, timeout=10)
    raw = pd.DataFrame(response.json())
    frame = pd.DataFrame({
        "close_time":pd.to_datetime(pd.to_numeric(raw[6]),unit="ms",utc=True),
        "open":pd.to_numeric(raw[1]),
        "high":pd.to_numeric(raw[2]),
        "low":pd.to_numeric(raw[3]),
        "close":pd.to_numeric(raw[4]),
        "volume":pd.to_numeric(raw[5]),
        "trades":pd.to_numeric(raw[8]),
        "taker_buy_volume":pd.to_numeric(raw[9]),
    }).set_index("close_time")
    return frame.loc[frame.index<=pd.Timestamp.now(tz="UTC")]


def _persist_subscriptions():
    subscription_store.write(push_subscriptions)


def _upsert_subscription(subscriptions, subscription):
    for index, existing in enumerate(subscriptions):
        if existing.get("endpoint") == subscription.get("endpoint"):
            subscriptions[index] = subscription
            return False
    subscriptions.append(subscription)
    return True


def _send_push(payload, subscriptions=None):
    if webpush is None: return 0, 0
    with push_lock: targets = list(subscriptions if subscriptions is not None else push_subscriptions)
    sent, failed, stale = 0, 0, []
    for sub in targets:
        try:
            webpush(subscription_info=sub, data=json.dumps(payload), vapid_private_key=str(vapid_path), vapid_claims={"sub":_vapid_subject})
            sent += 1
        except Exception as exc:
            logger.warning("Web Push delivery failed: %s", exc)
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
    binance_rest_retry_at = pd.Timestamp(0, tz="UTC")
    flow_retry_at = pd.Timestamp(0, tz="UTC")
    flow_bars_cache = None
    while True:
        try:
            poll_started = pd.Timestamp.now(tz="UTC")
            with live_lock: live_state.update({"status":"polling","updated_at":poll_started.isoformat()})
            ohlc, recent, frames = _bybit_data(); sources = "bybit"; trade_store.append(recent)
            binance_latest = trade_store.exchange_latest("binance")
            binance_lag = None
            if binance_latest is not None:
                if binance_latest.tzinfo is None:
                    binance_latest = binance_latest.tz_localize("UTC")
                binance_lag = float((poll_started - binance_latest).total_seconds())
            binance_stale = binance_lag is None or binance_lag > COLLECTOR_STALE_SECONDS
            # Prefer WebSocket collectors. REST is rare:
            # - always-on only when BINANCE_REST_ENABLED=1
            # - or emergency stale recovery when BINANCE_REST_ON_STALE=1 and WS is stale
            rest_due = poll_started >= binance_rest_retry_at
            rest_allowed = BINANCE_REST_ENABLED or (BINANCE_REST_ON_STALE and binance_stale)
            if rest_allowed and rest_due:
                try:
                    inserted = trade_store.append(_binance_trades())
                    cooldown_min = BINANCE_REST_MINUTES if BINANCE_REST_ENABLED else BINANCE_STALE_REST_MINUTES
                    binance_rest_retry_at = poll_started + pd.Timedelta(minutes=cooldown_min)
                    if inserted:
                        logger.info("Binance REST backfill inserted %s trades (stale=%s)", inserted, binance_stale)
                except Exception as exc:
                    cooldown_min = max((BINANCE_REST_MINUTES if BINANCE_REST_ENABLED else BINANCE_STALE_REST_MINUTES) * 2, 30)
                    binance_rest_retry_at = poll_started + pd.Timedelta(minutes=cooldown_min)
                    logger.warning("Binance REST trades unavailable; WebSocket collector remains active: %s", exc)
            now=ohlc.index[-1]; trades=trade_store.query(now-pd.Timedelta(minutes=int(os.getenv("TRADE_LOOKBACK_MINUTES","90"))), now, limit=int(os.getenv("TRADE_QUERY_LIMIT","60000"))); flow_bars=None
            recent_trades=trades.loc[trades.time>=now-pd.Timedelta(minutes=2)] if "time" in trades else trades
            available_exchanges=set(recent_trades.exchange.astype(str)) if "exchange" in recent_trades else set()
            sources="+".join(exchange for exchange in ("bybit","binance") if exchange in available_exchanges) or sources
            # Prefer durable store timestamps when collector latest is empty/stale.
            store_stats = trade_store.stats()
            for exchange, values in trade_store.collector_status().items():
                store_latest = (store_stats.get(exchange) or {}).get("latest")
                if store_latest and (not values.get("latest") or str(store_latest) > str(values.get("latest"))):
                    trade_store.set_collector_status(exchange, latest=store_latest)
            collectors = trade_store.collector_status(poll_started, COLLECTOR_STALE_SECONDS)
            # Flow baseline: prefer WebSocket-derived Binance 1m klines (aggTrade
            # deltas + kline taker-buy volume arrive on the same connection).
            # REST klines remain an explicit fallback only.
            ws_flow = trade_store.flow_bars_df("binance", limit=180)
            if len(ws_flow) >= 2 and ws_flow.index[-1] >= now - pd.Timedelta(minutes=3):
                flow_bars = ws_flow.loc[ws_flow.index <= now]
                flow_source = "websocket"
            elif BINANCE_REST_ENABLED and poll_started >= flow_retry_at:
                try:
                    flow_bars_cache = _binance_flow_bars()
                    flow_retry_at = poll_started + pd.Timedelta(minutes=BINANCE_REST_MINUTES)
                except Exception as exc:
                    flow_retry_at = poll_started + pd.Timedelta(minutes=max(BINANCE_REST_MINUTES * 2, 30))
                    logger.warning("Binance flow baseline unavailable; using stored trades: %s", exc)
                if flow_bars_cache is not None and not flow_bars_cache.empty and flow_bars_cache.index[-1] >= now - pd.Timedelta(minutes=2):
                    flow_bars = flow_bars_cache.loc[flow_bars_cache.index <= now]
                    flow_source = "rest"
                else:
                    flow_source = None
            else:
                flow_source = None
            result = predictor.predict(ohlc, trades, 100_000, frames=frames, flow_bars=flow_bars)
            paper_status = paper_ledger.update(result, ohlc)
            trade_store.prune(now-pd.Timedelta(minutes=int(os.getenv("TRADE_RETENTION_MINUTES","120"))))
            key = "|".join(str(getattr(result,k,None)) for k in ("bias","regime_4h","regime_1h","setup_type","zone","sweep_status","orderflow_confirmation","orderflow_reason","entry","stop","target"))
            now = pd.Timestamp.now(tz="UTC"); cooldown = pd.Timedelta(seconds=int(os.getenv("PUSH_COOLDOWN_SECONDS", "60")))
            if previous_key is not None and key != previous_key and (last_sent is None or now-last_sent >= cooldown):
                _send_push({"title":"BTC Predictor update","body":f"{result.bias.upper()} · {result.setup_type or result.no_trade_reason or result.sweep_status}"})
                last_sent = now
            previous_key = key
            push_state_store.write({"key":key,"sent_at":last_sent.isoformat() if last_sent is not None else None})
            stale_exchanges = [name for name, values in collectors.items() if values.get("stale")]
            feed_status = "degraded" if stale_exchanges else "live"
            next_state = {
                "status": feed_status,
                "source": sources,
                "market_type": MARKET_TYPE,
                "prediction": dict(result.__dict__),
                "paper": paper_status,
                "binance_feed_mode": collectors.get("binance", {}).get("mode", "unknown"),
                "flow_source": flow_source,
                "collectors": collectors,
                "stale_exchanges": stale_exchanges,
                "updated_at": now.isoformat(),
                "error": (f"stale feeds: {', '.join(stale_exchanges)}" if stale_exchanges else None),
            }
            live_state_store.write(next_state)
            with live_lock: live_state = next_state
        except Exception as exc:
            logger.exception("Live market poll failed")
            with live_lock: live_state.update({"status":"degraded","error":str(exc),"updated_at":pd.Timestamp.now(tz="UTC").isoformat()})
        time.sleep(max(20, int(os.getenv("LIVE_POLL_SECONDS", "45"))))


def start_live_loop():
    global live_thread_started, live_thread, collector_thread, _live_lock_handle
    if live_thread_started and live_thread is not None and live_thread.is_alive(): return True
    try:
        _live_lock_handle = open(data_dir / "live-loop.lock", "w")
        fcntl.flock(_live_lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError): return False
    live_thread_started = True
    collector_thread=start_collectors(trade_store)
    live_thread = threading.Thread(target=_live_loop, name="live-predictor", daemon=True); live_thread.start()
    return True


@app.get("/")
@app.get("/dashboard")
def index(): return render_template("dashboard.html")


@app.get("/health")
def health():
    start_live_loop()
    with live_lock: state = dict(live_state)
    collectors = trade_store.collector_status(pd.Timestamp.now(tz="UTC"), COLLECTOR_STALE_SECONDS)
    # Prefer durable store timestamps when collector latest is empty.
    store_stats = trade_store.stats()
    for exchange, values in collectors.items():
        if not values.get("latest"):
            store_latest = (store_stats.get(exchange) or {}).get("latest")
            if store_latest:
                trade_store.set_collector_status(exchange, latest=store_latest)
    collectors = trade_store.collector_status(pd.Timestamp.now(tz="UTC"), COLLECTOR_STALE_SECONDS)
    stale_exchanges = [name for name, values in collectors.items() if values.get("stale")]
    return jsonify({
        "status": "degraded" if stale_exchanges else "ok",
        "service":"btc-structure-flow-predictor",
        "paper_only":True,
        "market_type":MARKET_TYPE,
        "binance_rest_enabled":BINANCE_REST_ENABLED,
        "binance_rest_on_stale":BINANCE_REST_ON_STALE,
        "collector_stale_seconds":COLLECTOR_STALE_SECONDS,
        "market_feed":state["status"],
        "live_loop_owner":live_thread_started,
        "live_thread_alive":bool(live_thread and live_thread.is_alive()),
        "trade_store":store_stats,
        "collectors":collectors,
        "stale_exchanges":stale_exchanges,
    })


@app.get("/api/live")
def api_live():
    start_live_loop()
    with live_lock: state = dict(live_state)
    if not live_thread_started: state = live_state_store.read(state)
    if state.get("prediction"):
        state["prediction"] = dict(state["prediction"]); state["prediction"]["timestamp"] = str(state["prediction"]["timestamp"])
    return jsonify({"paper_only":True,**state})


@app.get("/api/backtest/one-year")
def backtest_status():
    result = research_status_store.read({"status":"idle","note":"Run the separate research worker."})
    result_path = Path(os.getenv("BTC_RESEARCH_DIR", str(data_dir / "research"))) / "result.json"
    if result_path.exists():
        try:
            result = {**result, **JsonStore(result_path).read({})}
        except Exception:
            pass
    return jsonify(result)


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
        _upsert_subscription(push_subscriptions, data)
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
