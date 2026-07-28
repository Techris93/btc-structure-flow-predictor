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
import uuid
from pathlib import Path
from urllib.parse import urlparse

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
from btc_predictor.signal_lifecycle import SignalLifecycle

app = Flask(__name__)
logger = logging.getLogger("btc_predictor")
predictor = Predictor()
data_dir = runtime_dir()
live_lock = threading.Lock()
push_lock = threading.Lock()
push_delivery_lock = threading.RLock()
push_decision_lock = threading.RLock()
live_start_lock = threading.Lock()
live_state = {"status":"starting","source":None,"prediction":None,"updated_at":None,"error":None}
live_thread_started = False
live_thread = None
collector_thread = None
_live_lock_handle = None
live_boot_thread = None

subscription_store = JsonStore(data_dir / "push_subscriptions.json")
push_delivery_store = JsonStore(data_dir / "push_delivery.json")
push_delivery_events_store = JsonStore(data_dir / "push_delivery_events.json")
push_decision_events_store = JsonStore(data_dir / "push_decision_events.json")
paper_exit_push_store = JsonStore(data_dir / "paper_exit_push.json")
signal_lifecycle_store = JsonStore(data_dir / "signal_lifecycle.json")
signal_event_queue_store = JsonStore(data_dir / "signal_event_queue.json")
live_state_store = JsonStore(data_dir / "live_state.json")
research_status_store = JsonStore(os.getenv("BTC_RESEARCH_STATUS", str(data_dir / "research/status.json")))
push_subscriptions = subscription_store.read([])

PUSH_ALLOWED_HOST_SUFFIXES = tuple(
    item.strip().lower()
    for item in os.getenv(
        "PUSH_ALLOWED_HOST_SUFFIXES",
        "push.apple.com,fcm.googleapis.com,push.services.mozilla.com,notify.windows.com",
    ).split(",")
    if item.strip()
)
PUSH_MAX_SUBSCRIPTIONS = max(1, int(os.getenv("PUSH_MAX_SUBSCRIPTIONS", "32")))
PUSH_SINGLE_INSTALLATION = os.getenv("PUSH_SINGLE_INSTALLATION", "1").lower() in (
    "1", "true", "yes", "on"
)
PUSH_ACK_RETRY_SECONDS = max(30, int(os.getenv("PUSH_ACK_RETRY_SECONDS", "90")))
PUSH_MAX_ACK_RETRIES = max(0, min(3, int(os.getenv("PUSH_MAX_ACK_RETRIES", "2"))))
PUSH_EVENT_RETENTION = max(100, int(os.getenv("PUSH_EVENT_RETENTION", "500")))
PUSH_STATE_COOLDOWN_SECONDS = max(0, int(os.getenv("PUSH_COOLDOWN_SECONDS", "60")))
SIGNAL_CONFIRM_OBSERVATIONS = max(1, int(os.getenv("SIGNAL_CONFIRM_OBSERVATIONS", "2")))
SIGNAL_INVALIDATION_OBSERVATIONS = max(1, int(os.getenv("SIGNAL_INVALIDATION_OBSERVATIONS", "3")))
BIAS_CONFIRM_OBSERVATIONS = max(1, int(os.getenv("BIAS_CONFIRM_OBSERVATIONS", "2")))
PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL", "https://btc-structure-flow-predictor.onrender.com"
).rstrip("/")

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
signal_lifecycle = SignalLifecycle(
    confirm_observations=SIGNAL_CONFIRM_OBSERVATIONS,
    invalidation_observations=SIGNAL_INVALIDATION_OBSERVATIONS,
    bias_observations=BIAS_CONFIRM_OBSERVATIONS,
)

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
# With WS silently dropped on some hosts, a ~1 min cadence keeps the feed fresh
# at roughly 22 weight/min, ~1% of the 2,400 weight/min fapi budget.
BINANCE_REST_ON_STALE = os.getenv("BINANCE_REST_ON_STALE", "1").lower() in ("1", "true", "yes", "on")
BINANCE_STALE_REST_MINUTES = max(1, int(os.getenv("BINANCE_STALE_REST_MINUTES", "3")))


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


def _utcnow():
    return pd.Timestamp.now(tz="UTC")


def _endpoint_hash(endpoint):
    return hashlib.sha256(str(endpoint).encode()).hexdigest()[:16]


def _subscription_info(subscription):
    return {
        "endpoint": subscription.get("endpoint"),
        "keys": subscription.get("keys") or {},
    }


def _normalize_subscription(subscription, now=None):
    now = now or _utcnow()
    normalized = {
        "endpoint": str(subscription.get("endpoint") or ""),
        "keys": {
            "auth": str((subscription.get("keys") or {}).get("auth") or ""),
            "p256dh": str((subscription.get("keys") or {}).get("p256dh") or ""),
        },
        "expirationTime": subscription.get("expirationTime"),
        "installation_id": str(subscription.get("installation_id") or "")[:128] or None,
        "created_at": subscription.get("created_at") or now.isoformat(),
        "last_seen_at": now.isoformat(),
        "last_ack_at": subscription.get("last_ack_at"),
        "ack_miss_count": int(subscription.get("ack_miss_count") or 0),
        "enabled": subscription.get("enabled") is not False,
        "user_agent": str(subscription.get("user_agent") or "")[:512] or None,
        "platform": str(subscription.get("platform") or "")[:128] or None,
        "app_mode": str(subscription.get("app_mode") or "")[:32] or None,
        "timezone": str(subscription.get("timezone") or "")[:128] or None,
        "status": subscription.get("status") or "unverified",
    }
    return normalized


def _upsert_subscription(subscriptions, subscription, now=None):
    normalized = _normalize_subscription(subscription, now=now)
    endpoint = normalized["endpoint"]
    installation_id = normalized.get("installation_id")
    exact_index = next(
        (index for index, item in enumerate(subscriptions) if item.get("endpoint") == endpoint),
        None,
    )
    installation_indexes = [
        index
        for index, item in enumerate(subscriptions)
        if installation_id and item.get("installation_id") == installation_id
    ]
    if exact_index is not None:
        existing = subscriptions[exact_index]
        normalized["created_at"] = existing.get("created_at") or normalized["created_at"]
        normalized["last_ack_at"] = existing.get("last_ack_at")
        normalized["ack_miss_count"] = int(existing.get("ack_miss_count") or 0)
        normalized["enabled"] = existing.get("enabled") is not False
        normalized["status"] = existing.get("status") or normalized["status"]
        subscriptions[exact_index] = normalized
        for index in reversed(installation_indexes):
            if index != exact_index:
                del subscriptions[index]
        return False
    if installation_indexes:
        first = installation_indexes[0]
        existing = subscriptions[first]
        normalized["created_at"] = existing.get("created_at") or normalized["created_at"]
        subscriptions[first] = normalized
        for index in reversed(installation_indexes[1:]):
            del subscriptions[index]
        return False
    subscriptions.append(normalized)
    return True


def _remove_subscription(subscriptions, endpoint):
    before = len(subscriptions)
    subscriptions[:] = [item for item in subscriptions if item.get("endpoint") != endpoint]
    return before - len(subscriptions)


def _is_allowed_push_endpoint(endpoint):
    try:
        parsed = urlparse(endpoint)
    except (TypeError, ValueError):
        return False
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return False
    host = parsed.hostname.lower().rstrip(".")
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in PUSH_ALLOWED_HOST_SUFFIXES)


def _valid_base64url(value, expected_length, required_prefix=None):
    if not isinstance(value, str) or not value or len(value) > 512:
        return False
    try:
        padding = "=" * ((4 - len(value) % 4) % 4)
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except Exception:
        return False
    return len(decoded) == expected_length and (
        required_prefix is None or decoded.startswith(required_prefix)
    )


def _validate_subscription(subscription):
    endpoint = subscription.get("endpoint")
    keys = subscription.get("keys") or {}
    installation_id = str(subscription.get("installation_id") or "")
    if not isinstance(endpoint, str) or len(endpoint) > 4096 or not _is_allowed_push_endpoint(endpoint):
        return "unsupported push endpoint"
    if not _valid_base64url(keys.get("auth"), 16) or not _valid_base64url(
        keys.get("p256dh"), 65, b"\x04"
    ):
        return "invalid push encryption keys"
    if installation_id and (
        len(installation_id) > 128
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in installation_id)
    ):
        return "invalid installation id"
    return None


def _issue_endpoint_token(endpoint, lifetime_seconds=900):
    expires = int(time.time()) + lifetime_seconds
    message = f"{expires}:{endpoint}".encode()
    signature = hmac.new(_push_secret, message, hashlib.sha256).hexdigest()
    return f"{expires}.{signature}"


def _verify_endpoint_token(endpoint, token):
    try:
        expires_text, supplied = str(token).split(".", 1)
        expires = int(expires_text)
    except (TypeError, ValueError):
        return False
    if expires < int(time.time()):
        return False
    expected = hmac.new(_push_secret, f"{expires}:{endpoint}".encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(supplied, expected)


def _delivery_ack_token(delivery_id, endpoint_fingerprint):
    message = f"push-ack:{delivery_id}:{endpoint_fingerprint}".encode()
    return hmac.new(_push_secret, message, hashlib.sha256).hexdigest()


def _declarative_push_payload(payload, delivery_id, ack_token):
    navigate = str(payload.get("url") or "/")
    if navigate.startswith("/"):
        navigate = f"{PUBLIC_BASE_URL}{navigate}"
    return {
        "web_push": 8030,
        "notification": {
            "title": str(payload.get("title") or "BTC Predictor"),
            "body": str(payload.get("body") or "Prediction update"),
            "navigate": navigate,
            "lang": "en-US",
            "dir": "ltr",
            "silent": False,
        },
        "event_id": payload.get("event_id"),
        "delivery_id": delivery_id,
        "ack_token": ack_token,
    }


def _push_error_status(exc):
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    text = str(exc)
    for candidate in (404, 410, 429, 500, 502, 503, 504):
        if str(candidate) in text:
            return candidate
    return None


def _delivery_counts(events, batch_id):
    batch = [event for event in events if event.get("batch_id") == batch_id]
    return {
        "attempted": len(batch),
        "accepted": sum(bool(event.get("accepted_at")) for event in batch),
        "failed": sum(bool(event.get("failed_at")) for event in batch),
        "received": sum(bool(event.get("received_at")) for event in batch),
        "notification_created": sum(bool(event.get("notification_created_at")) for event in batch),
        "retries": sum(int(event.get("retry_count") or 0) for event in batch),
    }


def _latest_delivery_summary(delivery_type):
    with push_delivery_lock:
        events = push_delivery_events_store.read([])
    if not isinstance(events, list):
        return {}
    matching = [
        event for event in events if event.get("delivery_type") == delivery_type
    ]
    if not matching:
        return {}
    latest = max(matching, key=lambda event: str(event.get("created_at") or ""))
    batch_id = latest.get("batch_id")
    batch = [event for event in matching if event.get("batch_id") == batch_id]
    counts = _delivery_counts(events, batch_id)
    errors = [event.get("error") for event in batch if event.get("error")]
    return {
        "batch_id": batch_id,
        "delivery_type": delivery_type,
        "attempted_at": min(
            (event.get("created_at") for event in batch if event.get("created_at")),
            default=None,
        ),
        **counts,
        "sent": counts["accepted"],
        "subscriptions": counts["attempted"],
        "error": errors[-1] if errors else None,
    }


def _record_push_decision(decision_id, **updates):
    now = _utcnow().isoformat()
    with push_decision_lock:
        events = push_decision_events_store.read([])
        if not isinstance(events, list):
            events = []
        event = next(
            (item for item in events if item.get("decision_id") == decision_id),
            None,
        )
        if event is None:
            event = {"decision_id": decision_id, "observed_at": now, "attempts": 0}
            events.append(event)
        event.update(updates)
        event["updated_at"] = now
        push_decision_events_store.write(events[-200:])
        return dict(event)


def _latest_push_decision():
    with push_decision_lock:
        events = push_decision_events_store.read([])
    if not isinstance(events, list) or not events:
        return {}
    return dict(max(events, key=lambda item: str(item.get("updated_at") or "")))


def _state_push_cooldown(now, last_state_push_at):
    """Return whether a generic state push is eligible and its next send time."""
    now = pd.Timestamp(now)
    if last_state_push_at is None or PUSH_STATE_COOLDOWN_SECONDS == 0:
        return True, now
    last = pd.Timestamp(last_state_push_at)
    eligible_at = last + pd.Timedelta(seconds=PUSH_STATE_COOLDOWN_SECONDS)
    return now >= eligible_at, eligible_at


def _signal_queue_state():
    state = signal_event_queue_store.read({})
    return {
        "pending": list(state.get("pending") or []),
        "notified_ids": list(state.get("notified_ids") or []),
        "last_generic_push_at": state.get("last_generic_push_at"),
    }


def _write_signal_queue(state):
    signal_event_queue_store.write({
        "pending": list(state.get("pending") or [])[-100:],
        "notified_ids": list(state.get("notified_ids") or [])[-500:],
        "last_generic_push_at": state.get("last_generic_push_at"),
    })


def _enqueue_signal_events(events):
    """Persist lifecycle events before any Web Push network call."""
    state = _signal_queue_state()
    incoming = [dict(event) for event in (events or [])]
    superseded = {}
    for event in incoming:
        event_type = event.get("event_type")
        signal_id = event.get("signal_id")
        if event_type in ("setup_invalidated", "setup_expired") and signal_id:
            superseded[signal_id] = event_type
        replaced = event.get("replaced_signal_id")
        if event_type == "setup_confirmed" and replaced:
            superseded[replaced] = "replaced_by_confirmed_setup"
    if superseded:
        retained = []
        for pending in state["pending"]:
            reason = superseded.get(pending.get("signal_id"))
            if reason:
                _record_push_decision(
                    pending.get("event_id"),
                    status="superseded",
                    superseded_at=_utcnow().isoformat(),
                    superseded_reason=reason,
                )
            else:
                retained.append(pending)
        state["pending"] = retained
    known = set(state["notified_ids"])
    known.update(item.get("event_id") for item in state["pending"])
    for event in incoming:
        event_id = event.get("event_id")
        if event_id and event_id not in known:
            state["pending"].append(dict(event))
            known.add(event_id)
            _record_push_decision(
                event_id,
                notification_kind=event.get("event_type"),
                status="detected",
                signal_id=event.get("signal_id"),
                summary=event.get("body"),
            )
    _write_signal_queue(state)
    return len(state["pending"])


def _cancel_signal_events(signal_id, reason):
    """Remove obsolete generic events when a definitive trade exit wins priority."""
    if not signal_id:
        return 0
    state = _signal_queue_state()
    remaining = []
    cancelled = 0
    for event in state["pending"]:
        if event.get("signal_id") == signal_id:
            cancelled += 1
            _record_push_decision(
                event.get("event_id"),
                status="superseded",
                superseded_at=_utcnow().isoformat(),
                superseded_reason=reason,
            )
        else:
            remaining.append(event)
    state["pending"] = remaining
    _write_signal_queue(state)
    return cancelled


def _dispatch_signal_event(now):
    """Submit at most one durable lifecycle event; retain it until accepted."""
    state = _signal_queue_state()
    if not state["pending"]:
        return 0
    event = state["pending"][0]
    event_id = event.get("event_id")
    eligible, eligible_at = _state_push_cooldown(
        now, state.get("last_generic_push_at")
    )
    if not eligible and not event.get("cooldown_exempt"):
        _record_push_decision(
            event_id,
            notification_kind=event.get("event_type"),
            status="deferred",
            eligible_at=eligible_at.isoformat(),
        )
        return 0

    decision = _record_push_decision(
        event_id,
        notification_kind=event.get("event_type"),
        status="submitting",
        signal_id=event.get("signal_id"),
        summary=event.get("body"),
    )
    accepted, failed = _send_push({
        "title": event.get("title") or "BTC Predictor update",
        "body": event.get("body") or "Signal lifecycle changed",
        "url": "/",
        "event_id": event_id,
    }, delivery_type="automatic")
    delivery = _latest_delivery_summary("automatic")
    status = (
        "accepted"
        if accepted > 0
        else "failed"
        if failed > 0
        else "no_enabled_subscription"
    )
    _record_push_decision(
        event_id,
        notification_kind=event.get("event_type"),
        status=status,
        attempts=int(decision.get("attempts") or 0) + 1,
        attempted=int(delivery.get("attempted") or 0),
        accepted=int(delivery.get("accepted") or 0),
        failed=int(delivery.get("failed") or 0),
        delivery_batch_id=delivery.get("batch_id"),
    )
    if accepted > 0:
        state["pending"].pop(0)
        state["notified_ids"].append(event_id)
        state["last_generic_push_at"] = pd.Timestamp(now).isoformat()
        _write_signal_queue(state)
    return accepted


def _signal_lifecycle_status():
    lifecycle = signal_lifecycle_store.read(SignalLifecycle.initial_state())
    queue = _signal_queue_state()
    active = lifecycle.get("active") or {}
    candidate = lifecycle.get("candidate") or {}
    return {
        "active_signal_id": active.get("signal_id"),
        "candidate_signal_id": candidate.get("signal_id"),
        "candidate_observations": int(candidate.get("observations") or 0),
        "pending_events": len(queue["pending"]),
        "last_generic_push_at": queue.get("last_generic_push_at"),
        "confirmation_observations": SIGNAL_CONFIRM_OBSERVATIONS,
        "invalidation_observations": SIGNAL_INVALIDATION_OBSERVATIONS,
        "bias_confirmation_observations": BIAS_CONFIRM_OBSERVATIONS,
    }


def _paper_exit_event_id(trade):
    identity = {
        key: trade.get(key)
        for key in (
            "entry_time",
            "exit_time",
            "side",
            "entry",
            "exit",
            "size",
            "exit_reason",
        )
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"paper-exit-{digest}"


def _paper_exit_payload(trade, event_id):
    reason = str(trade.get("exit_reason") or "").lower()
    outcome = "Target hit" if reason == "target" else "Stop hit"
    side = str(trade.get("side") or "trade").capitalize()
    exit_price = float(trade.get("exit") or 0.0)
    pnl = float(trade.get("pnl") or 0.0)
    r_multiple = float(trade.get("r_multiple") or 0.0)
    pnl_text = f"+${abs(pnl):,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
    return {
        "title": f"BTC paper trade · {outcome}",
        "body": (
            f"{side} · Exit ${exit_price:,.2f} · "
            f"P&L {pnl_text} · {r_multiple:+.2f}R"
        ),
        "url": "/",
        "event_id": event_id,
    }


def _notify_paper_exits(newly_closed):
    """Queue and immediately submit TP/SL notifications, with durable dedupe."""
    state = paper_exit_push_store.read({})
    pending = list(state.get("pending") or [])
    notified_ids = list(state.get("notified_ids") or [])
    known_ids = set(notified_ids)
    known_ids.update(item.get("event_id") for item in pending)
    for trade in newly_closed or []:
        if str(trade.get("exit_reason") or "").lower() not in ("target", "stop"):
            continue
        event_id = _paper_exit_event_id(trade)
        if event_id not in known_ids:
            pending.append({"event_id": event_id, "trade": dict(trade)})
            known_ids.add(event_id)
    # Persist before network I/O so a process restart cannot lose an exit.
    state = {"pending": pending[-100:], "notified_ids": notified_ids[-500:]}
    paper_exit_push_store.write(state)

    accepted_total = 0
    remaining = []
    for item in state["pending"]:
        event_id = item.get("event_id")
        trade = item.get("trade") or {}
        decision = _record_push_decision(
            event_id,
            notification_kind="paper_exit",
            status="detected",
            summary=(
                "Target hit"
                if str(trade.get("exit_reason") or "").lower() == "target"
                else "Stop hit"
            ),
        )
        accepted, _failed = _send_push(
            _paper_exit_payload(trade, event_id),
            delivery_type="automatic",
        )
        delivery = _latest_delivery_summary("automatic")
        if accepted > 0:
            accepted_total += accepted
            notified_ids.append(event_id)
            decision_status = "accepted"
        else:
            remaining.append(item)
            decision_status = "failed" if _failed else "no_enabled_subscription"
        _record_push_decision(
            event_id,
            notification_kind="paper_exit",
            status=decision_status,
            attempts=int(decision.get("attempts") or 0) + 1,
            attempted=int(delivery.get("attempted") or 0),
            accepted=int(delivery.get("accepted") or 0),
            failed=int(delivery.get("failed") or 0),
            delivery_batch_id=delivery.get("batch_id"),
        )
    paper_exit_push_store.write({
        "pending": remaining[-100:],
        "notified_ids": notified_ids[-500:],
    })
    return accepted_total


def _update_last_delivery_from_events(events, batch_id):
    summary = push_delivery_store.read({})
    if summary.get("batch_id") != batch_id:
        return
    counts = _delivery_counts(events, batch_id)
    summary.update(counts)
    summary["sent"] = counts["accepted"]  # Backward-compatible API field.
    push_delivery_store.write(summary)


def _send_push(payload, subscriptions=None, delivery_type="automatic"):
    attempted_at = _utcnow()
    batch_id = uuid.uuid4().hex
    if webpush is None:
        push_delivery_store.write({
            "batch_id": batch_id,
            "delivery_type": delivery_type,
            "attempted_at": attempted_at.isoformat(),
            "attempted": 0,
            "accepted": 0,
            "sent": 0,
            "failed": 0,
            "error": "pywebpush unavailable",
        })
        return 0, 0
    with push_lock:
        targets = [
            item
            for item in (subscriptions if subscriptions is not None else push_subscriptions)
            if item.get("enabled") is not False
        ]
    accepted, failed, stale = 0, 0, []
    last_error = None
    ttl = 900 if delivery_type == "automatic" else 120
    prepared = []
    for sub in targets:
        endpoint = str(sub.get("endpoint") or "")
        endpoint_fingerprint = _endpoint_hash(endpoint)
        delivery_id = uuid.uuid4().hex
        ack_token = _delivery_ack_token(delivery_id, endpoint_fingerprint)
        delivery_payload = _declarative_push_payload(payload, delivery_id, ack_token)
        event = {
            "delivery_id": delivery_id,
            "batch_id": batch_id,
            "delivery_type": delivery_type,
            "endpoint_hash": endpoint_fingerprint,
            "installation_id": sub.get("installation_id"),
            "payload": payload,
            "created_at": attempted_at.isoformat(),
            "accepted_at": None,
            "failed_at": None,
            "received_at": None,
            "notification_created_at": None,
            "retry_count": 0,
            "next_retry_at": (
                attempted_at + pd.Timedelta(seconds=PUSH_ACK_RETRY_SECONDS)
            ).isoformat(),
            "expires_at": (attempted_at + pd.Timedelta(seconds=ttl)).isoformat(),
            "error": None,
            "http_status": None,
        }
        prepared.append((sub, endpoint, delivery_payload, event))
    # Persist every delivery ID before sending. A fast device can acknowledge
    # immediately after the push service accepts the request.
    with push_delivery_lock:
        delivery_events = push_delivery_events_store.read([])
        if not isinstance(delivery_events, list):
            delivery_events = []
        delivery_events.extend(item[3] for item in prepared)
        push_delivery_events_store.write(delivery_events[-PUSH_EVENT_RETENTION:])
    for sub, endpoint, delivery_payload, event in prepared:
        try:
            webpush(
                subscription_info=_subscription_info(sub),
                data=json.dumps(delivery_payload),
                vapid_private_key=str(vapid_path),
                vapid_claims={"sub":_vapid_subject},
                timeout=10,
                ttl=ttl,
                headers={"Urgency":"high"},
            )
            accepted += 1
            event["accepted_at"] = _utcnow().isoformat()
        except Exception as exc:
            logger.warning("Web Push delivery failed: %s", exc)
            failed += 1
            last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            status = _push_error_status(exc)
            event["failed_at"] = _utcnow().isoformat()
            event["error"] = last_error
            event["http_status"] = status
            if status in (404, 410):
                stale.append(endpoint)
        with push_delivery_lock:
            latest_events = push_delivery_events_store.read([])
            for stored_event in latest_events:
                if stored_event.get("delivery_id") == event["delivery_id"]:
                    stored_event.update({
                        "accepted_at": event.get("accepted_at"),
                        "failed_at": event.get("failed_at"),
                        "error": event.get("error"),
                        "http_status": event.get("http_status"),
                    })
                    break
            push_delivery_events_store.write(latest_events[-PUSH_EVENT_RETENTION:])
    if stale:
        with push_lock:
            push_subscriptions[:] = [s for s in push_subscriptions if s.get("endpoint") not in stale]
            _persist_subscriptions()
    with push_delivery_lock:
        delivery_events = push_delivery_events_store.read([])
    counts = _delivery_counts(delivery_events, batch_id)
    push_delivery_store.write({
        "batch_id": batch_id,
        "delivery_type": delivery_type,
        "attempted_at": attempted_at.isoformat(),
        **counts,
        "sent": counts["accepted"],
        "error": last_error,
        "subscriptions": len(push_subscriptions),
    })
    logger.info(
        "Web Push %s submission: attempted=%s accepted=%s failed=%s",
        delivery_type,
        len(targets),
        accepted,
        failed,
    )
    return accepted, failed


def _retry_unacknowledged_pushes(now=None):
    if webpush is None or PUSH_MAX_ACK_RETRIES == 0:
        return 0
    now = now or _utcnow()
    retried = 0
    stale = []
    with push_lock:
        with push_delivery_lock:
            events = push_delivery_events_store.read([])
        if not isinstance(events, list):
            return 0
        subscriptions_by_hash = {
            _endpoint_hash(item.get("endpoint")): item for item in push_subscriptions
        }
        changed_batches = set()
        for event in events:
            if (
                event.get("delivery_type") != "automatic"
                or event.get("accepted_at")
                or event.get("received_at")
            ):
                continue
            try:
                retry_at = pd.Timestamp(event["next_retry_at"])
                expires_at = pd.Timestamp(event["expires_at"])
            except (KeyError, TypeError, ValueError):
                continue
            subscription = subscriptions_by_hash.get(event.get("endpoint_hash"))
            if now >= expires_at:
                continue
            if (
                now < retry_at
                or int(event.get("retry_count") or 0) >= PUSH_MAX_ACK_RETRIES
            ):
                continue
            if not subscription:
                continue
            payload = _declarative_push_payload(
                event.get("payload") or {},
                event["delivery_id"],
                _delivery_ack_token(event["delivery_id"], event["endpoint_hash"]),
            )
            remaining_ttl = max(1, int((expires_at - now).total_seconds()))
            try:
                webpush(
                    subscription_info=_subscription_info(subscription),
                    data=json.dumps(payload),
                    vapid_private_key=str(vapid_path),
                    vapid_claims={"sub": _vapid_subject},
                    timeout=10,
                    ttl=remaining_ttl,
                    headers={"Urgency": "high"},
                )
                event["retry_count"] = int(event.get("retry_count") or 0) + 1
                event["last_retry_at"] = now.isoformat()
                event["accepted_at"] = event.get("accepted_at") or now.isoformat()
                event["failed_at"] = None
                event["error"] = None
                event["http_status"] = None
                event["next_retry_at"] = (
                    now
                    + pd.Timedelta(
                        seconds=PUSH_ACK_RETRY_SECONDS * (2 ** event["retry_count"])
                    )
                ).isoformat()
                retried += 1
            except Exception as exc:
                status = _push_error_status(exc)
                event["retry_count"] = int(event.get("retry_count") or 0) + 1
                event["last_retry_at"] = now.isoformat()
                event["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                event["http_status"] = status
                if status in (404, 410):
                    event["failed_at"] = now.isoformat()
                    stale.append(subscription.get("endpoint"))
            changed_batches.add(event.get("batch_id"))
        if stale:
            push_subscriptions[:] = [
                item for item in push_subscriptions if item.get("endpoint") not in stale
            ]
            _persist_subscriptions()
        if retried or stale:
            with push_delivery_lock:
                push_delivery_events_store.write(events[-PUSH_EVENT_RETENTION:])
                for batch_id in changed_batches:
                    if batch_id:
                        _update_last_delivery_from_events(events, batch_id)
    return retried


def _subscription_counts():
    with push_lock:
        stored = len(push_subscriptions)
        enabled = sum(item.get("enabled") is not False for item in push_subscriptions)
        verified = sum(bool(item.get("last_ack_at")) for item in push_subscriptions)
    return {"stored": stored, "enabled": enabled, "verified": verified}


def _admin_request_authorized():
    configured = os.getenv("PUSH_ADMIN_TOKEN") or os.getenv("ADMIN_API_TOKEN")
    supplied = request.headers.get("Authorization", "").removeprefix("Bearer ")
    return bool(configured and supplied and hmac.compare_digest(supplied, configured))


def _ack_delivery(delivery_id, token, status):
    if status not in ("received", "notification_created"):
        return False, "invalid acknowledgement status"
    now = _utcnow()
    with push_lock:
        with push_delivery_lock:
            events = push_delivery_events_store.read([])
        if not isinstance(events, list):
            return False, "delivery not found"
        event = next(
            (item for item in events if item.get("delivery_id") == delivery_id),
            None,
        )
        if not event:
            return False, "delivery not found"
        expected = _delivery_ack_token(delivery_id, event.get("endpoint_hash"))
        if not token or not hmac.compare_digest(str(token), expected):
            return False, "unauthorized"
        field = "received_at" if status == "received" else "notification_created_at"
        event[field] = event.get(field) or now.isoformat()
        if status == "notification_created":
            event["received_at"] = event.get("received_at") or now.isoformat()
        for subscription in push_subscriptions:
            if _endpoint_hash(subscription.get("endpoint")) == event.get("endpoint_hash"):
                subscription["last_ack_at"] = now.isoformat()
                subscription["ack_miss_count"] = 0
                subscription["enabled"] = True
                subscription["status"] = "verified"
                break
        _persist_subscriptions()
        with push_delivery_lock:
            push_delivery_events_store.write(events[-PUSH_EVENT_RETENTION:])
            _update_last_delivery_from_events(events, event.get("batch_id"))
    return True, None


def _live_loop():
    global live_state
    binance_rest_retry_at = pd.Timestamp(0, tz="UTC")
    flow_bars_cache = None
    binance_rest_last_ok_at = None
    binance_rest_last_error = None
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
            collectors = trade_store.collector_status(poll_started, COLLECTOR_STALE_SECONDS)
            binance_ws_last = (collectors.get("binance", {}) or {}).get("last_message_at")
            binance_ws_fresh = False
            if binance_ws_last:
                binance_ws_ts = pd.Timestamp(binance_ws_last)
                if binance_ws_ts.tzinfo is None:
                    binance_ws_ts = binance_ws_ts.tz_localize("UTC")
                binance_ws_fresh = (poll_started - binance_ws_ts).total_seconds() <= COLLECTOR_STALE_SECONDS
            binance_stale = binance_lag is None or binance_lag > COLLECTOR_STALE_SECONDS
            # Prefer WebSocket collectors. REST is rare:
            # - always-on only when BINANCE_REST_ENABLED=1
            # - or emergency stale recovery when BINANCE_REST_ON_STALE=1 and WS is stale
            # Backfill whenever Binance data is older than one poll interval while the
            # WS stream is dead, so lag never reaches the stale threshold. Each cycle
            # costs ~22 weight (aggTrades 20 + klines 2) ~= 32 weight/min, ~1.3% of the
            # 2,400/min fapi budget.
            poll_seconds = max(20, int(os.getenv("LIVE_POLL_SECONDS", "45")))
            binance_backfill_due = binance_lag is None or binance_lag > poll_seconds
            rest_due = poll_started >= binance_rest_retry_at
            rest_allowed = BINANCE_REST_ENABLED or (BINANCE_REST_ON_STALE and binance_backfill_due and not binance_ws_fresh)
            binance_data_path = "websocket" if binance_ws_fresh else ("rest_backfill" if BINANCE_REST_ON_STALE else "stale")
            if rest_allowed and rest_due:
                try:
                    inserted = trade_store.append(_binance_trades())
                    binance_rest_last_ok_at = poll_started
                    binance_rest_last_error = None
                    # Success path: the always-on cadence applies only to explicit
                    # BINANCE_REST_ENABLED. Stale recovery runs every poll while data
                    # is old (~32 weight/min); error cooldowns below are the guard.
                    cooldown_min = BINANCE_REST_MINUTES if BINANCE_REST_ENABLED else 0
                    binance_rest_retry_at = poll_started + pd.Timedelta(minutes=cooldown_min)
                    if inserted:
                        logger.info("Binance REST backfill inserted %s trades (stale=%s)", inserted, binance_stale)
                    # Also refresh the flow baseline while we are paying REST weight anyway
                    # (klines cost weight ~2 for limit<=500; still a tiny share of 2400/min).
                    try:
                        flow_bars_cache = _binance_flow_bars()
                    except Exception as exc:
                        logger.warning("Binance REST flow baseline unavailable: %s", exc)
                except Exception as exc:
                    base_min = BINANCE_REST_MINUTES if BINANCE_REST_ENABLED else BINANCE_STALE_REST_MINUTES
                    # Hard back off on explicit rate-limit/ban signals; short retry otherwise.
                    cooldown_min = 30 if ("418" in str(exc) or "429" in str(exc)) else max(base_min * 4, 5)
                    binance_rest_retry_at = poll_started + pd.Timedelta(minutes=cooldown_min)
                    binance_rest_last_error = f"{type(exc).__name__}: {str(exc)[:160]} (cooldown {cooldown_min}m)"
                    logger.warning("Binance REST trades unavailable (cooldown %sm): %s", cooldown_min, exc)
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
            elif (BINANCE_REST_ENABLED or (BINANCE_REST_ON_STALE and binance_stale)) and flow_bars_cache is not None and not flow_bars_cache.empty:
                if flow_bars_cache is not None and not flow_bars_cache.empty and flow_bars_cache.index[-1] >= now - pd.Timedelta(minutes=2):
                    flow_bars = flow_bars_cache.loc[flow_bars_cache.index <= now]
                    flow_source = "rest_backfill"
                else:
                    flow_source = None
            else:
                flow_source = None
            result = predictor.predict(ohlc, trades, 100_000, frames=frames, flow_bars=flow_bars)
            paper_status = paper_ledger.update(result, ohlc)
            notification_now = pd.Timestamp.now(tz="UTC")
            lifecycle_before = signal_lifecycle_store.read(
                SignalLifecycle.initial_state()
            )
            active_before = lifecycle_before.get("active") or {}
            definitive_exit = any(
                str(trade.get("exit_reason") or "").lower() in ("target", "stop")
                for trade in (paper_status.get("newly_closed") or [])
            )
            try:
                _notify_paper_exits(paper_status.get("newly_closed") or [])
            except Exception:
                logger.exception("Paper exit notification dispatch failed")
            if definitive_exit:
                _cancel_signal_events(
                    active_before.get("signal_id"), "paper_exit"
                )
            lifecycle_state, lifecycle_events = signal_lifecycle.evaluate(
                lifecycle_before, result, paper_status, notification_now
            )
            signal_lifecycle_store.write(lifecycle_state)
            _enqueue_signal_events(lifecycle_events)
            # TP/SL is always the only alert dispatched in its poll. Lifecycle
            # alerts resume next poll and retain their durable queue position.
            if not definitive_exit:
                _dispatch_signal_event(notification_now)
            trade_store.prune(now-pd.Timedelta(minutes=int(os.getenv("TRADE_RETENTION_MINUTES","120"))))
            now = notification_now
            try:
                _retry_unacknowledged_pushes(now)
            except Exception as exc:
                logger.warning("Web Push acknowledgement retry failed: %s", exc)
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
                "binance_data_path": binance_data_path,
                "binance_rest_last_ok_at": binance_rest_last_ok_at.isoformat() if binance_rest_last_ok_at else None,
                "binance_rest_last_error": binance_rest_last_error,
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
    with live_start_lock:
        if live_thread_started and live_thread is not None and live_thread.is_alive(): return True
        lock_handle = None
        try:
            lock_handle = open(data_dir / "live-loop.lock", "w")
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            if lock_handle is not None:
                lock_handle.close()
            return False
        try:
            _live_lock_handle = lock_handle
            collector_thread = start_collectors(trade_store)
            live_thread = threading.Thread(target=_live_loop, name="live-predictor", daemon=True)
            live_thread.start()
            live_thread_started = True
            logger.info("Live predictor loop started independently of dashboard traffic")
            return True
        except Exception:
            logger.exception("Unable to start live predictor loop")
            live_thread_started = False
            if lock_handle is not None:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
                lock_handle.close()
            _live_lock_handle = None
            return False


def _supervise_live_loop():
    retry_seconds = max(1, int(os.getenv("LIVE_BOOT_RETRY_SECONDS", "5")))
    while not start_live_loop():
        logger.info("Live predictor lock is held by a retiring worker; retrying in %ss", retry_seconds)
        time.sleep(retry_seconds)


def start_live_boot_supervisor():
    global live_boot_thread
    if live_boot_thread is not None and live_boot_thread.is_alive():
        return live_boot_thread
    live_boot_thread = threading.Thread(target=_supervise_live_loop, name="live-boot-supervisor", daemon=True)
    live_boot_thread.start()
    return live_boot_thread


if os.getenv("START_LIVE_LOOP_ON_BOOT", "0").lower() in ("1", "true", "yes", "on"):
    start_live_boot_supervisor()


@app.get("/")
@app.get("/dashboard")
def index(): return render_template("dashboard.html")


def _generate_png_icon(width=180, height=180):
    import struct, zlib, math
    r1, g1, b1 = (247, 147, 26)  # Bitcoin Orange #F7931A
    r2, g2, b2 = (255, 255, 255)  # White B
    pixels = []
    scale = width / 180.0
    cx, cy = width / 2.0, height / 2.0
    angle = math.radians(-14)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for y in range(height):
        row = [0]
        for x in range(width):
            dx = (x - cx) / scale
            dy = (y - cy) / scale
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            d_spine = max(abs(rx + 14.0) - 8.0, abs(ry) - 34.0)
            d_bar1 = max(abs(rx + 10.0) - 3.5, abs(ry + 40.0) - 6.0)
            d_bar2 = max(abs(rx - 2.0) - 3.5, abs(ry + 40.0) - 6.0)
            d_bar3 = max(abs(rx + 10.0) - 3.5, abs(ry - 40.0) - 6.0)
            d_bar4 = max(abs(rx - 2.0) - 3.5, abs(ry - 40.0) - 6.0)
            d_hbar_top = max(abs(rx + 5.0) - 17.0, abs(ry + 28.0) - 6.0)
            d_hbar_mid = max(abs(rx + 6.0) - 18.0, abs(ry) - 6.0)
            d_hbar_bot = max(abs(rx + 5.0) - 17.0, abs(ry - 28.0) - 6.0)
            dist_top_c = math.sqrt((rx + 6.0)**2 + (ry + 17.0)**2)
            d_top_lobe = dist_top_c - 19.0
            if rx < -6.0 or ry > -3.0 or ry < -34.0:
                d_top_lobe = 999.0
            dist_bot_c = math.sqrt((rx + 6.0)**2 + (ry - 17.0)**2)
            d_bot_lobe = dist_bot_c - 22.0
            if rx < -6.0 or ry < 3.0 or ry > 36.0:
                d_bot_lobe = 999.0
            d_solid = min(d_spine, d_bar1, d_bar2, d_bar3, d_bar4, d_hbar_top, d_hbar_mid, d_hbar_bot, d_top_lobe, d_bot_lobe)
            d_top_hole = 8.5 - dist_top_c
            d_bot_hole = 10.0 - dist_bot_c
            if d_top_hole > 0 and rx > -6.0 and ry < -3.0:
                d_solid = max(d_solid, d_top_hole)
            if d_bot_hole > 0 and rx > -6.0 and ry > 3.0:
                d_solid = max(d_solid, d_bot_hole)
            alpha = max(0.0, min(1.0, 0.5 - d_solid / 1.5))
            r = int(r1 * (1.0 - alpha) + r2 * alpha)
            g = int(g1 * (1.0 - alpha) + g2 * alpha)
            b = int(b1 * (1.0 - alpha) + b2 * alpha)
            row.extend([r, g, b])
        pixels.append(bytes(row))
    raw_data = b"".join(pixels)
    compressed = zlib.compress(raw_data, 9)
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", compressed)
    iend = chunk(b"IEND", b"")
    return header + ihdr + idat + iend

_cached_png_512 = _generate_png_icon(512, 512)
_cached_png_192 = _generate_png_icon(192, 192)
_cached_png_180 = _generate_png_icon(180, 180)
_cached_png_32 = _generate_png_icon(32, 32)

@app.get("/manifest.json")
@app.get("/site.webmanifest")
def web_manifest():
    return {
        "id": "/",
        "name": "BTC Structure Flow",
        "short_name": "BTC Flow",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0b0e14",
        "theme_color": "#f7931a",
        "icons": [
            {
                "src": "/apple-touch-icon.png",
                "sizes": "180x180",
                "type": "image/png",
                "purpose": "any maskable"
            },
            {
                "src": "/icon-192.png",
                "sizes": "192x192",
                "type": "image/png",
                "purpose": "any maskable"
            },
            {
                "src": "/icon-512.png",
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "any maskable"
            },
            {
                "src": "/favicon.ico",
                "sizes": "32x32",
                "type": "image/png",
                "purpose": "any"
            }
        ]
    }, 200, {"Content-Type": "application/manifest+json", "Cache-Control": "public, max-age=86400"}

@app.get("/icon-192.png")
def icon_192():
    return _cached_png_192, 200, {"Content-Type": "image/png", "Cache-Control": "public, max-age=86400"}

@app.get("/icon-512.png")
def icon_512():
    return _cached_png_512, 200, {"Content-Type": "image/png", "Cache-Control": "public, max-age=86400"}

@app.get("/favicon.ico")
def favicon_ico():
    return _cached_png_32, 200, {"Content-Type": "image/png", "Cache-Control": "public, max-age=86400"}

@app.get("/apple-touch-icon.png")
@app.get("/apple-touch-icon-precomposed.png")
def apple_touch_icon():
    return _cached_png_180, 200, {"Content-Type": "image/png", "Cache-Control": "public, max-age=86400"}

@app.get("/icon.svg")
def icon_svg():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <rect width="512" height="512" rx="120" fill="#F7931A"/>
  <path fill="#FFFFFF" d="M361.3 227.1c4.8-32.3-19.8-49.7-53.5-61.3l10.9-43.8-26.6-6.6-10.6 42.6c-7-.1-14.2-.1-21.3.1l10.7-42.9-26.6-6.6-10.9 43.8c-5.8-1.3-11.6-2.6-17.3-3.9l.1-.4-36.7-9.2-7.1 28.4s19.8 4.5 19.3 4.8c10.8 2.7 12.8 9.8 12.4 15.5l-12.5 50c.7.2 1.7.5 2.8.9-1 .3-2 .5-2.9.3l-17.5 70.1c-1.3 3.3-4.7 8.3-12.3 6.4.3.4-19.3-4.8-19.3-4.8l-13.2 30.5 34.6 8.7c6.4 1.6 12.7 3.3 18.9 4.9l-11 44.3 26.6 6.6 10.9-43.8c7.2 2 14.2 3.8 21.2 5.5l-10.8 43.4 26.6 6.6 11-44.1c45.4 8.6 79.5 5.1 93.9-35.9 11.6-33.1-.6-52.2-24.6-64.6 17.5-4 30.6-15.5 34.1-39.2zm-61 85.7c-8.2 33.1-64.1 15.2-82.2 10.7l14.7-58.8c18.1 4.5 76 13.5 67.5 48.1zm8.3-86.2c-7.5 30.1-54 14.8-69.1 11.1l13.3-53.4c15.1 3.8 63.4 10.8 55.8 42.3z"/>
</svg>"""
    return svg, 200, {"Content-Type": "image/svg+xml", "Cache-Control": "public, max-age=86400"}




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
    last_automatic_delivery = _latest_delivery_summary("automatic")
    last_test_delivery = _latest_delivery_summary("test")
    last_notification_decision = _latest_push_decision()
    push_counts = _subscription_counts()
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
        "push":{
            "supported":webpush is not None,
            "single_installation":PUSH_SINGLE_INSTALLATION,
            "state_change_cooldown_seconds":PUSH_STATE_COOLDOWN_SECONDS,
            "signal_lifecycle":_signal_lifecycle_status(),
            "subscriptions":push_counts["stored"],
            "stored_subscriptions":push_counts["stored"],
            "enabled_subscriptions":push_counts["enabled"],
            "verified_subscriptions":push_counts["verified"],
            # `last_delivery` remains for older clients, but it now means the
            # latest automatic signal push. Manual tests have their own field.
            "last_delivery":last_automatic_delivery,
            "last_automatic_delivery":last_automatic_delivery,
            "last_test_delivery":last_test_delivery,
            "last_notification_decision":last_notification_decision,
        },
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
    last_automatic_delivery = _latest_delivery_summary("automatic")
    last_test_delivery = _latest_delivery_summary("test")
    last_notification_decision = _latest_push_decision()
    push_counts = _subscription_counts()
    return jsonify({
        "paper_only": True,
        **state,
        "push": {
            "supported": webpush is not None,
            "single_installation": PUSH_SINGLE_INSTALLATION,
            "state_change_cooldown_seconds": PUSH_STATE_COOLDOWN_SECONDS,
            "signal_lifecycle": _signal_lifecycle_status(),
            "subscriptions": push_counts["stored"],
            "stored_subscriptions": push_counts["stored"],
            "enabled_subscriptions": push_counts["enabled"],
            "verified_subscriptions": push_counts["verified"],
            "last_delivery": last_automatic_delivery,
            "last_automatic_delivery": last_automatic_delivery,
            "last_test_delivery": last_test_delivery,
            "last_notification_decision": last_notification_decision,
        },
    })


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
    public_key = json.dumps(_vapid_public_key)
    script = f"""
const VAPID_PUBLIC_KEY = {public_key};
const decodeKey = value => {{
  const padding = '='.repeat((4 - value.length % 4) % 4);
  const raw = atob((value + padding).replace(/-/g, '+').replace(/_/g, '/'));
  return Uint8Array.from([...raw].map(char => char.charCodeAt(0)));
}};
self.addEventListener('install', event => event.waitUntil(self.skipWaiting()));
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));
self.addEventListener('push', event => {{
  let data = {{}};
  try {{
    data = event.data ? event.data.json() : {{}};
  }} catch (_) {{
    data = {{ notification: {{ title: 'BTC Predictor', body: 'New prediction update', navigate: '/' }} }};
  }}
  const proposed = data.notification || {{}};
  const title = proposed.title || data.title || 'BTC Predictor';
  const body = proposed.body || data.body || 'Prediction update';
  const navigate = proposed.navigate || data.url || '/';
  const iconUrl = new URL('/apple-touch-icon.png', self.location.origin).href;
  const options = {{
    body,
    icon: iconUrl,
    tag: data.event_id ? `btc-predictor-${{data.event_id}}` : `btc-predictor-${{Date.now()}}`,
    data: {{ url: navigate }},
  }};
  const acknowledge = status => {{
    if (!data.delivery_id || !data.ack_token) return Promise.resolve();
    return fetch('/push/ack', {{
      method: 'POST',
      headers: {{ 'content-type': 'application/json' }},
      body: JSON.stringify({{
        delivery_id: data.delivery_id,
        ack_token: data.ack_token,
        status,
      }}),
    }}).catch(() => undefined);
  }};
  event.waitUntil((async () => {{
    const receivedAcknowledgement = acknowledge('received');
    try {{
      await self.registration.showNotification(title, options);
    }} catch (_) {{
      await self.registration.showNotification(title, {{
        body,
        icon: iconUrl,
      }});
    }}
    await Promise.allSettled([
      receivedAcknowledgement,
      acknowledge('notification_created'),
    ]);
  }})());
}});
self.addEventListener('notificationclick', event => {{
  event.notification.close();
  const target = new URL((event.notification.data || {{}}).url || '/', self.location.origin).href;
  event.waitUntil(self.clients.matchAll({{type: 'window', includeUncontrolled: true}}).then(windows => {{
    const existing = windows.find(client => client.url.startsWith(self.location.origin));
    return existing ? existing.focus().then(() => existing.navigate(target)) : self.clients.openWindow(target);
  }}));
}});
self.addEventListener('pushsubscriptionchange', event => {{
  event.waitUntil(self.registration.pushManager.subscribe({{
    userVisibleOnly: true,
    applicationServerKey: decodeKey(VAPID_PUBLIC_KEY),
  }}).then(subscription => fetch('/push/subscribe', {{
    method: 'POST',
    headers: {{'content-type': 'application/json'}},
    body: JSON.stringify({{
      ...subscription.toJSON(),
      previous_endpoint: event.oldSubscription ? event.oldSubscription.endpoint : null,
      app_mode: 'standalone',
    }}),
  }})));
}});
"""
    return script, 200, {"Content-Type":"application/javascript","Service-Worker-Allowed":"/","Cache-Control":"no-store"}


@app.get("/push/config")
def push_config():
    return jsonify({
        "supported": webpush is not None,
        "vapid_public_key": _vapid_public_key,
    })


@app.post("/push/subscribe")
def push_subscribe():
    data = request.get_json(silent=True) or {}
    validation_error = _validate_subscription(data)
    if validation_error:
        return jsonify({"error": validation_error}), 400
    previous_endpoint = str(data.get("previous_endpoint") or "")
    with push_lock:
        endpoint = data["endpoint"]
        installation_id = data.get("installation_id")
        if PUSH_SINGLE_INSTALLATION:
            push_subscriptions[:] = [
                item for item in push_subscriptions if item.get("endpoint") == endpoint
            ]
        known = any(
            item.get("endpoint") == endpoint
            or (installation_id and item.get("installation_id") == installation_id)
            for item in push_subscriptions
        )
        if not known and len(push_subscriptions) >= PUSH_MAX_SUBSCRIPTIONS:
            return jsonify({"error": "subscription capacity reached"}), 409
        if (
            previous_endpoint
            and previous_endpoint != endpoint
            and _is_allowed_push_endpoint(previous_endpoint)
        ):
            _remove_subscription(push_subscriptions, previous_endpoint)
        created = _upsert_subscription(push_subscriptions, data)
        _persist_subscriptions()
        stored = len(push_subscriptions)
        current = next(
            (item for item in push_subscriptions if item.get("endpoint") == endpoint),
            {},
        )
    token = _issue_endpoint_token(data["endpoint"])
    return jsonify({
        "ok": True,
        "created": created,
        "stored_subscriptions": stored,
        "subscriptions": stored,
        "current_verified": bool(current.get("last_ack_at")),
        "current_status": current.get("status") or "unverified",
        "current_ack_miss_count": int(current.get("ack_miss_count") or 0),
        "single_installation": PUSH_SINGLE_INSTALLATION,
        "test_token": token,
        "test_token_expires_in": 900,
    })


@app.post("/push/unsubscribe")
def push_unsubscribe():
    data = request.get_json(silent=True) or {}
    endpoint = data.get("endpoint", "")
    token = data.get("test_token", "")
    if not endpoint or not _verify_endpoint_token(endpoint, token):
        return jsonify({"error":"unauthorized"}), 401
    with push_lock:
        removed = _remove_subscription(push_subscriptions, endpoint)
        if removed:
            _persist_subscriptions()
    return jsonify({"ok":True,"removed":removed,"subscriptions":len(push_subscriptions)})


@app.post("/push/test")
def push_test():
    data = request.get_json(silent=True) or {}; endpoint, token = data.get("endpoint", ""), data.get("test_token", "")
    if not endpoint or not _verify_endpoint_token(endpoint, token): return jsonify({"error":"unauthorized"}), 401
    if webpush is None: return jsonify({"error":"pywebpush unavailable"}), 503
    with push_lock: target = [s for s in push_subscriptions if s.get("endpoint") == endpoint]
    accepted, failed = _send_push(
        {"title":"BTC Predictor test","body":"Web Push is connected and delivering notifications.","url":"/","event_id":"test"},
        target,
        delivery_type="test",
    )
    return jsonify({"ok":accepted > 0,"accepted":accepted,"sent":accepted,"failed":failed})


@app.post("/push/ack")
def push_ack():
    data = request.get_json(silent=True) or {}
    ok, error = _ack_delivery(
        str(data.get("delivery_id") or ""),
        str(data.get("ack_token") or ""),
        str(data.get("status") or ""),
    )
    if ok:
        return jsonify({"ok": True})
    status = 401 if error == "unauthorized" else 400
    return jsonify({"ok": False, "error": error}), status


@app.post("/push/broadcast-test")
def push_broadcast_test():
    if not _admin_request_authorized():
        return jsonify({"error": "unauthorized"}), 401
    if webpush is None: return jsonify({"error":"pywebpush unavailable"}), 503
    accepted, failed = _send_push(
        {"title":"BTC Predictor test","body":"Live push notification test — verifying background delivery!","url":"/","event_id":f"broadcast-{int(time.time())}"},
        subscriptions=None,
        delivery_type="test",
    )
    return jsonify({"ok":accepted > 0,"accepted":accepted,"sent":accepted,"failed":failed})


@app.get("/push/admin/status")
def push_admin_status():
    if not _admin_request_authorized():
        return jsonify({"error": "unauthorized"}), 401
    with push_lock:
        subscriptions = [
            {
                "endpoint_hash": _endpoint_hash(item.get("endpoint")),
                "host": urlparse(item.get("endpoint") or "").hostname,
                "installation_id": item.get("installation_id"),
                "created_at": item.get("created_at"),
                "last_seen_at": item.get("last_seen_at"),
                "last_ack_at": item.get("last_ack_at"),
                "ack_miss_count": int(item.get("ack_miss_count") or 0),
                "enabled": item.get("enabled") is not False,
                "status": item.get("status") or "legacy",
                "app_mode": item.get("app_mode"),
                "platform": item.get("platform"),
            }
            for item in push_subscriptions
        ]
    return jsonify({
        "subscriptions": subscriptions,
        "last_delivery": _latest_delivery_summary("automatic"),
        "last_automatic_delivery": _latest_delivery_summary("automatic"),
        "last_test_delivery": _latest_delivery_summary("test"),
        "last_notification_decision": _latest_push_decision(),
    })


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
