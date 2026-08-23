from __future__ import annotations

import base64
import ctypes
import fcntl
import gc
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
from btc_predictor.binance_backfill import BinanceBackfillController
from btc_predictor.binance_rest import (
    BinanceRateLimited,
    BinanceRestConfig,
    BinanceRestDeferred,
    BinanceRestLimiter,
)
from btc_predictor.strategy import Predictor
from btc_predictor.trade_store import TradeStore, start_collectors
from btc_predictor.paper_position import PaperLedger
from btc_predictor.signal_lifecycle import SignalLifecycle
from btc_predictor.flow_gate import load_flow_gate
from btc_predictor.flow_state import FlowStateStore
from btc_predictor import live_policy

try:
    _libc = ctypes.CDLL("libc.so.6")
    _malloc_trim = _libc.malloc_trim
    _malloc_trim.argtypes = [ctypes.c_size_t]
    _malloc_trim.restype = ctypes.c_int
except (AttributeError, OSError):
    _malloc_trim = None

app = Flask(__name__)
logger = logging.getLogger("btc_predictor")
_log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger.setLevel(_log_level)
data_dir = runtime_dir()
COLLECTOR_STALE_SECONDS = max(30, int(os.getenv("COLLECTOR_STALE_SECONDS", "90")))
# Late-entry guard: deep sweeps enter on a retrace limit. Set empty to disable.
_retrace_atr = os.getenv("RETRACE_ENTRY_ATR", "1.2").strip()
flow_gate_config, flow_calibration_artifact = load_flow_gate(
    os.getenv("FLOW_CALIBRATION_PATH", str(data_dir / "flow_calibration.json")),
    requested_mode=os.getenv("FLOW_GATE_MODE", "independent"),
    overrides={
        "legacy_threshold": float(os.getenv("ORDERFLOW_THRESHOLD", "0.40")),
        "market_threshold": float(os.getenv("MARKET_FLOW_THRESHOLD", "0.40")),
        "raw_threshold": float(os.getenv("RAW_FOOTPRINT_THRESHOLD", "0.40")),
        "price_bucket": float(os.getenv("FOOTPRINT_PRICE_BUCKET", "25")),
        "full_credit_ratio": float(os.getenv("FOOTPRINT_FULL_CREDIT_RATIO", "1.5")),
    },
)
if os.getenv("FLOW_GATE_MODE", "independent").lower() == "calibrated" and flow_gate_config["gate_mode"] != "calibrated":
    logger.warning("Calibrated flow gate requested but no passed artifact is available; remaining in shadow mode")
predictor = Predictor(
    retrace_entry_atr=float(_retrace_atr) if _retrace_atr else None,
    retrace_pct=float(os.getenv("RETRACE_PCT", "0.5")),
    sweep_rearm_bars=max(1, int(os.getenv("SWEEP_REARM_BARS", "3"))),
    sweep_rearm_atr=max(0.0, float(os.getenv("SWEEP_REARM_ATR", "0.5"))),
    flow_gate_mode=flow_gate_config["gate_mode"],
    legacy_orderflow_threshold=flow_gate_config["legacy_threshold"],
    market_flow_threshold=flow_gate_config["market_threshold"],
    raw_footprint_threshold=flow_gate_config["raw_threshold"],
    footprint_price_bucket=flow_gate_config["price_bucket"],
    footprint_full_credit_ratio=flow_gate_config["full_credit_ratio"],
    venue_freshness_seconds=COLLECTOR_STALE_SECONDS,
    use_fixed_pct_exits=os.getenv("USE_FIXED_PCT_EXITS", "1").lower() in ("1", "true", "yes", "on"),
    stop_pct=float(os.getenv("FIXED_STOP_PCT", "0.01")),
    target_pct=float(os.getenv("FIXED_TARGET_PCT", "0.02")),
)
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
# The web process and the market loop share one process by design.  Keep the
# loop's heartbeat in memory so liveness checks never need to touch SQLite or
# the durable stores used by the predictor.
live_loop_started_at = None
live_loop_last_attempt_at = None
live_loop_last_completed_at = None
live_loop_last_error_at = None
live_loop_last_error = None
live_loop_last_attempt_monotonic = None
live_loop_last_completed_monotonic = None
live_loop_watchdog_alerted = False
live_loop_watchdog_thread = None
live_loop_started_monotonic = None
live_loop_phase = "not_started"
live_loop_phase_at = None
binance_pipeline_alerted = False
push_dispatch_alerted = False

subscription_store = JsonStore(data_dir / "push_subscriptions.json")
push_delivery_store = JsonStore(data_dir / "push_delivery.json")
push_delivery_events_store = JsonStore(data_dir / "push_delivery_events.json")
push_decision_events_store = JsonStore(data_dir / "push_decision_events.json")
paper_exit_push_store = JsonStore(data_dir / "paper_exit_push.json")
signal_lifecycle_store = JsonStore(data_dir / "signal_lifecycle.json")
signal_event_queue_store = JsonStore(data_dir / "signal_event_queue.json")
live_state_store = JsonStore(data_dir / "live_state.json")
research_status_store = JsonStore(os.getenv("BTC_RESEARCH_STATUS", str(data_dir / "research/status.json")))
funnel_diary_store = JsonStore(data_dir / "funnel_diary.json")
decision_snapshot_store = JsonStore(data_dir / "decision_snapshots.json")
shadow_book_store = JsonStore(data_dir / "shadow_book.json")
ops_reliability_store = JsonStore(data_dir / "ops_reliability.json")
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
SIGNAL_REPLACEMENT_DISTANCE_ATR = max(0.0, float(os.getenv("SIGNAL_REPLACEMENT_DISTANCE_ATR", "0.25")))
CONTINUATION_REARM_SECONDS = max(0, int(os.getenv("CONTINUATION_REARM_SECONDS", "1800")))
CONTINUATION_REARM_ATR = max(0.0, float(os.getenv("CONTINUATION_REARM_ATR", "1.0")))
PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL", "https://btc-structure-flow-predictor.onrender.com"
).rstrip("/")
# The predictor only needs a short, recent trade window for footprint and
# five-minute exchange-agreement features. Bound both the database result and
# the retention store so pandas cannot materialize an unbounded working set.
TRADE_LOOKBACK_MINUTES = max(20, min(30, int(os.getenv("TRADE_LOOKBACK_MINUTES", "30"))))
TRADE_QUERY_LIMIT = max(5_000, min(20_000, int(os.getenv("TRADE_QUERY_LIMIT", "20_000"))))
TRADE_STORE_MAX_ROWS = max(10_000, min(40_000, int(os.getenv("TRADE_STORE_MAX_ROWS", "40_000"))))

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
trade_store = TradeStore(data_dir / "live_trades.sqlite3", max_rows=TRADE_STORE_MAX_ROWS)
flow_state_store = FlowStateStore(
    data_dir / "flow_state.json",
    price_bucket=flow_gate_config["price_bucket"],
    retention_minutes=540,
)
paper_ledger = PaperLedger(
    os.getenv("PAPER_LEDGER_PATH", str(data_dir / "paper_ledger.json")),
    neutral_exit_observations=max(1, int(os.getenv("NEUTRAL_EXIT_OBSERVATIONS", "3"))),
    fee_bps=float(os.getenv("PAPER_FEE_BPS", str(live_policy.RESEARCH_FEE_BPS))),
    slippage_bps=float(os.getenv("PAPER_SLIPPAGE_BPS", str(live_policy.RESEARCH_SLIPPAGE_BPS))),
    max_notional_multiple=float(os.getenv("PAPER_MAX_NOTIONAL_MULTIPLE", str(live_policy.MAX_NOTIONAL_MULTIPLE))),
    daily_loss_r=float(os.getenv("PAPER_DAILY_LOSS_R", str(live_policy.DAILY_LOSS_R))),
    weekly_loss_r=float(os.getenv("PAPER_WEEKLY_LOSS_R", str(live_policy.WEEKLY_LOSS_R))),
    risk_fraction=float(os.getenv("PAPER_RISK_FRACTION", str(live_policy.RISK_FRACTION))),
    soft_filters=os.getenv("PAPER_SOFT_FILTERS", "1").lower() in ("1", "true", "yes", "on"),
    apply_research_costs=os.getenv("PAPER_APPLY_RESEARCH_COSTS", "1").lower() in ("1", "true", "yes", "on"),
    use_fixed_pct_exits=os.getenv("USE_FIXED_PCT_EXITS", "1").lower() in ("1", "true", "yes", "on"),
    max_hold_hours=float(os.getenv("PAPER_MAX_HOLD_HOURS", str(live_policy.MAX_HOLD_HOURS))),
    fill_min_rr=float(os.getenv("PAPER_FILL_MIN_RR", str(live_policy.FILL_MIN_RR))),
)
signal_lifecycle = SignalLifecycle(
    confirm_observations=SIGNAL_CONFIRM_OBSERVATIONS,
    invalidation_observations=SIGNAL_INVALIDATION_OBSERVATIONS,
    bias_observations=BIAS_CONFIRM_OBSERVATIONS,
    replacement_distance_atr=SIGNAL_REPLACEMENT_DISTANCE_ATR,
    continuation_rearm_seconds=CONTINUATION_REARM_SECONDS,
    continuation_rearm_atr=CONTINUATION_REARM_ATR,
)
funnel_diary = live_policy.FunnelDiary(funnel_diary_store)
decision_snapshot_log = live_policy.DecisionSnapshotLog(decision_snapshot_store)
shadow_book = live_policy.ShadowBook(shadow_book_store)
ops_reliability = live_policy.OpsReliability(ops_reliability_store)
try:
    live_policy.write_seeded_rescore_report(Path(data_dir) / "seeded_trade_rescore.json")
    # Also keep a repo-visible copy for offline review when runtime is work/.
    live_policy.write_seeded_rescore_report(Path("outputs") / "seeded_trade_rescore.json")
except Exception:
    logger.exception("Failed to write seeded trade rescore report")

MARKET_TYPE = os.getenv("MARKET_TYPE", "linear").lower()
if MARKET_TYPE not in ("spot", "linear"):
    MARKET_TYPE = "linear"
BINANCE_REST_ENABLED = os.getenv("BINANCE_REST_ENABLED", "1").lower() in ("1", "true", "yes", "on")
BINANCE_REST_STARTUP_BACKFILL = os.getenv("BINANCE_REST_STARTUP_BACKFILL", "1").lower() in ("1", "true", "yes", "on")
BINANCE_REST_GAP_RECOVERY = os.getenv("BINANCE_REST_GAP_RECOVERY", os.getenv("BINANCE_REST_ON_STALE", "1")).lower() in ("1", "true", "yes", "on")
BINANCE_REST_GAP_SECONDS = max(35, int(os.getenv("BINANCE_REST_GAP_SECONDS", "45")))
BINANCE_REST_ERROR_RETRY_SECONDS = max(5, int(os.getenv("BINANCE_REST_ERROR_RETRY_SECONDS", "60")))
BINANCE_TRADE_LIMIT = max(50, min(200, int(os.getenv("BINANCE_TRADE_LIMIT", "100"))))
BINANCE_FLOW_LIMIT = max(30, min(180, int(os.getenv("BINANCE_FLOW_LIMIT", "60"))))
BINANCE_USER_AGENT = os.getenv("BINANCE_USER_AGENT", "btc-structure-flow-predictor/0.1 (+local-research)")
BINANCE_REST_WEIGHT_CAP = max(100.0, min(1800.0, float(os.getenv("BINANCE_REST_WEIGHT_CAP", "1200"))))
BINANCE_REST_429_DEFAULT_COOLDOWN_SECONDS = max(1.0, float(os.getenv("BINANCE_REST_429_DEFAULT_COOLDOWN_SECONDS", "60")))
BINANCE_REST_418_DEFAULT_COOLDOWN_SECONDS = max(60.0, float(os.getenv("BINANCE_REST_418_DEFAULT_COOLDOWN_SECONDS", "600")))
BINANCE_AGG_TRADES_WEIGHT = max(1.0, float(os.getenv("BINANCE_AGG_TRADES_WEIGHT", "20")))
BINANCE_KLINES_WEIGHT = max(1.0, float(os.getenv("BINANCE_KLINES_WEIGHT", "2")))
BINANCE_REST_ON_STALE = BINANCE_REST_GAP_RECOVERY
LIVE_WATCHDOG_MISSED_POLLS = max(2, int(os.getenv("LIVE_WATCHDOG_MISSED_POLLS", "3")))
LIVE_WATCHDOG_CHECK_SECONDS = max(5, int(os.getenv("LIVE_WATCHDOG_CHECK_SECONDS", "15")))
LIVE_SUMMARY_LOG_SECONDS = max(60, int(os.getenv("LIVE_SUMMARY_LOG_SECONDS", "300")))
BINANCE_WS_STALE_ALERT_SECONDS = max(
    COLLECTOR_STALE_SECONDS,
    int(os.getenv("BINANCE_WS_STALE_ALERT_SECONDS", str(COLLECTOR_STALE_SECONDS))),
)
PUSH_LAG_ALERT_SECONDS = max(60, int(os.getenv("PUSH_LAG_ALERT_SECONDS", "300")))

binance_rest_limiter = BinanceRestLimiter(
    BinanceRestConfig(
        capacity_weight=BINANCE_REST_WEIGHT_CAP,
        window_seconds=60.0,
        default_429_cooldown_seconds=BINANCE_REST_429_DEFAULT_COOLDOWN_SECONDS,
        default_418_cooldown_seconds=BINANCE_REST_418_DEFAULT_COOLDOWN_SECONDS,
    )
)


def _configured_proxy_url():
    """Return the explicitly configured outbound proxy, if any.

    QuotaGuard's variable isn't one Requests automatically recognizes, so it
    must be mapped explicitly. Standard proxy variables remain supported.
    """
    for key in (
        "BINANCE_WS_PROXY_URL",
        "QUOTAGUARDSTATIC_URL",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
    ):
        value = os.getenv(key, "").strip()
        if value:
            return key, value
    return None, None


def _proxy_diagnostics():
    key, value = _configured_proxy_url()
    if not value:
        return {
            "configured": False,
            "required": False,
            "status": "direct",
            "variable": None,
            "provider": None,
            "host": None,
        }
    host = urlparse(value).hostname
    provider = (
        "quotaguard"
        if key == "QUOTAGUARDSTATIC_URL"
        else "binance_proxy"
        if key == "BINANCE_WS_PROXY_URL"
        else "generic"
    )
    return {
        "configured": True,
        "required": False,
        "status": "configured",
        "variable": key,
        "provider": provider,
        "host": host,
    }


def _binance_proxy_required():
    """Compatibility diagnostic; proxy routing is always optional."""
    return False


def _http_get(url, params=None, timeout=10, binance_weight=1.0):
    headers = {"User-Agent": BINANCE_USER_AGENT, "Accept": "application/json"}
    hostname = (urlparse(url).hostname or "").lower()
    is_binance = hostname == "fapi.binance.com"
    _, binance_proxy_url = _configured_proxy_url()
    proxy_url = (
        binance_proxy_url
        if is_binance
        else os.getenv("BYBIT_REST_PROXY_URL", "").strip() or None
    )
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
    if is_binance:
        binance_rest_limiter.acquire(binance_weight)
    try:
        response = requests.get(
            url,
            params=params or {},
            headers=headers,
            timeout=timeout,
            proxies=proxies,
        )
    except Exception as exc:
        if is_binance:
            binance_rest_limiter.observe_error(exc)
        raise
    if is_binance:
        binance_rest_limiter.observe_response(
            getattr(response, "status_code", 200),
            getattr(response, "headers", {}) or {},
        )
    try:
        response.raise_for_status()
    except Exception as exc:
        if is_binance:
            binance_rest_limiter.observe_error(exc)
        raise
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
    response = _http_get(url, params={"symbol":"BTCUSDT","limit":BINANCE_TRADE_LIMIT}, timeout=10, binance_weight=BINANCE_AGG_TRADES_WEIGHT)
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
    response = _http_get(url, params={"symbol":"BTCUSDT","interval":"1m","limit":limit}, timeout=10, binance_weight=BINANCE_KLINES_WEIGHT)
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
        "accepted_at": max(
            (event.get("accepted_at") for event in batch if event.get("accepted_at")),
            default=None,
        ),
        "received_at": max(
            (event.get("received_at") for event in batch if event.get("received_at")),
            default=None,
        ),
        "notification_created_at": max(
            (
                event.get("notification_created_at")
                for event in batch
                if event.get("notification_created_at")
            ),
            default=None,
        ),
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
        if event_type == "trade_opened" and signal_id:
            superseded[signal_id] = "paper_entry_filled"
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


def _discard_strategy_only_notifications():
    """Remove legacy signal alerts that do not prove paper execution.

    Strategy confirmation remains visible in diagnostics, but Web Push is
    reserved for an actual paper fill or terminal paper exit.  This migration
    also prevents setup alerts queued by an older deployment from being sent
    after the execution-truth change ships.
    """
    state = _signal_queue_state()
    retained = []
    removed = 0
    for event in state["pending"]:
        if str(event.get("event_type") or "").startswith("setup_"):
            removed += 1
            _record_push_decision(
                event.get("event_id"),
                status="suppressed_not_executed",
                suppressed_at=_utcnow().isoformat(),
                suppressed_reason="paper_execution_not_proven",
            )
        else:
            retained.append(event)
    if removed:
        state["pending"] = retained
        _write_signal_queue(state)
    return removed


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


def _push_dispatch_diagnostics(now=None):
    """Distinguish an idle event queue from a stalled push dispatcher."""
    now = pd.Timestamp(now or _utcnow())
    queue = _signal_queue_state()
    pending = list(queue.get("pending") or [])
    oldest = None
    for item in pending:
        created = item.get("created_at")
        if not created:
            continue
        try:
            timestamp = pd.Timestamp(created)
        except (TypeError, ValueError):
            continue
        oldest = timestamp if oldest is None or timestamp < oldest else oldest
    pending_age = (
        max(0.0, (now - oldest).total_seconds()) if oldest is not None else None
    )
    delivery = _latest_delivery_summary("automatic")
    decision = _latest_push_decision()
    if pending:
        status = "lagging" if pending_age is not None and pending_age >= PUSH_LAG_ALERT_SECONDS else "pending"
    elif delivery:
        status = "idle_no_pending_event"
    else:
        status = "never_attempted"
    return {
        "status": status,
        "pending_events": len(pending),
        "oldest_pending_at": oldest.isoformat() if oldest is not None else None,
        "pending_age_seconds": round(pending_age, 1) if pending_age is not None else None,
        "last_event_at": decision.get("observed_at"),
        "last_decision_status": decision.get("status"),
        "last_attempt_at": delivery.get("attempted_at"),
        "last_accepted_at": delivery.get("accepted_at"),
        "last_received_at": delivery.get("received_at"),
        "last_notification_created_at": delivery.get("notification_created_at"),
        "lag_alert_threshold_seconds": PUSH_LAG_ALERT_SECONDS,
    }


def _pipeline_watchdog(collectors, now):
    """Surface Binance and push stalls as transition-based error logs."""
    global binance_pipeline_alerted, push_dispatch_alerted
    binance = (collectors or {}).get("binance") or {}
    lag_seconds = binance.get("lag_seconds")
    binance_stale = (
        bool(binance.get("stale"))
        or lag_seconds is None
        or float(lag_seconds) > BINANCE_WS_STALE_ALERT_SECONDS
    )
    if binance_stale and not binance_pipeline_alerted:
        binance_pipeline_alerted = True
        logger.error(
            "Binance pipeline alert: stale_or_silent lag=%ss error=%s proxy_required=%s proxy_configured=%s",
            binance.get("lag_seconds"),
            binance.get("error"),
            binance.get("proxy_required"),
            binance.get("proxy_configured"),
        )
    elif not binance_stale and binance_pipeline_alerted:
        binance_pipeline_alerted = False
        logger.info("Binance pipeline alert cleared: heartbeat recovered")

    push = _push_dispatch_diagnostics(now)
    push_lagging = push["status"] == "lagging"
    if push_lagging and not push_dispatch_alerted:
        push_dispatch_alerted = True
        logger.error(
            "Push dispatcher alert: pending event age=%ss threshold=%ss decision=%s",
            push.get("pending_age_seconds"),
            PUSH_LAG_ALERT_SECONDS,
            push.get("last_decision_status"),
        )
    elif not push_lagging and push_dispatch_alerted:
        push_dispatch_alerted = False
        logger.info("Push dispatcher alert cleared: queue is no longer lagging")
    return push


def _paper_open_event_id(position):
    identity = {
        key: position.get(key)
        for key in ("signal_id", "entry_time", "side", "entry", "size")
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"paper-open-{digest}"


def _paper_open_event(position):
    side = str(position.get("side") or "trade").capitalize()
    entry = float(position.get("entry") or 0.0)
    stop = float(position.get("stop") or 0.0)
    target = float(position.get("target") or 0.0)
    event_id = _paper_open_event_id(position)
    return {
        "event_id": event_id,
        "event_type": "trade_opened",
        "signal_id": position.get("signal_id"),
        "created_at": position.get("filled_at") or position.get("entry_time") or _utcnow().isoformat(),
        "cooldown_exempt": True,
        "title": "BTC paper trade opened",
        "body": (
            f"{side} filled · Entry ${entry:,.2f} · "
            f"SL ${stop:,.2f} · TP ${target:,.2f}"
        ),
    }


def _paper_open_notification_events(paper_status):
    """Return fill notifications only for positions that remain open."""
    newly_closed = list((paper_status or {}).get("newly_closed") or [])
    closed_identities = {
        (trade.get("signal_id"), trade.get("entry_time"))
        for trade in newly_closed
    }
    events = []
    for position in (paper_status or {}).get("newly_opened") or []:
        identity = (position.get("signal_id"), position.get("entry_time"))
        if identity in closed_identities:
            continue
        events.append(_paper_open_event(position))
    return events


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
    if reason == "target":
        outcome = "Target hit"
    elif reason == "stop":
        outcome = "Stop hit"
    elif reason == "superseded_by_confirmed_setup":
        outcome = "Setup replaced"
    elif reason == "signal_neutralized":
        outcome = "Signal neutralized"
    elif reason == "signal_flipped":
        outcome = "Signal flipped"
    else:
        outcome = reason.replace("_", " ").capitalize()
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
        if str(trade.get("exit_reason") or "").lower() not in ("target", "stop", "superseded_by_confirmed_setup", "signal_neutralized", "signal_flipped"):
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


def _live_poll_interval_seconds():
    return max(20, int(os.getenv("LIVE_POLL_SECONDS", "45")))


def _prediction_fail_closed(result, reason: str):
    """Clear tradeable fields while preserving diagnostics (research-only mode)."""
    from dataclasses import replace
    from btc_predictor.models import PredictorOutput
    if result is None:
        return result
    if isinstance(result, PredictorOutput):
        return replace(
            result,
            setup_type=None,
            entry=None,
            stop=None,
            target=None,
            position_size=0.0,
            orderflow_confirmation=False,
            no_trade_reason=reason,
            probability_tp_before_sl=result.probability_tp_before_sl,
        )
    # Dict-shaped persisted predictions.
    blocked = dict(result)
    blocked.update({
        "setup_type": None,
        "entry": None,
        "stop": None,
        "target": None,
        "position_size": 0.0,
        "orderflow_confirmation": False,
        "no_trade_reason": reason,
    })
    return blocked


def _governance_payload(paper_status=None, data_quality=None):
    paper_status = paper_status if paper_status is not None else paper_ledger._status()
    closed = int(paper_status.get("closed_trades") or 0)
    return {
        "policy": live_policy.policy_manifest(),
        "data_quality": data_quality or live_policy.evaluate_data_quality(
            market_type=MARKET_TYPE,
            binance_feed_mode=None,
            stale_exchanges=[],
        ),
        "economics": paper_status.get("economics") or live_policy.research_economics(),
        "pnl_reporting": paper_status.get("pnl_reporting"),
        "retune_discipline": paper_status.get("retune_discipline")
            or live_policy.retune_discipline_status(closed),
        "funnel": funnel_diary.status(),
        "shadow_book": shadow_book.status(),
        "decision_snapshots": decision_snapshot_log.status(),
        "calibration": live_policy.calibration_status(
            flow_calibration_artifact, flow_gate_config
        ),
        "ops": ops_reliability.status(
            paper_status=paper_status,
            live_loop=_live_loop_diagnostics(),
            data_quality=data_quality,
        ),
        "seeded_trade_rescore": live_policy.rescore_seeded_trades(),
        "probability_policy": {
            "source": live_policy.PROBABILITY_SOURCE,
            "use": live_policy.PROBABILITY_USE,
            "sizing": "fixed_risk_fraction_only",
            "lifecycle_ranking": "soft_diagnostic_only",
        },
    }


def _process_rss_mb():
    """Read current resident memory on Linux without a third-party package."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(float(line.split()[1]) / 1024.0, 1)
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None
    return None


def _release_transient_memory():
    """Collect pandas object graphs and return free glibc arenas to the OS."""
    gc.collect()
    if _malloc_trim is not None:
        try:
            _malloc_trim(0)
        except Exception:
            logger.debug("malloc_trim unavailable", exc_info=True)


def _select_flow_baseline(ws_flow, rest_flow, decision_time):
    """Choose a fresh flow baseline independently of raw-trade freshness."""
    decision_time = pd.Timestamp(decision_time)
    if ws_flow is not None and len(ws_flow) >= 2 and ws_flow.index[-1] >= decision_time - pd.Timedelta(minutes=3):
        return ws_flow.loc[ws_flow.index <= decision_time], "websocket"
    if rest_flow is not None and not rest_flow.empty and rest_flow.index[-1] >= decision_time - pd.Timedelta(minutes=2):
        return rest_flow.loc[rest_flow.index <= decision_time], "rest_backfill"
    return None, None


class ClosedBarDecisionGate:
    """Make each closed-bar decision immutable after its first evaluation."""

    def __init__(self):
        self.bar_at = None
        self.value = None

    def should_evaluate(self, bar_at):
        bar_at = pd.Timestamp(bar_at)
        return self.bar_at is None or bar_at > self.bar_at

    def commit(self, bar_at, value):
        """Commit only after lifecycle/accounting writes have succeeded."""
        self.bar_at = pd.Timestamp(bar_at)
        self.value = value


def _live_loop_diagnostics():
    """Return in-memory loop health without acquiring application or DB locks."""
    now = time.monotonic()
    threshold = float(_live_poll_interval_seconds() * LIVE_WATCHDOG_MISSED_POLLS)
    started_age = None
    completed_age = None
    if live_loop_started_at and live_loop_started_monotonic is not None:
        started_age = round(max(0.0, now - live_loop_started_monotonic), 1)
    if live_loop_last_completed_monotonic is not None:
        completed_age = round(max(0.0, now - live_loop_last_completed_monotonic), 1)
    stale = False
    if live_loop_started_at:
        stale = (
            completed_age is None
            and started_age is not None
            and started_age > threshold
        ) or (completed_age is not None and completed_age > threshold)
    return {
        "thread_alive": bool(live_thread and live_thread.is_alive()),
        "started_at": live_loop_started_at,
        "last_attempt_at": live_loop_last_attempt_at,
        "last_completed_at": live_loop_last_completed_at,
        "last_error_at": live_loop_last_error_at,
        "last_error": live_loop_last_error,
        "phase": live_loop_phase,
        "phase_at": live_loop_phase_at,
        "last_completed_age_seconds": completed_age,
        "stale": stale,
        "stale_after_seconds": int(threshold),
        "watchdog_alerted": bool(live_loop_watchdog_alerted),
    }


def _set_live_loop_phase(phase):
    """Publish the active poll phase without disk or shared-lock access."""
    global live_loop_phase, live_loop_phase_at
    live_loop_phase = str(phase)
    live_loop_phase_at = pd.Timestamp.now(tz="UTC").isoformat()


def _live_loop_watchdog():
    """Log stalled-loop alerts without restarting the web process."""
    global live_loop_watchdog_alerted
    while True:
        time.sleep(LIVE_WATCHDOG_CHECK_SECONDS)
        if not live_thread_started:
            continue
        diagnostics = _live_loop_diagnostics()
        if diagnostics["stale"] and not live_loop_watchdog_alerted:
            live_loop_watchdog_alerted = True
            logger.error(
                "Live predictor watchdog: no completed poll for %ss (last_completed=%s, last_error=%s)",
                diagnostics["stale_after_seconds"],
                diagnostics["last_completed_at"],
                diagnostics["last_error"],
            )
        elif not diagnostics["stale"] and live_loop_watchdog_alerted:
            live_loop_watchdog_alerted = False
            logger.info("Live predictor watchdog: poll loop recovered")


def _live_loop():
    global live_state
    global live_loop_started_at, live_loop_last_attempt_at
    global live_loop_last_completed_at, live_loop_last_error_at, live_loop_last_error
    global live_loop_last_attempt_monotonic, live_loop_last_completed_monotonic
    global live_loop_started_monotonic
    backfill_controller = BinanceBackfillController(
        startup_enabled=BINANCE_REST_ENABLED and BINANCE_REST_STARTUP_BACKFILL,
        gap_recovery_enabled=BINANCE_REST_ENABLED and BINANCE_REST_GAP_RECOVERY,
        gap_seconds=BINANCE_REST_GAP_SECONDS,
    )
    flow_bars_cache = None
    persisted_state = live_state_store.read({})
    def persisted_timestamp(key):
        value = persisted_state.get(key)
        try:
            return pd.Timestamp(value) if value else None
        except (TypeError, ValueError):
            return None
    binance_rest_last_ok_at = persisted_timestamp("binance_rest_last_ok_at")
    binance_rest_last_error = persisted_state.get("binance_rest_last_error")
    binance_rest_last_reason = persisted_state.get("binance_rest_last_reason")
    binance_flow_last_ok_at = persisted_timestamp("binance_flow_last_ok_at")
    binance_flow_last_error = persisted_state.get("binance_flow_last_error")
    bybit_rest_last_ok_at = persisted_timestamp("bybit_rest_last_ok_at")
    bybit_rest_last_error = persisted_state.get("bybit_rest_last_error")
    flow_baseline_last_ok_at = persisted_timestamp("flow_baseline_last_ok_at")
    orderflow_input_last_ok_at = persisted_timestamp("orderflow_input_last_ok_at")
    last_summary_log_at = pd.Timestamp(0, tz="UTC")
    decision_gate = ClosedBarDecisionGate()
    while True:
        try:
            poll_started = pd.Timestamp.now(tz="UTC")
            if live_loop_started_at is None:
                live_loop_started_at = poll_started.isoformat()
                live_loop_started_monotonic = time.monotonic()
            live_loop_last_attempt_at = poll_started.isoformat()
            live_loop_last_attempt_monotonic = time.monotonic()
            with live_lock: live_state.update({"status":"polling","updated_at":poll_started.isoformat()})
            _set_live_loop_phase("fetching_bybit")
            ohlc, recent, frames = _bybit_data(); sources = "bybit"; trade_store.append(recent)
            bybit_rest_last_ok_at = poll_started
            bybit_rest_last_error = None
            _set_live_loop_phase("checking_binance")
            binance_latest = trade_store.exchange_latest("binance")
            binance_lag = None
            if binance_latest is not None:
                if binance_latest.tzinfo is None:
                    binance_latest = binance_latest.tz_localize("UTC")
                binance_lag = float((poll_started - binance_latest).total_seconds())
            collectors = trade_store.collector_status(poll_started, COLLECTOR_STALE_SECONDS)
            binance_collector = collectors.get("binance", {}) or {}
            binance_ws_last = binance_collector.get("last_message_at")
            binance_ws_fresh = False
            if binance_ws_last:
                binance_ws_ts = pd.Timestamp(binance_ws_last)
                if binance_ws_ts.tzinfo is None:
                    binance_ws_ts = binance_ws_ts.tz_localize("UTC")
                binance_ws_fresh = bool(binance_collector.get("connected")) and (
                    (poll_started - binance_ws_ts).total_seconds()
                    <= COLLECTOR_STALE_SECONDS
                )
            proxy_diagnostics = _proxy_diagnostics()
            rest_reason = (
                backfill_controller.decide(poll_started, binance_ws_fresh)
                if BINANCE_REST_ENABLED
                else None
            )
            binance_data_path = (
                "websocket"
                if binance_ws_fresh
                else "cached"
                if binance_lag is not None and binance_lag <= COLLECTOR_STALE_SECONDS
                else "stale"
            )
            if rest_reason:
                try:
                    backfill_trades = _binance_trades()
                    backfill_flow = _binance_flow_bars()
                    inserted = trade_store.append(backfill_trades)
                    flow_bars_cache = backfill_flow
                    binance_rest_last_ok_at = poll_started
                    binance_rest_last_error = None
                    binance_rest_last_reason = rest_reason
                    binance_flow_last_ok_at = poll_started
                    binance_flow_last_error = None
                    backfill_controller.mark_success(
                        rest_reason, poll_started, binance_ws_fresh
                    )
                    binance_data_path = f"rest_{rest_reason}"
                    logger.info(
                        "Binance REST %s backfill completed inserted=%s ws_fresh=%s",
                        rest_reason,
                        inserted,
                        binance_ws_fresh,
                    )
                except (BinanceRateLimited, BinanceRestDeferred) as exc:
                    backfill_controller.mark_failure(
                        rest_reason,
                        poll_started,
                        exc.retry_after_seconds,
                    )
                    binance_rest_last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                    logger.warning(
                        "Binance REST %s deferred for %.3fs: %s",
                        rest_reason,
                        exc.retry_after_seconds,
                        exc,
                    )
                except Exception as exc:
                    backfill_controller.mark_failure(
                        rest_reason,
                        poll_started,
                        BINANCE_REST_ERROR_RETRY_SECONDS,
                    )
                    binance_rest_last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                    logger.warning(
                        "Binance REST %s failed; retrying after %ss: %s",
                        rest_reason,
                        BINANCE_REST_ERROR_RETRY_SECONDS,
                        exc,
                    )
            now=ohlc.index[-1]
            _set_live_loop_phase("querying_trades")
            trades=trade_store.query(
                now-pd.Timedelta(minutes=TRADE_LOOKBACK_MINUTES),
                now,
                limit=TRADE_QUERY_LIMIT,
                include_trade_id=False,
            )
            raw_flow_feature_bars = (
                int(pd.to_datetime(trades.time, utc=True).dt.ceil("1min").nunique())
                if not trades.empty and "time" in trades
                else 0
            )
            flow_bars=None
            recent_trades=trades.loc[trades.time>=now-pd.Timedelta(minutes=2)] if "time" in trades else trades
            available_exchanges=set(recent_trades.exchange.astype(str)) if "exchange" in recent_trades else set()
            sources="+".join(exchange for exchange in ("bybit","binance") if exchange in available_exchanges) or sources
            store_stats = trade_store.stats()
            collectors = trade_store.collector_status(poll_started, COLLECTOR_STALE_SECONDS)
            # Flow baseline: prefer WebSocket-derived Binance 1m klines (aggTrade
            # deltas + kline taker-buy volume arrive on the same connection).
            # REST klines remain an explicit fallback only.
            ws_flow = trade_store.flow_bars_df("binance", limit=180)
            # A successfully fetched REST flow cache remains valid regardless
            # of the separate raw-trade freshness test. The old coupling
            # discarded valid bars on most polls.
            flow_bars, flow_source = _select_flow_baseline(ws_flow, flow_bars_cache, now)
            if flow_source:
                flow_baseline_last_ok_at = poll_started
            if flow_source or raw_flow_feature_bars >= 20:
                orderflow_input_last_ok_at = poll_started
            # Persist only immutable closed-minute aggregates. Repeated polls
            # of the same bar and late events cannot revise a prior decision.
            flow_state_store.update(trades, now)
            flow_aggregates = flow_state_store.footprint_bars(
                now - pd.Timedelta(minutes=predictor.reclaim_bars + 2), now
            )
            session_cvd = flow_state_store.session_cvd(now)
            _set_live_loop_phase("predicting")
            # One immutable strategy/lifecycle decision per unique closed 1m
            # bar. Late trades never repaint an already-published decision.
            new_decision = decision_gate.should_evaluate(now)
            if new_decision:
                result = predictor.predict(
                    ohlc,
                    trades,
                    100_000,
                    frames=frames,
                    flow_bars=flow_bars,
                    flow_source=flow_source or "recent_trades_fallback",
                    flow_aggregates=flow_aggregates,
                    session_cvd=session_cvd,
                )
            else:
                result = decision_gate.value
            # Fail closed on bad data before paper lifecycle can confirm entries.
            binance_ws_fresh_pre = bool((collectors.get("binance") or {}).get("fresh"))
            bybit_pipeline_ok_pre = bool((collectors.get("bybit") or {}).get("fresh"))
            binance_pipeline_ok_pre = binance_ws_fresh_pre and flow_source is not None
            stale_pre = []
            if not bybit_pipeline_ok_pre:
                stale_pre.append("bybit")
            if not binance_pipeline_ok_pre:
                stale_pre.append("binance")
            data_quality = live_policy.evaluate_data_quality(
                market_type=MARKET_TYPE,
                binance_feed_mode=(collectors.get("binance") or {}).get("mode"),
                stale_exchanges=stale_pre,
                collectors=collectors,
                binance_data_path=binance_data_path,
            )
            if new_decision and not data_quality.get("tradable"):
                result = _prediction_fail_closed(result, "data_quality_fail_closed")
            _set_live_loop_phase("updating_paper")
            # Apply market facts first. Raw predictions are never allowed to
            # open, invalidate or supersede a paper position.
            market_paper_status = paper_ledger.update_market(ohlc)
            notification_now = pd.Timestamp.now(tz="UTC")
            lifecycle_before = signal_lifecycle_store.read(
                SignalLifecycle.initial_state()
            )
            legacy_position = (
                market_paper_status.get("open_position")
                or market_paper_status.get("pending_order")
            )
            lifecycle_before, adopted_signal_id = signal_lifecycle.adopt_open_position(
                lifecycle_before, legacy_position, notification_now
            )
            if adopted_signal_id:
                paper_ledger.bind_active_signal(adopted_signal_id)
                logger.info("Adopted legacy paper position into lifecycle signal_id=%s", adopted_signal_id)
            active_before = lifecycle_before.get("active") or {}
            definitive_exit = any(
                str(trade.get("exit_reason") or "").lower() in ("target", "stop")
                for trade in (market_paper_status.get("newly_closed") or [])
            )
            if definitive_exit:
                _cancel_signal_events(
                    active_before.get("signal_id"), "paper_exit"
                )
            _set_live_loop_phase("notifications")
            lifecycle_events = []
            paper_status = market_paper_status
            if new_decision:
                lifecycle_state, lifecycle_events = signal_lifecycle.evaluate(
                    lifecycle_before, result, market_paper_status, notification_now
                )
                # Drop new entries when data quality is research-only; still
                # allow invalidation/expiry events for open theses.
                if not data_quality.get("tradable"):
                    lifecycle_events = [
                        event for event in lifecycle_events
                        if str(event.get("event_type") or "") != "setup_confirmed"
                    ]
                decision_paper_status = paper_ledger.apply_lifecycle(lifecycle_events, ohlc)
                # Persist lifecycle after idempotent ledger application. If
                # the process dies between these writes, replaying the event
                # cannot duplicate an entry or close.
                signal_lifecycle_store.write(lifecycle_state)
                flow_state_store.record_sweeps(result.sweep_observations, now)
                decision_gate.commit(now, result)
                paper_status = decision_paper_status
                paper_status["newly_closed"] = (
                    list(market_paper_status.get("newly_closed") or [])
                    + list(decision_paper_status.get("newly_closed") or [])
                )
                paper_status["newly_opened"] = (
                    list(market_paper_status.get("newly_opened") or [])
                    + list(decision_paper_status.get("newly_opened") or [])
                )
                # Instrumentation: funnel, decision snapshots, shadow book.
                try:
                    funnel_diary.record_prediction(
                        result,
                        ts=notification_now,
                        blocked_data=not data_quality.get("tradable"),
                    )
                    if paper_status.get("newly_closed"):
                        funnel_diary.record("paper_exits", ts=notification_now, n=len(paper_status["newly_closed"]))
                    if paper_status.get("newly_opened"):
                        funnel_diary.record("paper_entries", ts=notification_now)
                    reject = paper_status.get("last_reject") or {}
                    if reject.get("reason") == "soft_filter":
                        funnel_diary.record("soft_filter_skips", ts=notification_now)
                    if reject.get("reason") == "risk_cap":
                        funnel_diary.record("risk_cap_skips", ts=notification_now)
                    decision_snapshot_log.append(
                        SignalLifecycle.snapshot(result),
                        meta={
                            "data_quality": data_quality,
                            "lifecycle_events": [e.get("event_type") for e in lifecycle_events],
                            "last_reject": reject or None,
                        },
                    )
                    for event in lifecycle_events:
                        if str(event.get("event_type") or "") == "setup_confirmed":
                            shadow_book.observe_confirmed(event)
                except Exception:
                    logger.exception("Live governance instrumentation failed")
            try:
                _notify_paper_exits(paper_status.get("newly_closed") or [])
            except Exception:
                logger.exception("Paper exit notification dispatch failed")
            for closed_trade in paper_status.get("newly_closed") or []:
                _cancel_signal_events(
                    closed_trade.get("signal_id"), "paper_trade_closed"
                )
            # Strategy lifecycle events are diagnostics, not proof that the
            # ledger accepted or filled an entry.  Push only actual fills;
            # paper exits are handled by the durable exit dispatcher above.
            _discard_strategy_only_notifications()
            _enqueue_signal_events(_paper_open_notification_events(paper_status))
            # TP/SL is always the only alert dispatched in its poll. Lifecycle
            # alerts resume next poll and retain their durable queue position.
            if not paper_status.get("newly_closed"):
                _dispatch_signal_event(notification_now)
            push_dispatch_diagnostics = _pipeline_watchdog(collectors, notification_now)
            _set_live_loop_phase("pruning")
            trade_store.prune(now-pd.Timedelta(minutes=int(os.getenv("TRADE_RETENTION_MINUTES","120"))))
            now = notification_now
            try:
                _retry_unacknowledged_pushes(now)
            except Exception as exc:
                logger.warning("Web Push acknowledgement retry failed: %s", exc)
            bybit_pipeline_ok = bool((collectors.get("bybit") or {}).get("fresh"))
            binance_pipeline_ok = binance_ws_fresh and flow_source is not None
            stale_exchanges = []
            if not bybit_pipeline_ok:
                stale_exchanges.append("bybit")
            if not binance_pipeline_ok:
                stale_exchanges.append("binance")
            feed_status = "degraded" if stale_exchanges else "live"
            limiter_status = binance_rest_limiter.snapshot()
            backfill_status = backfill_controller.snapshot(notification_now)
            cached_status = (
                "fresh"
                if binance_lag is not None and binance_lag <= COLLECTOR_STALE_SECONDS
                else "stale"
                if binance_lag is not None
                else "empty"
            )
            data_health = {
                "bybit_rest_market": {
                    "status": "ok" if bybit_rest_last_ok_at else "never_succeeded",
                    "last_successful_fetch_at": bybit_rest_last_ok_at.isoformat() if bybit_rest_last_ok_at else None,
                    "last_error": bybit_rest_last_error,
                },
                "bybit_websocket_trades": {
                    "status": "ok" if bybit_pipeline_ok else "stale",
                    "last_successful_fetch_at": (collectors.get("bybit") or {}).get("last_message_at"),
                    "last_error": (collectors.get("bybit") or {}).get("error"),
                },
                "binance_websocket": {
                    "status": "ok" if binance_ws_fresh else "stale_or_silent",
                    "last_successful_fetch_at": binance_ws_last,
                    "last_kline_at": (collectors.get("binance") or {}).get("latest_kline_at"),
                    "last_error": (collectors.get("binance") or {}).get("error"),
                    "transport": (collectors.get("binance") or {}).get("transport", "direct"),
                    "proxy_required": False,
                    "proxy_configured": (collectors.get("binance") or {}).get("proxy_configured", proxy_diagnostics.get("configured")),
                    "reconnect_attempt": (collectors.get("binance") or {}).get("reconnect_attempt"),
                    "reconnect_in_seconds": (collectors.get("binance") or {}).get("reconnect_in_seconds"),
                },
                "binance_rest_backfill": {
                    "status": (
                        "cooldown"
                        if limiter_status["cooldown"]
                        else "error"
                        if binance_rest_last_error
                        else "idle"
                        if binance_rest_last_ok_at and not binance_rest_last_error
                        else "not_used"
                    ),
                    "last_successful_fetch_at": binance_rest_last_ok_at.isoformat() if binance_rest_last_ok_at else None,
                    "last_error": binance_rest_last_error,
                    "last_reason": binance_rest_last_reason,
                    "controller": backfill_status,
                    "rate_limit": limiter_status,
                },
                "binance_cached_data": {
                    "status": cached_status,
                    "latest_trade_at": binance_latest.isoformat() if binance_latest is not None else None,
                    "lag_seconds": round(binance_lag, 1) if binance_lag is not None else None,
                },
                "binance_flow_baseline": {
                    "status": "ok" if flow_source else "unavailable",
                    "source": flow_source,
                    "last_successful_fetch_at": flow_baseline_last_ok_at.isoformat() if flow_baseline_last_ok_at else None,
                    "last_rest_fetch_at": binance_flow_last_ok_at.isoformat() if binance_flow_last_ok_at else None,
                    "last_error": binance_flow_last_error,
                },
                "orderflow_input": {
                    "status": "ok" if flow_source or raw_flow_feature_bars >= 20 else "warmup",
                    "source": flow_source or "recent_trades_fallback",
                    "feature_bars": len(flow_bars) if flow_bars is not None else raw_flow_feature_bars,
                    "minimum_feature_bars": 20,
                    "last_successful_fetch_at": orderflow_input_last_ok_at.isoformat() if orderflow_input_last_ok_at else None,
                    "last_error": None,
                },
                "session_cvd": session_cvd,
                "flow_gate": {
                    "mode": flow_gate_config["gate_mode"],
                    "legacy_threshold": flow_gate_config["legacy_threshold"],
                    "market_threshold": flow_gate_config["market_threshold"],
                    "raw_threshold": flow_gate_config["raw_threshold"],
                    "price_bucket": flow_gate_config["price_bucket"],
                    "full_credit_ratio": flow_gate_config["full_credit_ratio"],
                    "calibration_run_hash": flow_gate_config.get("artifact_run_hash"),
                },
            }
            # Refresh data-quality with final feed_status inputs.
            data_quality = live_policy.evaluate_data_quality(
                market_type=MARKET_TYPE,
                binance_feed_mode=collectors.get("binance", {}).get("mode", "unknown"),
                stale_exchanges=stale_exchanges,
                collectors=collectors,
                binance_data_path=binance_data_path,
            )
            pred_dict = dict(result.__dict__) if hasattr(result, "__dict__") else dict(result or {})
            pred_dict["signal_id"] = SignalLifecycle.signal_id(
                SignalLifecycle.snapshot(result)
            )
            pred_dict["probability_source"] = live_policy.PROBABILITY_SOURCE
            pred_dict["probability_use"] = live_policy.PROBABILITY_USE
            pred_dict["probability_tp_before_sl_is_heuristic"] = True
            next_state = {
                "status": feed_status,
                "source": sources,
                "market_type": MARKET_TYPE,
                "prediction": pred_dict,
                "paper": paper_status,
                "binance_feed_mode": collectors.get("binance", {}).get("mode", "unknown"),
                "flow_source": flow_source,
                "binance_data_path": binance_data_path,
                "binance_rest_last_ok_at": binance_rest_last_ok_at.isoformat() if binance_rest_last_ok_at else None,
                "binance_rest_last_error": binance_rest_last_error,
                "binance_rest_last_reason": binance_rest_last_reason,
                "binance_flow_last_ok_at": binance_flow_last_ok_at.isoformat() if binance_flow_last_ok_at else None,
                "binance_flow_last_error": binance_flow_last_error,
                "bybit_rest_last_ok_at": bybit_rest_last_ok_at.isoformat() if bybit_rest_last_ok_at else None,
                "bybit_rest_last_error": bybit_rest_last_error,
                "flow_baseline_last_ok_at": flow_baseline_last_ok_at.isoformat() if flow_baseline_last_ok_at else None,
                "orderflow_input_last_ok_at": orderflow_input_last_ok_at.isoformat() if orderflow_input_last_ok_at else None,
                "data_health": data_health,
                "data_quality": data_quality,
                "governance": _governance_payload(paper_status, data_quality),
                "decision": {
                    "policy": "immutable_closed_bar",
                    "bar_at": decision_gate.bar_at.isoformat() if decision_gate.bar_at is not None else None,
                    "evaluated_this_poll": new_decision,
                },
                "proxy": proxy_diagnostics,
                "push_dispatch": push_dispatch_diagnostics,
                "collectors": collectors,
                "stale_exchanges": stale_exchanges,
                "updated_at": now.isoformat(),
                "error": (f"stale feeds: {', '.join(stale_exchanges)}" if stale_exchanges else None),
            }
            live_state_store.write(next_state)
            with live_lock: live_state = next_state
            live_loop_last_completed_at = now.isoformat()
            live_loop_last_completed_monotonic = time.monotonic()
            live_loop_last_error_at = None
            live_loop_last_error = None
            _set_live_loop_phase("sleeping")
            if (now - last_summary_log_at).total_seconds() >= LIVE_SUMMARY_LOG_SECONDS:
                logger.info(
                    "live_poll_complete feed=%s source=%s flow_source=%s bias=%s reason=%s sweep=%s flow_eval=%s flow_score=%s binance_path=%s proxy=%s rss_mb=%s",
                    feed_status,
                    sources,
                    flow_source,
                    result.bias,
                    result.no_trade_reason,
                    result.sweep_status,
                    result.orderflow_evaluation_status,
                    result.orderflow_score,
                    binance_data_path,
                    proxy_diagnostics.get("status") or "unknown",
                    _process_rss_mb(),
                )
                last_summary_log_at = now
        except Exception as exc:
            logger.exception("Live market poll failed")
            error_at = pd.Timestamp.now(tz="UTC")
            live_loop_last_error_at = error_at.isoformat()
            live_loop_last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            if live_loop_phase == "fetching_bybit":
                bybit_rest_last_error = live_loop_last_error
            _set_live_loop_phase("failed")
            with live_lock: live_state.update({"status":"degraded","error":str(exc),"updated_at":pd.Timestamp.now(tz="UTC").isoformat()})
        finally:
            # Release transient pandas/NumPy object graphs before the next
            # cycle.  The bounded query above is the primary guard; explicit
            # collection prevents allocator retention from accumulating across
            # repeated order-flow transforms.
            _release_transient_memory()
        time.sleep(_live_poll_interval_seconds())


def start_live_loop():
    global live_thread_started, live_thread, collector_thread, _live_lock_handle
    global live_loop_watchdog_thread, live_loop_watchdog_alerted
    global live_loop_started_at, live_loop_started_monotonic
    global live_loop_last_attempt_at, live_loop_last_completed_at
    global live_loop_last_error_at, live_loop_last_error
    global live_loop_last_attempt_monotonic, live_loop_last_completed_monotonic
    global live_loop_phase, live_loop_phase_at
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
            live_loop_started_monotonic = time.monotonic()
            live_loop_started_at = pd.Timestamp.now(tz="UTC").isoformat()
            live_loop_last_attempt_at = None
            live_loop_last_completed_at = None
            live_loop_last_error_at = None
            live_loop_last_error = None
            live_loop_last_attempt_monotonic = None
            live_loop_last_completed_monotonic = None
            live_loop_watchdog_alerted = False
            live_loop_phase = "starting_collectors"
            live_loop_phase_at = pd.Timestamp.now(tz="UTC").isoformat()
            collector_thread = start_collectors(trade_store)
            live_thread = threading.Thread(target=_live_loop, name="live-predictor", daemon=True)
            live_thread.start()
            live_thread_started = True
            live_loop_watchdog_thread = threading.Thread(
                target=_live_loop_watchdog,
                name="live-predictor-watchdog",
                daemon=True,
            )
            live_loop_watchdog_thread.start()
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




@app.get("/healthz")
def healthz():
    """Render liveness probe: no network, disk, database, or application locks."""
    return jsonify({
        "status": "ok",
        "service": "btc-structure-flow-predictor",
        "process_alive": True,
    })


@app.get("/health")
def health():
    # Detailed diagnostics intentionally remain separate from /healthz.  This
    # endpoint may inspect stores and collectors, but is never used by Render
    # to decide whether the process is alive.
    with live_lock: state = dict(live_state)
    collectors = trade_store.collector_status(pd.Timestamp.now(tz="UTC"), COLLECTOR_STALE_SECONDS)
    store_stats = trade_store.stats()
    # `collectors` describes WebSocket health only. Durable REST-backfilled
    # timestamps remain in `trade_store`; never merge them into the collector
    # object or a silent socket will look healthy.
    stale_exchanges = list(state.get("stale_exchanges") or [])
    last_automatic_delivery = _latest_delivery_summary("automatic")
    last_test_delivery = _latest_delivery_summary("test")
    last_notification_decision = _latest_push_decision()
    push_dispatch = _push_dispatch_diagnostics()
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
        "process_rss_mb": _process_rss_mb(),
        "memory_limit_mb": 512,
        "live_loop_owner":live_thread_started,
        "live_thread_alive":bool(live_thread and live_thread.is_alive()),
        "live_loop": _live_loop_diagnostics(),
        "data_health": state.get("data_health") or {},
        "proxy": state.get("proxy") or _proxy_diagnostics(),
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
            "dispatch": push_dispatch,
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
        state["prediction"]["probability_source"] = live_policy.PROBABILITY_SOURCE
        state["prediction"]["probability_use"] = live_policy.PROBABILITY_USE
        state["prediction"]["probability_tp_before_sl_is_heuristic"] = True
    last_automatic_delivery = _latest_delivery_summary("automatic")
    last_test_delivery = _latest_delivery_summary("test")
    last_notification_decision = _latest_push_decision()
    push_dispatch = _push_dispatch_diagnostics()
    push_counts = _subscription_counts()
    paper_status = state.get("paper") or paper_ledger._status()
    data_quality = state.get("data_quality")
    return jsonify({
        "paper_only": True,
        **state,
        "governance": state.get("governance") or _governance_payload(paper_status, data_quality),
        "live_loop": _live_loop_diagnostics(),
        "process_rss_mb": _process_rss_mb(),
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
            "dispatch": push_dispatch,
        },
    })


@app.get("/api/policy")
def api_policy():
    paper_status = paper_ledger._status()
    return jsonify(_governance_payload(paper_status, None))


@app.get("/api/funnel")
def api_funnel():
    return jsonify(funnel_diary.status())


@app.get("/api/shadow")
def api_shadow():
    return jsonify(shadow_book.status())


@app.get("/api/paper/economics")
def api_paper_economics():
    status = paper_ledger._status()
    return jsonify({
        "economics": status.get("economics"),
        "pnl_reporting": status.get("pnl_reporting"),
        "gross_pnl": status.get("gross_pnl"),
        "net_pnl": status.get("net_pnl"),
        "fees_paid": status.get("fees_paid"),
        "slippage_cost": status.get("slippage_cost"),
        "equity_gross": status.get("equity_gross"),
        "equity_net": status.get("equity_net"),
        "performance_closed_trades": status.get("performance_closed_trades"),
        "performance_excluded_trades": status.get("performance_excluded_trades"),
        "performance_excluded_superseded_churn": status.get("performance_excluded_superseded_churn"),
        "performance_net_pnl": status.get("performance_net_pnl"),
        "performance_gross_pnl": status.get("performance_gross_pnl"),
        "ledger_reset_id": status.get("ledger_reset_id"),
        "performance": status.get("performance"),
        "accounting": status.get("accounting"),
        "expectancy_r_net": status.get("expectancy_r_net"),
        "sum_r_gross": status.get("sum_r_gross"),
        "sum_r_net": status.get("sum_r_net"),
        "closed_trades": status.get("closed_trades"),
        "recent_closed": status.get("recent_closed"),
        "retune_discipline": status.get("retune_discipline"),
    })


@app.get("/api/paper/rescore")
def api_paper_rescore():
    return jsonify(live_policy.rescore_seeded_trades())


@app.get("/api/calibration")
def api_calibration():
    return jsonify(live_policy.calibration_status(flow_calibration_artifact, flow_gate_config))


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


if __name__ == "__main__":
    if os.getenv("START_LIVE_LOOP_ON_BOOT", "1").lower() in ("1", "true", "yes", "on"):
        start_live_boot_supervisor()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
