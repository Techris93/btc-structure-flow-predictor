#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -q --upgrade pip setuptools wheel

# Prefer already-installed packages; only install missing runtime deps.
python - <<'PY2'
import importlib, subprocess, sys
needed = [
    "flask",
    "gunicorn",
    "pandas",
    "numpy",
    "requests",
    "websockets",
    "pywebpush",
    "cryptography",
]
missing = []
for name in needed:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    print("Installing missing packages:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
else:
    print("Runtime packages already present")
PY2

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export MARKET_TYPE="${MARKET_TYPE:-linear}"
export BTC_DATA_DIR="${BTC_DATA_DIR:-$ROOT_DIR/work/runtime-local}"
export BINANCE_REST_ENABLED="${BINANCE_REST_ENABLED:-0}"
export BINANCE_REST_MINUTES="${BINANCE_REST_MINUTES:-15}"
export BINANCE_TRADE_LIMIT="${BINANCE_TRADE_LIMIT:-100}"
export BINANCE_FLOW_LIMIT="${BINANCE_FLOW_LIMIT:-60}"
export BINANCE_WS_RECONNECT_SECONDS="${BINANCE_WS_RECONNECT_SECONDS:-15}"
export BINANCE_WS_RECONNECT_MAX_SECONDS="${BINANCE_WS_RECONNECT_MAX_SECONDS:-120}"
export BYBIT_WS_RECONNECT_SECONDS="${BYBIT_WS_RECONNECT_SECONDS:-5}"
export LIVE_POLL_SECONDS="${LIVE_POLL_SECONDS:-45}"
export PORT="${PORT:-8000}"
export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR/src}"
export PYTHONUNBUFFERED=1

mkdir -p "$BTC_DATA_DIR"

echo "Starting BTC Structure Flow locally"
echo "  MARKET_TYPE=$MARKET_TYPE"
echo "  BINANCE_REST_ENABLED=$BINANCE_REST_ENABLED"
echo "  BTC_DATA_DIR=$BTC_DATA_DIR"
echo "  URL=http://127.0.0.1:$PORT"
echo
echo "Futures rate-limit posture:"
echo "  - Binance live trades: one long-lived fstream WebSocket"
echo "  - Binance REST: OFF by default (no fapi polling)"
echo "  - Bybit: linear REST for OHLCV + linear publicTrade WebSocket"
echo "  - Reconnect backoff avoids 300-connect/5min storms"

exec python -u -c 'import os; from app import app; app.run(host="127.0.0.1", port=int(os.environ.get("PORT","8000")), debug=False, use_reloader=False)'
