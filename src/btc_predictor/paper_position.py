from __future__ import annotations

import logging
from pathlib import Path
import threading

import pandas as pd

from btc_predictor.persistence import JsonStore

logger = logging.getLogger("btc_predictor.paper_position")


HISTORICAL_SEEDED_TRADES = [
    {
        "entry_time": "2026-07-23T15:15:21+00:00",
        "exit_time": "2026-07-24T00:15:00+00:00",
        "side": "short",
        "entry": 64729.40,
        "exit": 64999.90,
        "stop": 64999.90,
        "target": 63678.80,
        "size": 0.9242,
        "pnl": -249.99,
        "r_multiple": -1.0,
        "exit_reason": "stop",
        "zone": "untested_breakout:31cb58ceba1a16",
    },
    {
        "entry_time": "2026-07-24T01:12:36+00:00",
        "exit_time": "2026-07-24T14:20:00+00:00",
        "side": "short",
        "entry": 65032.30,
        "exit": 64657.90,
        "stop": 65233.34,
        "target": 64657.90,
        "size": 1.2436,
        "pnl": 465.61,
        "r_multiple": 1.86,
        "exit_reason": "target",
        "zone": "vwap_lower:fa2baad13bbb59",
    },
    {
        "entry_time": "2026-07-25T00:46:44+00:00",
        "exit_time": "2026-07-25T01:05:00+00:00",
        "side": "short",
        "entry": 64055.80,
        "exit": 63845.00,
        "stop": 64168.99,
        "target": 63845.00,
        "size": 2.2088,
        "pnl": 465.62,
        "r_multiple": 1.86,
        "exit_reason": "target",
        "zone": "untested_breakout:19ff79eebd224b",
    },
]


class PaperLedger:
    """Track hypothetical fills and P&L for emitted signals."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.store = JsonStore(self.path) if self.path else None
        self.lock = threading.RLock()
        self._open: dict | None = None
        self._closed: list[dict] = []
        self._equity = 100_000.0
        self._load()

    def _load(self):
        if self.store and self.path and self.path.exists():
            try:
                with self.lock:
                    data = self.store.read({})
                    self._open = data.get("open")
                    self._closed = list(data.get("closed", []))
                    self._equity = float(data.get("equity", 100_000.0))
            except Exception as exc:
                logger.warning("Failed to load paper ledger from %s: %s", self.path, exc)
        elif self.store and self.path and not self.path.exists():
            with self.lock:
                self._closed = list(HISTORICAL_SEEDED_TRADES)
                self._equity = round(100_000.0 + sum(t["pnl"] for t in self._closed), 2)
                self._save()

    def _save(self):
        if self.store:
            with self.lock:
                self.store.write({
                    "open": self._open,
                    "closed": self._closed[-5000:],
                    "equity": self._equity,
                })

    def update(self, prediction, current_ohlc: pd.DataFrame | None = None):
        with self.lock:
            last_close = None
            if current_ohlc is not None and not current_ohlc.empty:
                last_close = float(current_ohlc["close"].iloc[-1])

            # 1. Evaluate exits on existing open position
            if self._open is not None and current_ohlc is not None and not current_ohlc.empty:
                self._check_exit(current_ohlc)

            # 2. Check for signal flip or superseded position if still open
            if self._open is not None and prediction.bias != "neutral":
                current_side = "long" if prediction.bias == "bullish" else "short" if prediction.bias == "bearish" else None
                is_new_setup = (
                    prediction.entry is not None
                    and prediction.stop is not None
                    and prediction.target is not None
                    and prediction.position_size
                )
                zone_changed = prediction.zone and prediction.zone != self._open.get("zone")
                time_changed = (
                    prediction.timestamp
                    and pd.Timestamp(prediction.timestamp).isoformat() != self._open.get("entry_time")
                )
                if current_side != self._open["side"]:
                    self._close(prediction.timestamp, last_close, "signal_flipped")
                elif is_new_setup and (zone_changed or time_changed):
                    self._close(prediction.timestamp, last_close, "superseded_by_new_setup")

            # 3. Enter new position if no position is open and setup is confirmed
            if self._open is None and prediction.entry is not None and prediction.stop is not None and prediction.target is not None and prediction.position_size:
                self._open = {
                    "entry_time": pd.Timestamp(prediction.timestamp).isoformat(),
                    "side": "long" if prediction.bias == "bullish" else "short" if prediction.bias == "bearish" else "neutral",
                    "entry": float(prediction.entry),
                    "stop": float(prediction.stop),
                    "target": float(prediction.target),
                    "size": float(prediction.position_size),
                    "zone": prediction.zone,
                    "probability_tp_before_sl": prediction.probability_tp_before_sl,
                }

                # Check exit immediately on entry candle
                if current_ohlc is not None and not current_ohlc.empty:
                    self._check_exit(current_ohlc)

            self._save()
            return self._status()

    def _check_exit(self, ohlc: pd.DataFrame):
        if self._open is None:
            return
        side = self._open["side"]
        stop = self._open["stop"]
        target = self._open["target"]
        entry_time = pd.Timestamp(self._open["entry_time"])
        ohlc_index_utc = pd.to_datetime(ohlc.index, utc=True)
        future = ohlc.loc[ohlc_index_utc >= entry_time]

        if future.empty and not ohlc.empty and ohlc_index_utc[0] > entry_time:
            future = ohlc

        for ts, bar in future.iterrows():
            if side == "long":
                if float(bar.low) <= stop:
                    self._close(ts, stop, "stop")
                    return
                if float(bar.high) >= target:
                    self._close(ts, target, "target")
                    return
            else:
                if float(bar.high) >= stop:
                    self._close(ts, stop, "stop")
                    return
                if float(bar.low) <= target:
                    self._close(ts, target, "target")
                    return

    def _close(self, exit_time, exit_price, reason):
        if self._open is None:
            return
        entry = self._open["entry"]
        side = self._open["side"]
        size = self._open["size"]
        if exit_price is None:
            exit_price = entry
        else:
            exit_price = float(exit_price)
        if side == "long":
            pnl = (exit_price - entry) * size
            risk = entry - self._open["stop"]
            r = (exit_price - entry) / risk if risk else 0
        elif side == "short":
            pnl = (entry - exit_price) * size
            risk = self._open["stop"] - entry
            r = (entry - exit_price) / risk if risk else 0
        else:
            return
        self._equity += pnl
        trade = {
            "entry_time": self._open["entry_time"],
            "exit_time": pd.Timestamp(exit_time).isoformat(),
            "side": side,
            "entry": entry,
            "exit": float(exit_price),
            "stop": self._open["stop"],
            "target": self._open["target"],
            "size": size,
            "pnl": float(pnl),
            "r_multiple": float(r),
            "exit_reason": reason,
            "zone": self._open.get("zone"),
        }
        self._closed.append(trade)
        self._open = None

    def _status(self):
        with self.lock:
            closed = self._closed
            wins = sum(1 for t in closed if t["pnl"] > 0)
            losses = sum(1 for t in closed if t["pnl"] <= 0)
            gross_profit = sum(t["pnl"] for t in closed if t["pnl"] > 0)
            gross_loss = -sum(t["pnl"] for t in closed if t["pnl"] < 0)
            return {
                "equity": round(self._equity, 2),
                "open_position": self._open,
                "closed_trades": len(closed),
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / len(closed), 4) if closed else 0.0,
                "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
                "net_pnl": round(sum(t["pnl"] for t in closed), 2),
                "last_closed": closed[-1] if closed else None,
            }

    def close_all(self, exit_price, exit_time, reason="manual"):
        with self.lock:
            if self._open is not None:
                self._close(exit_time, exit_price, reason)
                self._save()
