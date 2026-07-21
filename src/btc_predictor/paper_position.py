from __future__ import annotations

import json
from pathlib import Path
import threading

import pandas as pd


class PaperLedger:
    """Track hypothetical fills and P&L for emitted signals."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.lock = threading.RLock()
        self._open: dict | None = None
        self._closed: list[dict] = []
        self._equity = 100_000.0
        self._load()

    def _load(self):
        if self.path and self.path.exists():
            try:
                with self.lock:
                    data = json.loads(self.path.read_text())
                    self._open = data.get("open")
                    self._closed = list(data.get("closed", []))
                    self._equity = float(data.get("equity", 100_000.0))
            except Exception:
                pass

    def _save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock:
                self.path.write_text(json.dumps({
                    "open": self._open,
                    "closed": self._closed[-5000:],
                    "equity": self._equity,
                }, default=str, indent=2))

    def update(self, prediction, current_ohlc: pd.DataFrame | None = None):
        with self.lock:
            if self._open is not None and prediction.bias != "neutral":
                current_side = "long" if prediction.bias == "bullish" else "short" if prediction.bias == "bearish" else None
                if current_side != self._open["side"]:
                    self._close(prediction.timestamp, None, "signal_flipped")
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
            if self._open is not None and current_ohlc is not None and not current_ohlc.empty:
                self._check_exit(current_ohlc)
            self._save()
            return self._status()

    def _check_exit(self, ohlc: pd.DataFrame):
        side = self._open["side"]
        stop = self._open["stop"]
        target = self._open["target"]
        # Only evaluate bars at/after entry. Passing the full history would let
        # pre-entry lows/highs immediately stop out a brand-new paper position.
        entry_time = pd.Timestamp(self._open["entry_time"])
        future = ohlc.loc[pd.to_datetime(ohlc.index, utc=True) >= entry_time]
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
