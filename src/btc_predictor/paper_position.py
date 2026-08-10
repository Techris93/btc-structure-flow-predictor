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

    def __init__(self, path: str | Path | None = None, neutral_exit_observations: int = 3):
        self.path = Path(path) if path else None
        self.store = JsonStore(self.path) if self.path else None
        self.lock = threading.RLock()
        # Consecutive neutral predictions required before an open trade is
        # closed as "signal_neutralized" (mirrors lifecycle invalidation).
        self.neutral_exit_observations = max(1, int(neutral_exit_observations))
        self._open: dict | None = None
        self._pending: dict | None = None
        self._closed: list[dict] = []
        self._equity = 100_000.0
        self._load()

    def _load(self):
        if self.store and self.path and self.path.exists():
            try:
                with self.lock:
                    data = self.store.read({})
                    self._open = data.get("open")
                    self._pending = data.get("pending")
                    self._closed = list(data.get("closed", []))
                    self._equity = float(data.get("equity", 100_000.0))
            except Exception as exc:
                logger.warning("Failed to load paper ledger from %s: %s", self.path, exc)
        with self.lock:
            if not self._closed and self.store and self.path:
                self._closed = list(HISTORICAL_SEEDED_TRADES)
                self._equity = round(100_000.0 + sum(t["pnl"] for t in self._closed), 2)
                if self._open and any(t["entry_time"] == self._open.get("entry_time") for t in HISTORICAL_SEEDED_TRADES):
                    self._open = None
                self._save()

    def _save(self):
        if self.store:
            with self.lock:
                self.store.write({
                    "open": self._open,
                    "pending": self._pending,
                    "closed": self._closed[-5000:],
                    "equity": self._equity,
                })

    def update_market(self, current_ohlc: pd.DataFrame | None = None):
        """Apply market facts only: pending fills/expiry and TP/SL exits."""
        with self.lock:
            closed_before_update = len(self._closed)
            last_close = None
            if current_ohlc is not None and not current_ohlc.empty:
                last_close = float(current_ohlc["close"].iloc[-1])
            self._update_pending_market(current_ohlc)
            if self._open is not None and current_ohlc is not None and not current_ohlc.empty:
                self._check_exit(current_ohlc)
            self._save()
            status = self._status(last_close)
            status["newly_closed"] = [dict(trade) for trade in self._closed[closed_before_update:]]
            return status

    def apply_lifecycle(self, events, current_ohlc: pd.DataFrame | None = None):
        """Execute only durable lifecycle decisions, never raw predictions."""
        with self.lock:
            closed_before_update = len(self._closed)
            last_close = None
            if current_ohlc is not None and not current_ohlc.empty:
                last_close = float(current_ohlc["close"].iloc[-1])
            for event in events or []:
                event_type = str(event.get("event_type") or "")
                signal_id = event.get("signal_id")
                if event_type == "setup_confirmed":
                    existing = self._open or self._pending
                    if existing and existing.get("signal_id") == signal_id:
                        continue
                    snapshot = event.get("snapshot") or {}
                    expected_side = "long" if snapshot.get("bias") == "bullish" else "short"
                    if (
                        existing
                        and existing.get("signal_id") is None
                        and existing.get("side") == expected_side
                        and existing.get("zone") == snapshot.get("zone")
                    ):
                        # One-time migration for a position created before the
                        # lifecycle became execution authority. Adopt it without
                        # fabricating a close/reopen on deployment.
                        existing.update({
                            "signal_id": signal_id,
                            "lifecycle_event_id": event.get("event_id"),
                            "confirmed_at": event.get("created_at"),
                            "decision_time": snapshot.get("timestamp"),
                            "reclaim_time": snapshot.get("reclaim_time"),
                            "setup_atr": snapshot.get("setup_atr"),
                        })
                        continue
                    if self._open is not None:
                        self._close(event.get("created_at"), last_close, "superseded_by_confirmed_setup")
                    if self._pending is not None:
                        self._pending = None
                    self._place_confirmed(event)
                elif event_type in ("setup_invalidated", "setup_expired"):
                    reason = str(event.get("reason") or event_type)
                    if self._open is not None and self._matches_signal(self._open, signal_id):
                        self._close(event.get("created_at"), last_close, reason)
                    if self._pending is not None and self._matches_signal(self._pending, signal_id):
                        self._pending = None
            self._save()
            status = self._status(last_close)
            status["newly_closed"] = [dict(trade) for trade in self._closed[closed_before_update:]]
            return status

    @staticmethod
    def _matches_signal(position, signal_id):
        # Legacy persisted positions have no signal_id. Allow their first
        # lifecycle terminal event to resolve them instead of leaving limbo.
        return position.get("signal_id") in (None, signal_id)

    def bind_active_signal(self, signal_id):
        """Persist lifecycle identity onto a legacy open or pending order."""
        if not signal_id:
            return False
        with self.lock:
            position = self._open or self._pending
            if position is None:
                return False
            if position.get("signal_id") not in (None, signal_id):
                return False
            position["signal_id"] = signal_id
            self._save()
            return True

    def _place_confirmed(self, event):
        snapshot = dict(event.get("snapshot") or {})
        required = (snapshot.get("entry"), snapshot.get("stop"), snapshot.get("target"), snapshot.get("position_size"))
        if not all(value is not None for value in required) or float(snapshot.get("position_size") or 0) <= 0:
            logger.error("Refusing malformed setup_confirmed event %s", event.get("event_id"))
            return
        order = {
            "signal_id": event.get("signal_id"),
            "lifecycle_event_id": event.get("event_id"),
            "confirmed_at": event.get("created_at"),
            "decision_time": snapshot.get("timestamp"),
            "entry_time": snapshot.get("timestamp"),
            "side": "long" if snapshot.get("bias") == "bullish" else "short",
            "entry": float(snapshot["entry"]),
            "stop": float(snapshot["stop"]),
            "target": float(snapshot["target"]),
            "size": float(snapshot["position_size"]),
            "zone": snapshot.get("zone"),
            "sweep_time": snapshot.get("sweep_time"),
            "reclaim_time": snapshot.get("reclaim_time"),
            "setup_atr": snapshot.get("setup_atr"),
            "probability_tp_before_sl": snapshot.get("probability_tp_before_sl"),
            "entry_type": snapshot.get("entry_type", "market"),
        }
        if snapshot.get("entry_type", "market") == "limit":
            order["signal_time"] = order["entry_time"]
            order["pending_ttl_bars"] = 240
            self._pending = order
        else:
            # A close-time signal cannot be filled at that same close without
            # lookahead. Queue it for the first open strictly after the
            # immutable decision bar.
            order["entry_type"] = "market_next_open"
            order["signal_time"] = order["entry_time"]
            order["planned_entry"] = order["entry"]
            order["planned_risk_notional"] = abs(order["entry"]-order["stop"])*order["size"]
            self._pending = order

    def update(self, prediction, current_ohlc: pd.DataFrame | None = None):
        """Compatibility helper for isolated research/tests.

        It can seed one position but deliberately cannot replace or invalidate
        an existing trade from a raw predictor output. Production uses
        update_market() + apply_lifecycle().
        """
        market = self.update_market(current_ohlc)
        if self._open is None and self._pending is None and prediction.entry is not None and prediction.stop is not None and prediction.target is not None and prediction.position_size:
            snapshot = {
                key: getattr(prediction, key, None)
                for key in ("timestamp", "bias", "entry", "stop", "target", "position_size", "zone", "sweep_time", "reclaim_time", "setup_atr", "probability_tp_before_sl", "entry_type")
            }
            snapshot["timestamp"] = pd.Timestamp(snapshot["timestamp"]).isoformat()
            event = {"event_id": "compatibility-entry", "event_type": "setup_confirmed", "signal_id": None, "created_at": snapshot["timestamp"], "snapshot": snapshot}
            decision = self.apply_lifecycle([event], current_ohlc)
            decision["newly_closed"] = list(market.get("newly_closed") or []) + list(decision.get("newly_closed") or [])
            return decision
        return market

    def _update_pending_market(self, current_ohlc: pd.DataFrame | None):
        """Fill or expire a confirmed pending retracement limit order."""
        if self._pending is None:
            return
        pending = self._pending
        side = pending["side"]
        if current_ohlc is None or current_ohlc.empty:
            return
        ohlc_index_utc = pd.to_datetime(current_ohlc.index, utc=True)
        signal_time = pd.Timestamp(pending.get("signal_time", pending["entry_time"]))
        future = current_ohlc.loc[ohlc_index_utc > signal_time]
        limit = pending["entry"]
        ttl = int(pending.get("pending_ttl_bars", 240))

        if pending.get("entry_type") == "market_next_open":
            if future.empty:
                return
            ts,bar=next(iter(future.iterrows()))
            fill=float(bar.open)
            risk=(fill-pending["stop"]) if side=="long" else (pending["stop"]-fill)
            reward=(pending["target"]-fill) if side=="long" else (fill-pending["target"])
            if risk<=0 or reward<=0:
                logger.info(
                    "Pending %s market signal invalidated by next-open gap: fill=%.2f stop=%.2f target=%.2f",
                    side,fill,pending["stop"],pending["target"],
                )
                self._pending=None
                return
            planned_risk=float(pending.get("planned_risk_notional") or 0.0)
            size=planned_risk/risk if planned_risk>0 else pending["size"]
            self._open=dict(
                pending,
                entry=fill,
                size=float(size),
                entry_time=pd.Timestamp(ts).isoformat(),
                filled_at=pd.Timestamp(ts).isoformat(),
            )
            self._pending=None
            logger.info(
                "Pending %s market signal filled at next open %.2f (%s)",
                side,fill,self._open["entry_time"],
            )
            return

        for i, (ts, bar) in enumerate(future.iterrows()):
            if i >= ttl:
                logger.info("Pending %s order at %.2f expired after %d bars without fill", side, limit, ttl)
                self._pending = None
                return
            touched = float(bar.low) <= limit if side == "long" else float(bar.high) >= limit
            if touched:
                # Conservative same-bar rule: if the stop was hit on the fill
                # bar, the retracement would not have been tradable.
                stopped = float(bar.low) <= pending["stop"] if side == "long" else float(bar.high) >= pending["stop"]
                if stopped:
                    logger.info("Pending %s order at %.2f invalidated: stop hit before fill", side, limit)
                    self._pending = None
                    return
                self._open = dict(pending, entry_time=pd.Timestamp(ts).isoformat())
                self._pending = None
                logger.info("Pending %s order filled at %.2f (%s)", side, limit, self._open["entry_time"])
                return

    def _check_exit(self, ohlc: pd.DataFrame):
        if self._open is None:
            return
        side = self._open["side"]
        stop = self._open["stop"]
        target = self._open["target"]
        entry_time = pd.Timestamp(self._open["entry_time"])
        ohlc_index_utc = pd.to_datetime(ohlc.index, utc=True)
        # A market signal is known only after its decision candle closes. Do
        # not let that already-completed candle retrospectively hit TP or SL.
        include_entry_bar=self._open.get("entry_type")=="market_next_open"
        future = ohlc.loc[ohlc_index_utc >= entry_time] if include_entry_bar else ohlc.loc[ohlc_index_utc > entry_time]

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
            "signal_id": self._open.get("signal_id"),
            "lifecycle_event_id": self._open.get("lifecycle_event_id"),
            "confirmed_at": self._open.get("confirmed_at"),
            "decision_time": self._open.get("decision_time"),
            "sweep_time": self._open.get("sweep_time"),
            "reclaim_time": self._open.get("reclaim_time"),
            "setup_atr": self._open.get("setup_atr"),
        }
        self._closed.append(trade)
        self._open = None

    def _status(self, last_price: float | None = None):
        with self.lock:
            closed = self._closed
            wins = sum(1 for t in closed if t["pnl"] > 0)
            losses = sum(1 for t in closed if t["pnl"] <= 0)
            gross_profit = sum(t["pnl"] for t in closed if t["pnl"] > 0)
            gross_loss = -sum(t["pnl"] for t in closed if t["pnl"] < 0)
            unrealized = None
            if self._open is not None and last_price is not None:
                if self._open["side"] == "long":
                    unrealized = (float(last_price) - self._open["entry"]) * self._open["size"]
                elif self._open["side"] == "short":
                    unrealized = (self._open["entry"] - float(last_price)) * self._open["size"]
                unrealized = round(unrealized, 2)
            return {
                "equity": round(self._equity, 2),
                "open_position": self._open,
                "pending_order": self._pending,
                "open_unrealized_pnl": unrealized,
                "mark_to_market_equity": round(self._equity + unrealized, 2) if unrealized is not None else round(self._equity, 2),
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
            self._pending = None
            if self._open is not None:
                self._close(exit_time, exit_price, reason)
            self._save()
