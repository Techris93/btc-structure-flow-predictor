from __future__ import annotations

import logging
from pathlib import Path
import threading

import pandas as pd

from btc_predictor.persistence import JsonStore
from btc_predictor import live_policy

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
        "zone_kind": "untested_breakout",
        "source": "seeded",
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
        "zone_kind": "vwap_lower",
        "source": "seeded",
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
        "zone_kind": "untested_breakout",
        "source": "seeded",
    },
]


def _annotate_economics(trade: dict) -> dict:
    """Attach gross/net cost fields without inventing missing prices."""
    row = dict(trade)
    if row.get("entry") is None or row.get("exit") is None or row.get("size") is None:
        return row
    scored = live_policy.rescore_closed_trade(row)
    row.setdefault("source", row.get("source") or "live")
    row["gross_pnl"] = scored["gross_pnl"]
    row["fees"] = scored["fees"]
    row["slippage_cost"] = scored["slippage_cost"]
    row["net_pnl"] = scored["net_pnl"]
    row["r_multiple_gross"] = scored["r_multiple_gross"]
    row["r_multiple_net"] = scored["r_multiple_net"]
    row["planned_rr"] = scored.get("planned_rr")
    row["stop_distance"] = scored.get("stop_distance")
    row["stop_distance_pct"] = scored.get("stop_distance_pct")
    row["stop_on_round_magnet"] = scored.get("stop_on_round_magnet")
    row["stop_magnets"] = scored.get("stop_magnets")
    row["zone_kind"] = scored.get("zone_kind") or row.get("zone_kind")
    row["hold_hours"] = scored.get("hold_hours")
    row["fee_bps"] = scored["fee_bps"]
    row["slippage_bps"] = scored["slippage_bps"]
    # Canonical pnl/r for equity: net after research costs.
    row["pnl_gross"] = scored["gross_pnl"]
    row["pnl"] = scored["net_pnl"]
    row["r_multiple"] = scored["r_multiple_net"]
    return row


class PaperLedger:
    """Track hypothetical fills and P&L for emitted signals."""

    def __init__(
        self,
        path: str | Path | None = None,
        neutral_exit_observations: int = 3,
        fee_bps: float = live_policy.RESEARCH_FEE_BPS,
        slippage_bps: float = live_policy.RESEARCH_SLIPPAGE_BPS,
        max_notional_multiple: float = live_policy.MAX_NOTIONAL_MULTIPLE,
        daily_loss_r: float = live_policy.DAILY_LOSS_R,
        weekly_loss_r: float = live_policy.WEEKLY_LOSS_R,
        risk_fraction: float = live_policy.RISK_FRACTION,
        soft_filters: bool = live_policy.SOFT_FILTERS_ENABLED,
        apply_research_costs: bool = True,
        use_fixed_pct_exits: bool = live_policy.USE_FIXED_PCT_EXITS,
        max_hold_hours: float = live_policy.MAX_HOLD_HOURS,
        fill_min_rr: float = live_policy.FILL_MIN_RR,
    ):
        self.path = Path(path) if path else None
        self.store = JsonStore(self.path) if self.path else None
        self.lock = threading.RLock()
        # Consecutive neutral predictions required before an open trade is
        # closed as "signal_neutralized" (mirrors lifecycle invalidation).
        self.neutral_exit_observations = max(1, int(neutral_exit_observations))
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.max_notional_multiple = float(max_notional_multiple)
        self.daily_loss_r = float(daily_loss_r)
        self.weekly_loss_r = float(weekly_loss_r)
        self.risk_fraction = float(risk_fraction)
        self.soft_filters = bool(soft_filters)
        self.apply_research_costs = bool(apply_research_costs)
        self.use_fixed_pct_exits = bool(use_fixed_pct_exits)
        self.max_hold_hours = float(max_hold_hours)
        self.fill_min_rr = float(fill_min_rr)
        self._open: dict | None = None
        self._pending: dict | None = None
        self._closed: list[dict] = []
        self._equity = live_policy.RESEARCH_INITIAL_EQUITY
        self._equity_gross = live_policy.RESEARCH_INITIAL_EQUITY
        self._last_reject: dict | None = None
        self._load()

    def _load(self):
        if self.store and self.path and self.path.exists():
            try:
                with self.lock:
                    data = self.store.read({})
                    self._open = data.get("open")
                    self._pending = data.get("pending")
                    self._closed = [_annotate_economics(t) for t in list(data.get("closed", []))]
                    # Prefer net equity; fall back to recomputing from annotated trades.
                    if data.get("equity_net") is not None:
                        self._equity = float(data["equity_net"])
                    elif data.get("equity") is not None and data.get("economics_version"):
                        self._equity = float(data["equity"])
                    else:
                        self._equity = round(
                            live_policy.RESEARCH_INITIAL_EQUITY
                            + sum(float(t.get("net_pnl", t.get("pnl", 0))) for t in self._closed),
                            2,
                        )
                    self._equity_gross = round(
                        live_policy.RESEARCH_INITIAL_EQUITY
                        + sum(float(t.get("gross_pnl", t.get("pnl_gross", t.get("pnl", 0)))) for t in self._closed),
                        2,
                    )
            except Exception as exc:
                logger.warning("Failed to load paper ledger from %s: %s", self.path, exc)
        with self.lock:
            if not self._closed and self.store and self.path:
                self._closed = [_annotate_economics(dict(t)) for t in HISTORICAL_SEEDED_TRADES]
                self._equity = round(
                    live_policy.RESEARCH_INITIAL_EQUITY
                    + sum(float(t["net_pnl"]) for t in self._closed),
                    2,
                )
                self._equity_gross = round(
                    live_policy.RESEARCH_INITIAL_EQUITY
                    + sum(float(t["gross_pnl"]) for t in self._closed),
                    2,
                )
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
                    "equity_net": self._equity,
                    "equity_gross": self._equity_gross,
                    "economics_version": 1,
                    "fee_bps": self.fee_bps,
                    "slippage_bps": self.slippage_bps,
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
                        self._attach_decision_fields(existing, snapshot, event)
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

    def _attach_decision_fields(self, order: dict, snapshot: dict, event: dict | None = None):
        enriched = live_policy.enrich_decision_snapshot(snapshot)
        order["decision_snapshot"] = enriched
        order["zone_kind"] = enriched.get("zone_kind")
        order["sweep_depth_atr"] = enriched.get("sweep_depth_atr")
        order["regime_4h"] = enriched.get("regime_4h")
        order["regime_1h"] = enriched.get("regime_1h")
        order["setup_15m"] = enriched.get("setup_15m")
        order["market_flow_score"] = enriched.get("market_flow_score")
        order["raw_footprint_score"] = enriched.get("raw_footprint_score")
        order["flow_gate_mode"] = enriched.get("flow_gate_mode")
        order["planned_rr"] = enriched.get("planned_rr") or enriched.get("reward_risk")
        order["reward_risk"] = enriched.get("reward_risk")
        order["setup_type"] = enriched.get("setup_type") or snapshot.get("setup_type") or "reversal"
        order["stop_distance"] = enriched.get("stop_distance")
        order["stop_distance_pct"] = enriched.get("stop_distance_pct")
        order["stop_on_round_magnet"] = enriched.get("stop_on_round_magnet")
        order["stop_magnets"] = enriched.get("stop_magnets")
        order["decision_bar"] = enriched.get("decision_bar") or enriched.get("timestamp")
        order["entry_type"] = enriched.get("entry_type") or order.get("entry_type")
        order["probability_tp_before_sl"] = enriched.get("probability_tp_before_sl")
        order["probability_source"] = live_policy.PROBABILITY_SOURCE
        order["probability_use"] = live_policy.PROBABILITY_USE
        order["lifecycle_event_id"] = (event or {}).get("event_id") or order.get("lifecycle_event_id")
        order["lifecycle_event_type"] = (event or {}).get("event_type") or order.get("lifecycle_event_type")
        return order

    def _place_confirmed(self, event):
        snapshot = live_policy.enrich_decision_snapshot(dict(event.get("snapshot") or {}))
        required = (snapshot.get("entry"), snapshot.get("stop"), snapshot.get("target"), snapshot.get("position_size"))
        if not all(value is not None for value in required) or float(snapshot.get("position_size") or 0) <= 0:
            logger.error("Refusing malformed setup_confirmed event %s", event.get("event_id"))
            self._last_reject = {"reason": "malformed_setup", "event_id": event.get("event_id")}
            return

        last_closed = self._closed[-1] if self._closed else None
        soft = live_policy.evaluate_soft_filters(
            snapshot, last_closed=last_closed, enabled=self.soft_filters
        )
        snapshot = soft.get("snapshot") or snapshot
        if not soft.get("allow", True):
            logger.info(
                "Paper soft-filter skip signal=%s reasons=%s warnings=%s",
                event.get("signal_id"),
                soft.get("hard_skips"),
                soft.get("warnings"),
            )
            self._last_reject = {
                "reason": "soft_filter",
                "hard_skips": soft.get("hard_skips"),
                "warnings": soft.get("warnings"),
                "signal_id": event.get("signal_id"),
                "validated": False,
            }
            return

        risk = live_policy.apply_risk_caps(
            float(snapshot["entry"]),
            float(snapshot["stop"]),
            float(snapshot["position_size"]),
            self._equity,
            closed=self._closed,
            has_open_or_pending=self._open is not None or self._pending is not None,
            now=event.get("created_at") or snapshot.get("timestamp"),
            max_notional_multiple=self.max_notional_multiple,
            daily_loss_r=self.daily_loss_r,
            weekly_loss_r=self.weekly_loss_r,
            risk_fraction=self.risk_fraction,
        )
        if not risk.get("allow"):
            logger.info(
                "Paper risk-cap skip signal=%s reasons=%s",
                event.get("signal_id"),
                risk.get("reasons"),
            )
            self._last_reject = {
                "reason": "risk_cap",
                "risk": risk,
                "signal_id": event.get("signal_id"),
            }
            return

        size = float(risk["size"])
        order = {
            "signal_id": event.get("signal_id"),
            "lifecycle_event_id": event.get("event_id"),
            "lifecycle_event_type": event.get("event_type"),
            "confirmed_at": event.get("created_at"),
            "decision_time": snapshot.get("timestamp"),
            "entry_time": snapshot.get("timestamp"),
            "side": "long" if snapshot.get("bias") == "bullish" else "short",
            "entry": float(snapshot["entry"]),
            "stop": float(snapshot["stop"]),
            "target": float(snapshot["target"]),
            "size": size,
            "zone": snapshot.get("zone"),
            "sweep_time": snapshot.get("sweep_time"),
            "reclaim_time": snapshot.get("reclaim_time"),
            "setup_atr": snapshot.get("setup_atr"),
            "probability_tp_before_sl": snapshot.get("probability_tp_before_sl"),
            "entry_type": snapshot.get("entry_type", "market"),
            "source": "live",
            "soft_filter_warnings": soft.get("warnings") or [],
            "risk_cap_notes": risk.get("reasons") or [],
            "notional_at_signal": risk.get("notional"),
            "risk_cash": risk.get("risk_cash"),
        }
        self._attach_decision_fields(order, snapshot, event)
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
            order["planned_risk_notional"] = abs(order["entry"] - order["stop"]) * order["size"]
            self._pending = order
        self._last_reject = None

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
                for key in (
                    "timestamp", "bias", "entry", "stop", "target", "position_size", "zone",
                    "zone_kind", "sweep_time", "reclaim_time", "setup_atr",
                    "probability_tp_before_sl", "entry_type", "reward_risk",
                    "regime_4h", "regime_1h", "setup_15m", "sweep_depth_atr",
                    "market_flow_score", "raw_footprint_score", "flow_gate_mode",
                    "orderflow_score", "orderflow_confirmation", "setup_type",
                    "sweep_status", "orderflow_reason",
                )
            }
            snapshot["timestamp"] = pd.Timestamp(snapshot["timestamp"]).isoformat()
            event = {
                "event_id": "compatibility-entry",
                "event_type": "setup_confirmed",
                "signal_id": None,
                "created_at": snapshot["timestamp"],
                "snapshot": snapshot,
            }
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
            ts, bar = next(iter(future.iterrows()))
            fill = float(bar.open)
            stop, target = pending["stop"], pending["target"]
            if self.use_fixed_pct_exits:
                geo = live_policy.fixed_pct_exits(fill, side)
                stop, target = geo["stop"], geo["target"]
            risk = (fill - stop) if side == "long" else (stop - fill)
            reward = (target - fill) if side == "long" else (fill - target)
            if risk <= 0 or reward <= 0 or not live_policy.fill_min_rr_ok(fill, stop, target, side, self.fill_min_rr):
                logger.info(
                    "Pending %s market signal invalidated at next open: fill=%.2f stop=%.2f target=%.2f",
                    side, fill, stop, target,
                )
                self._pending = None
                self._last_reject = {"reason": "fill_rr_below_minimum", "fill": fill, "stop": stop, "target": target}
                return
            planned_risk = float(pending.get("planned_risk_notional") or 0.0)
            size = planned_risk / risk if planned_risk > 0 else pending["size"]
            notional = abs(fill * size)
            max_notional = self._equity * self.max_notional_multiple
            if notional > max_notional and fill > 0:
                size = max_notional / fill
            self._open = dict(
                pending,
                entry=fill,
                stop=float(stop),
                target=float(target),
                size=float(size),
                entry_time=pd.Timestamp(ts).isoformat(),
                filled_at=pd.Timestamp(ts).isoformat(),
                planned_rr=(target - fill) / risk if side == "long" else (fill - target) / risk,
            )
            self._pending = None
            logger.info(
                "Pending %s market signal filled at next open %.2f sl=%.2f tp=%.2f (%s)",
                side, fill, stop, target, self._open["entry_time"],
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
                filled = dict(pending, entry_time=pd.Timestamp(ts).isoformat(), filled_at=pd.Timestamp(ts).isoformat())
                if self.use_fixed_pct_exits:
                    geo = live_policy.fixed_pct_exits(limit, side)
                    if not live_policy.fill_min_rr_ok(limit, geo["stop"], geo["target"], side, self.fill_min_rr):
                        logger.info("Pending %s limit invalidated: fill RR below minimum", side)
                        self._pending = None
                        return
                    filled["stop"] = geo["stop"]
                    filled["target"] = geo["target"]
                    filled["planned_rr"] = geo["reward_risk"]
                self._open = filled
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
        include_entry_bar = self._open.get("entry_type") == "market_next_open"
        future = ohlc.loc[ohlc_index_utc >= entry_time] if include_entry_bar else ohlc.loc[ohlc_index_utc > entry_time]

        if future.empty and not ohlc.empty and ohlc_index_utc[0] > entry_time:
            future = ohlc

        max_hold = pd.Timedelta(hours=self.max_hold_hours) if self.max_hold_hours > 0 else None
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
            if max_hold is not None and (pd.Timestamp(ts) - entry_time) >= max_hold:
                self._close(ts, float(bar.close), "max_hold")
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

        risk = abs(entry - self._open["stop"])
        if side == "long":
            gross = (exit_price - entry) * size
            gross_r = (exit_price - entry) / risk if risk else 0.0
        elif side == "short":
            gross = (entry - exit_price) * size
            gross_r = (entry - exit_price) / risk if risk else 0.0
        else:
            return

        if self.apply_research_costs:
            costs = live_policy.trade_costs(
                side, entry, exit_price, size,
                fee_bps=self.fee_bps, slippage_bps=self.slippage_bps,
            )
            net = costs["net_pnl"]
            fees = costs["fees"]
            slip = costs["slippage_cost"]
            net_r = net / (risk * size) if risk and size else 0.0
        else:
            net = gross
            fees = 0.0
            slip = 0.0
            net_r = gross_r

        self._equity += net
        self._equity_gross += gross
        trade = {
            "entry_time": self._open["entry_time"],
            "exit_time": pd.Timestamp(exit_time).isoformat(),
            "side": side,
            "entry": entry,
            "exit": float(exit_price),
            "stop": self._open["stop"],
            "target": self._open["target"],
            "size": size,
            "gross_pnl": float(gross),
            "pnl_gross": float(gross),
            "fees": float(fees),
            "slippage_cost": float(slip),
            "net_pnl": float(net),
            "pnl": float(net),
            "r_multiple_gross": float(gross_r),
            "r_multiple_net": float(net_r),
            "r_multiple": float(net_r),
            "exit_reason": reason,
            "zone": self._open.get("zone"),
            "zone_kind": self._open.get("zone_kind"),
            "setup_type": self._open.get("setup_type", "reversal"),
            "signal_id": self._open.get("signal_id"),
            "lifecycle_event_id": self._open.get("lifecycle_event_id"),
            "lifecycle_event_type": self._open.get("lifecycle_event_type"),
            "confirmed_at": self._open.get("confirmed_at"),
            "decision_time": self._open.get("decision_time"),
            "decision_bar": self._open.get("decision_bar"),
            "decision_snapshot": self._open.get("decision_snapshot"),
            "sweep_time": self._open.get("sweep_time"),
            "reclaim_time": self._open.get("reclaim_time"),
            "setup_atr": self._open.get("setup_atr"),
            "sweep_depth_atr": self._open.get("sweep_depth_atr"),
            "regime_4h": self._open.get("regime_4h"),
            "regime_1h": self._open.get("regime_1h"),
            "setup_15m": self._open.get("setup_15m"),
            "market_flow_score": self._open.get("market_flow_score"),
            "raw_footprint_score": self._open.get("raw_footprint_score"),
            "flow_gate_mode": self._open.get("flow_gate_mode"),
            "planned_rr": self._open.get("planned_rr") or self._open.get("reward_risk"),
            "stop_distance": self._open.get("stop_distance"),
            "stop_distance_pct": self._open.get("stop_distance_pct"),
            "stop_on_round_magnet": self._open.get("stop_on_round_magnet"),
            "stop_magnets": self._open.get("stop_magnets"),
            "entry_type": self._open.get("entry_type"),
            "probability_tp_before_sl": self._open.get("probability_tp_before_sl"),
            "probability_source": self._open.get("probability_source") or live_policy.PROBABILITY_SOURCE,
            "probability_use": self._open.get("probability_use") or live_policy.PROBABILITY_USE,
            "fee_bps": self.fee_bps,
            "slippage_bps": self.slippage_bps,
            "source": self._open.get("source") or "live",
        }
        if trade.get("entry_time") and trade.get("exit_time"):
            trade["hold_hours"] = (
                pd.Timestamp(trade["exit_time"]) - pd.Timestamp(trade["entry_time"])
            ).total_seconds() / 3600.0
        self._closed.append(trade)
        self._open = None

    def _status(self, last_price: float | None = None):
        with self.lock:
            closed = self._closed
            wins = sum(1 for t in closed if float(t.get("net_pnl", t.get("pnl", 0))) > 0)
            losses = sum(1 for t in closed if float(t.get("net_pnl", t.get("pnl", 0))) <= 0)
            gross_profit = sum(float(t.get("net_pnl", t.get("pnl", 0))) for t in closed if float(t.get("net_pnl", t.get("pnl", 0))) > 0)
            gross_loss = -sum(float(t.get("net_pnl", t.get("pnl", 0))) for t in closed if float(t.get("net_pnl", t.get("pnl", 0))) < 0)
            sum_gross = sum(float(t.get("gross_pnl", t.get("pnl_gross", t.get("pnl", 0)))) for t in closed)
            sum_net = sum(float(t.get("net_pnl", t.get("pnl", 0))) for t in closed)
            sum_fees = sum(float(t.get("fees", 0) or 0) for t in closed)
            sum_slip = sum(float(t.get("slippage_cost", 0) or 0) for t in closed)
            sum_r_net = sum(float(t.get("r_multiple_net", t.get("r_multiple", 0)) or 0) for t in closed)
            sum_r_gross = sum(float(t.get("r_multiple_gross", t.get("r_multiple", 0)) or 0) for t in closed)
            setup_type_stats = {}
            for st in ("reversal", "continuation"):
                sub = [t for t in closed if str(t.get("setup_type") or "reversal").lower() == st]
                sub_wins = sum(1 for t in sub if float(t.get("net_pnl", t.get("pnl", 0))) > 0)
                sub_losses = sum(1 for t in sub if float(t.get("net_pnl", t.get("pnl", 0))) <= 0)
                setup_type_stats[st] = {
                    "trades": len(sub),
                    "wins": sub_wins,
                    "losses": sub_losses,
                    "win_rate": round(sub_wins / len(sub), 4) if sub else 0.0,
                    "net_pnl": round(sum(float(t.get("net_pnl", t.get("pnl", 0))) for t in sub), 2),
                }
            unrealized = None
            unrealized_gross = None
            if self._open is not None and last_price is not None:
                if self._open["side"] == "long":
                    unrealized_gross = (float(last_price) - self._open["entry"]) * self._open["size"]
                elif self._open["side"] == "short":
                    unrealized_gross = (self._open["entry"] - float(last_price)) * self._open["size"]
                unrealized_gross = round(unrealized_gross, 2)
                # Mark-to-market net estimate applies exit costs only.
                if self.apply_research_costs and unrealized_gross is not None:
                    costs = live_policy.trade_costs(
                        self._open["side"],
                        self._open["entry"],
                        float(last_price),
                        self._open["size"],
                        fee_bps=self.fee_bps,
                        slippage_bps=self.slippage_bps,
                    )
                    unrealized = round(costs["net_pnl"], 2)
                else:
                    unrealized = unrealized_gross
            economics = live_policy.research_economics()
            economics = {
                **economics,
                "fee_bps": self.fee_bps,
                "slippage_bps": self.slippage_bps,
            }
            retune = live_policy.retune_discipline_status(len(closed))
            return {
                "equity": round(self._equity, 2),
                "equity_net": round(self._equity, 2),
                "equity_gross": round(self._equity_gross, 2),
                "open_position": self._open,
                "pending_order": self._pending,
                "open_unrealized_pnl": unrealized,
                "open_unrealized_pnl_gross": unrealized_gross,
                "mark_to_market_equity": round(self._equity + unrealized, 2) if unrealized is not None else round(self._equity, 2),
                "mark_to_market_equity_gross": (
                    round(self._equity_gross + unrealized_gross, 2)
                    if unrealized_gross is not None
                    else round(self._equity_gross, 2)
                ),
                "closed_trades": len(closed),
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / len(closed), 4) if closed else 0.0,
                "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
                "profit_factor_net": round(gross_profit / gross_loss, 3) if gross_loss else None,
                "net_pnl": round(sum_net, 2),
                "gross_pnl": round(sum_gross, 2),
                "fees_paid": round(sum_fees, 2),
                "slippage_cost": round(sum_slip, 2),
                "sum_r_net": round(sum_r_net, 4),
                "sum_r_gross": round(sum_r_gross, 4),
                "expectancy_r_net": round(sum_r_net / len(closed), 4) if closed else None,
                "setup_type_stats": setup_type_stats,
                "last_closed": closed[-1] if closed else None,
                "last_reject": self._last_reject,
                "economics": economics,
                "pnl_reporting": {
                    "gross_pnl": round(sum_gross, 2),
                    "approx_net_pnl": round(sum_net, 2),
                    "do_not_treat_gross_as_alpha": True,
                    "label": "gross / approx net (research fee+slip)",
                },
                "risk_policy": {
                    "risk_fraction": self.risk_fraction,
                    "max_notional_multiple": self.max_notional_multiple,
                    "daily_loss_r": self.daily_loss_r,
                    "weekly_loss_r": self.weekly_loss_r,
                    "one_open_risk_unit": True,
                    "soft_filters": self.soft_filters,
                },
                "retune_discipline": retune,
                "recent_closed": closed[-10:],
            }

    def close_all(self, exit_price, exit_time, reason="manual"):
        with self.lock:
            self._pending = None
            if self._open is not None:
                self._close(exit_time, exit_price, reason)
            self._save()
