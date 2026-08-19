from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from btc_predictor import live_policy


def _value(source: Any, name: str, default=None):
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _timestamp(value):
    if value in (None, ""):
        return None
    return pd.Timestamp(value).isoformat()


def _money(value):
    return f"${float(value):,.2f}"


def _human(value):
    return str(value or "state changed").replace("_", " ").strip().capitalize()


class SignalLifecycle:
    """Pure, durable notification lifecycle for predictor outputs.

    This is the single authority for setup activation, replacement and
    invalidation. Callers may use its emitted events for notifications and
    paper execution, keeping both state machines synchronized.
    """

    VERSION = 2

    def __init__(self, confirm_observations=2, invalidation_observations=3, bias_observations=2, replacement_distance_atr=.25):
        self.confirm_observations = max(1, int(confirm_observations))
        self.invalidation_observations = max(1, int(invalidation_observations))
        self.bias_observations = max(1, int(bias_observations))
        self.replacement_distance_atr = max(0.0, float(replacement_distance_atr))

    @staticmethod
    def initial_state():
        return {
            "version": SignalLifecycle.VERSION,
            "candidate": None,
            "active": None,
            "missing_observations": 0,
            "last_missing_decision_at": None,
            "stable_bias": None,
            "bias_candidate": None,
            "bias_observations": 0,
            "last_bias_decision_at": None,
            "event_sequence": 0,
            "retired_signal_ids": [],
            "updated_at": None,
        }

    @staticmethod
    def snapshot(prediction):
        fields = (
            "timestamp",
            "bias",
            "setup_type",
            "zone",
            "zone_kind",
            "sweep_status",
            "sweep_depth_atr",
            "sweep_time",
            "reclaim_time",
            "orderflow_confirmation",
            "orderflow_reason",
            "entry",
            "stop",
            "target",
            "reward_risk",
            "probability_tp_before_sl",
            "position_size",
            "no_trade_reason",
            "regime_4h",
            "regime_1h",
            "setup_15m",
            "orderflow_score",
            "market_flow_score",
            "market_flow_threshold",
            "market_flow_confirmed",
            "raw_footprint_score",
            "raw_footprint_threshold",
            "raw_footprint_confirmed",
            "raw_footprint_eligible",
            "orderflow_fresh_exchanges",
            "flow_gate_mode",
            "flow_state",
            "setup_atr",
            "entry_type",
        )
        snapshot = {field: _value(prediction, field) for field in fields}
        for field in ("timestamp", "sweep_time", "reclaim_time"):
            snapshot[field] = _timestamp(snapshot.get(field))
        # Geometry + heuristic-p metadata for durable decision logs.
        return live_policy.enrich_decision_snapshot(snapshot)

    @staticmethod
    def is_actionable(snapshot):
        return bool(
            snapshot.get("bias") in ("bullish", "bearish")
            and snapshot.get("setup_type")
            and snapshot.get("zone")
            and snapshot.get("sweep_status") == "confirmed"
            and snapshot.get("orderflow_confirmation") is True
            and snapshot.get("entry") is not None
            and snapshot.get("stop") is not None
            and snapshot.get("target") is not None
            and float(snapshot.get("position_size") or 0.0) > 0.0
            and not snapshot.get("no_trade_reason")
        )

    @staticmethod
    def signal_id(snapshot):
        if not SignalLifecycle.is_actionable(snapshot):
            return None
        identity = {
            "bias": snapshot.get("bias"),
            "setup_type": snapshot.get("setup_type"),
            "zone": snapshot.get("zone"),
            "sweep_event": snapshot.get("sweep_time") or snapshot.get("reclaim_time"),
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()[:20]

    @staticmethod
    def _new_decision(previous, snapshot):
        """True only once per completed decision candle."""
        decision_at = snapshot.get("timestamp")
        return bool(decision_at and decision_at != previous)

    @staticmethod
    def _expectancy(snapshot):
        """Soft diagnostic only — probability is heuristic/uncalibrated."""
        probability = snapshot.get("probability_tp_before_sl")
        reward_risk = snapshot.get("reward_risk")
        if probability is None or reward_risk is None:
            return None
        probability = float(probability)
        return probability * float(reward_risk) - (1.0 - probability)

    def _material_replacement(self, active, snapshot, has_active_position=True):
        """Reject same-direction replacements while an actual trade is active.

        Under Option A (strict single-position rule), while a position is active,
        additional setups in the SAME direction are diagnostic-only and cannot
        replace or mutate the active trade. Only confirmed setups in the OPPOSITE
        direction can replace/flip the active position.
        """
        current = (active or {}).get("snapshot") or {}
        if not current:
            return True
        if not has_active_position:
            return True
        if snapshot.get("bias") != current.get("bias"):
            return True
        # Same-direction setup while a trade/signal is active: diagnostic only.
        snapshot["replacement_status"] = "ignored_same_direction_active"
        snapshot["diagnostic_only"] = True
        return False

    @staticmethod
    def _retire(state, signal_id):
        if not signal_id:
            return
        retired = [item for item in state.get("retired_signal_ids") or [] if item != signal_id]
        retired.append(signal_id)
        state["retired_signal_ids"] = retired[-100:]

    def adopt_open_position(self, persisted_state, position, observed_at):
        """Bind a legacy persisted position to lifecycle state without churn."""
        state = {**self.initial_state(), **(persisted_state or {})}
        if state.get("version") != self.VERSION:
            previous = state
            state = self.initial_state()
            state["event_sequence"] = int(previous.get("event_sequence") or 0)
            state["stable_bias"] = previous.get("stable_bias")
        if state.get("active") or not position:
            return state, None
        side = position.get("side")
        bias = "bullish" if side == "long" else "bearish" if side == "short" else None
        if bias is None:
            return state, None
        snapshot = {
            "timestamp": _timestamp(position.get("decision_time") or position.get("entry_time")),
            "bias": bias,
            "setup_type": "reversal",
            "zone": position.get("zone"),
            "zone_kind": None,
            "sweep_status": "confirmed",
            "sweep_time": _timestamp(position.get("sweep_time")),
            "reclaim_time": _timestamp(position.get("reclaim_time")),
            "orderflow_confirmation": True,
            "orderflow_reason": "legacy_position_adopted",
            "entry": position.get("entry"),
            "stop": position.get("stop"),
            "target": position.get("target"),
            "reward_risk": None,
            "probability_tp_before_sl": position.get("probability_tp_before_sl"),
            "position_size": position.get("size"),
            "no_trade_reason": None,
            "regime_4h": None,
            "regime_1h": None,
            "setup_15m": None,
            "orderflow_score": None,
            "setup_atr": position.get("setup_atr"),
            "entry_type": "limit" if position.get("signal_time") else "market",
        }
        signal_id = position.get("signal_id") or self.signal_id(snapshot)
        if signal_id is None:
            encoded = json.dumps({"side": side, "zone": position.get("zone"), "entry_time": position.get("entry_time")}, sort_keys=True)
            signal_id = hashlib.sha256(encoded.encode()).hexdigest()[:20]
        now = pd.Timestamp(observed_at).isoformat()
        state["active"] = {
            "signal_id": signal_id,
            "activated_at": position.get("confirmed_at") or position.get("entry_time") or now,
            "last_seen_at": now,
            "last_decision_at": snapshot.get("timestamp"),
            "snapshot": snapshot,
            "adopted": True,
        }
        state["stable_bias"] = state.get("stable_bias") or bias
        state["updated_at"] = now
        return state, signal_id

    @staticmethod
    def _classify_terminal(snapshot):
        sweep = str(snapshot.get("sweep_status") or "")
        reason = str(snapshot.get("no_trade_reason") or "")
        if "expired" in sweep or "expired" in reason:
            return "setup_expired"
        return "setup_invalidated"

    @staticmethod
    def _event(state, event_type, signal_id, observed_at, **payload):
        state["event_sequence"] = int(state.get("event_sequence") or 0) + 1
        event_id = f"lifecycle-{state['event_sequence']}-{event_type}-{(signal_id or 'none')[:8]}"
        return {
            "event_id": event_id,
            "event_type": event_type,
            "signal_id": signal_id,
            "created_at": pd.Timestamp(observed_at).isoformat(),
            "cooldown_exempt": False,
            **payload,
        }

    @staticmethod
    def _setup_event(state, candidate, observed_at, bias_reversal=False, replaced_signal_id=None):
        snapshot = candidate["snapshot"]
        bias = str(snapshot["bias"]).capitalize()
        setup = _human(snapshot.get("setup_type"))
        label = f"{bias} reversal confirmed" if bias_reversal else f"{bias} {setup} confirmed"
        body = (
            f"{label} · Entry {_money(snapshot['entry'])} · "
            f"SL {_money(snapshot['stop'])} · TP {_money(snapshot['target'])}"
        )
        return SignalLifecycle._event(
            state,
            "setup_confirmed",
            candidate["signal_id"],
            observed_at,
            title="BTC setup confirmed",
            body=body,
            snapshot=snapshot,
            bias_reversal=bias_reversal,
            replaced_signal_id=replaced_signal_id,
        )

    def evaluate(self, persisted_state, prediction, paper_status, observed_at):
        state = {**self.initial_state(), **(persisted_state or {})}
        if state.get("version") != self.VERSION:
            previous = state
            state = self.initial_state()
            # Never reuse durable event IDs after a schema migration.
            state["event_sequence"] = int(previous.get("event_sequence") or 0)
            state["stable_bias"] = previous.get("stable_bias")
        events = []
        snapshot = self.snapshot(prediction)
        signal_id = self.signal_id(snapshot)
        actionable = signal_id is not None
        active = state.get("active")
        is_real_paper_status = (
            isinstance(paper_status, dict)
            and (
                "closed_trades" in paper_status
                or "open_position" in paper_status
            )
        )
        if is_real_paper_status:
            has_active_position = bool(
                paper_status.get("open_position")
                or paper_status.get("pending_order")
            )
        else:
            has_active_position = bool(active)

        # If the paper ledger is flat (e.g. previous trade closed or was skipped),
        # clear any stale active signal lock so fresh valid setups can be taken and notified.
        if active and is_real_paper_status and not has_active_position:
            self._retire(state, active.get("signal_id"))
            active = None
            state["active"] = None
            state["missing_observations"] = 0

        # Definitive paper closures terminate the matching active lifecycle.
        for trade in (paper_status or {}).get("newly_closed") or []:
            reason = str(trade.get("exit_reason") or "")
            if not active:
                continue
            if reason in ("target", "stop"):
                self._retire(state, active.get("signal_id"))
                active = None
                state["active"] = None
                state["missing_observations"] = 0
            elif reason in ("signal_flipped", "superseded_by_new_setup", "signal_neutralized"):
                events.append(self._event(
                    state,
                    "setup_invalidated",
                    active.get("signal_id"),
                    observed_at,
                    title="BTC setup invalidated",
                    body=f"{str(active['snapshot'].get('bias')).capitalize()} · {_human(reason)}",
                    reason=reason,
                    snapshot=active.get("snapshot"),
                ))
                self._retire(state, active.get("signal_id"))
                active = None
                state["active"] = None
                state["missing_observations"] = 0

        # Candidate confirmation counts unique completed decision bars, never
        # repeated 45-second polls of the same timestamp.
        if actionable and signal_id in set(state.get("retired_signal_ids") or []):
            actionable = False
            signal_id = None
            state["candidate"] = None
        if actionable:
            if active and active.get("signal_id") != signal_id and not self._material_replacement(active, snapshot, has_active_position=has_active_position):
                actionable = False
                signal_id = None
                state["candidate"] = None

        if actionable:
            candidate = state.get("candidate")
            if candidate and candidate.get("signal_id") == signal_id:
                if self._new_decision(candidate.get("last_decision_at"), snapshot):
                    candidate["observations"] = int(candidate.get("observations") or 0) + 1
                    candidate["snapshot"] = snapshot
                    candidate["last_decision_at"] = snapshot.get("timestamp")
                candidate["last_seen_at"] = pd.Timestamp(observed_at).isoformat()
            else:
                candidate = {
                    "signal_id": signal_id,
                    "observations": 1,
                    "first_seen_at": pd.Timestamp(observed_at).isoformat(),
                    "last_seen_at": pd.Timestamp(observed_at).isoformat(),
                    "last_decision_at": snapshot.get("timestamp"),
                    "snapshot": snapshot,
                }
            state["candidate"] = candidate
            confirmed = candidate["observations"] >= self.confirm_observations
            if active and active.get("signal_id") == signal_id:
                active["last_seen_at"] = pd.Timestamp(observed_at).isoformat()
                active["last_decision_at"] = snapshot.get("timestamp")
                state["active"] = active
                state["missing_observations"] = 0
                state["last_missing_decision_at"] = None
                state["candidate"] = None
            elif confirmed:
                previous_bias = state.get("stable_bias")
                bias_reversal = bool(
                    previous_bias in ("bullish", "bearish")
                    and snapshot.get("bias") != previous_bias
                )
                replaced_signal_id = active.get("signal_id") if active else None
                self._retire(state, replaced_signal_id)
                events.append(self._setup_event(
                    state,
                    candidate,
                    observed_at,
                    bias_reversal=bias_reversal,
                    replaced_signal_id=replaced_signal_id,
                ))
                state["active"] = {
                    "signal_id": signal_id,
                    "activated_at": pd.Timestamp(observed_at).isoformat(),
                    "last_seen_at": pd.Timestamp(observed_at).isoformat(),
                    "last_decision_at": snapshot.get("timestamp"),
                    "snapshot": snapshot,
                }
                active = state["active"]
                state["candidate"] = None
                state["missing_observations"] = 0
                state["last_missing_decision_at"] = None
                state["stable_bias"] = snapshot.get("bias")
                state["bias_candidate"] = None
                state["bias_observations"] = 0
        else:
            state["candidate"] = None
            if active:
                active_bias = (active.get("snapshot") or {}).get("bias")
                bias = snapshot.get("bias")
                reason = str(snapshot.get("no_trade_reason") or snapshot.get("sweep_status") or "")
                explicit_expiry = "expired" in reason
                structural_exit = bias == "neutral" or (bias in ("bullish", "bearish") and bias != active_bias)
                should_count = explicit_expiry or structural_exit
                if should_count and self._new_decision(state.get("last_missing_decision_at"), snapshot):
                    state["missing_observations"] = int(state.get("missing_observations") or 0) + 1
                    state["last_missing_decision_at"] = snapshot.get("timestamp")
                elif not should_count:
                    # A scanner choosing another zone, or temporary footprint /
                    # sweep loss, does not kill an already confirmed thesis.
                    state["missing_observations"] = 0
                    state["last_missing_decision_at"] = None
                if should_count and state["missing_observations"] >= self.invalidation_observations:
                    event_type = self._classify_terminal(snapshot)
                    terminal_reason = "signal_flipped" if bias in ("bullish", "bearish") and bias != active_bias else ("signal_neutralized" if bias == "neutral" else reason)
                    events.append(self._event(
                        state,
                        event_type,
                        active.get("signal_id"),
                        observed_at,
                        title=("BTC setup expired" if event_type == "setup_expired" else "BTC setup invalidated"),
                        body=(
                            f"{str(active['snapshot'].get('bias')).capitalize()} · "
                            f"{_human(terminal_reason)}"
                        ),
                        reason=terminal_reason,
                        snapshot=active.get("snapshot"),
                    ))
                    self._retire(state, active.get("signal_id"))
                    state["active"] = None
                    active = None
                    state["missing_observations"] = 0
                    state["last_missing_decision_at"] = None

        # Bias reversals are notified only after the predictor's non-neutral bias
        # persists. A setup-confirmed event above already carries the reversal.
        bias = snapshot.get("bias")
        if state.get("stable_bias") is None and bias in ("bullish", "bearish"):
            state["stable_bias"] = bias
        elif bias in ("bullish", "bearish") and bias != state.get("stable_bias"):
            is_new_bias_bar = self._new_decision(state.get("last_bias_decision_at"), snapshot)
            if state.get("bias_candidate") == bias:
                if is_new_bias_bar:
                    state["bias_observations"] = int(state.get("bias_observations") or 0) + 1
            else:
                state["bias_candidate"] = bias
                state["bias_observations"] = 1
            if is_new_bias_bar:
                state["last_bias_decision_at"] = snapshot.get("timestamp")
            setup_already_announced = any(
                event.get("event_type") == "setup_confirmed" and event.get("bias_reversal")
                for event in events
            )
            if state["bias_observations"] >= self.bias_observations and not setup_already_announced:
                old_bias = state.get("stable_bias")
                events.append(self._event(
                    state,
                    "bias_reversal",
                    None,
                    observed_at,
                    title="BTC bias reversal",
                    body=f"{str(old_bias).capitalize()} → {str(bias).capitalize()} · Confirmed structure change",
                    previous_bias=old_bias,
                    bias=bias,
                ))
                state["stable_bias"] = bias
                state["bias_candidate"] = None
                state["bias_observations"] = 0
                state["last_bias_decision_at"] = None
        elif bias == state.get("stable_bias"):
            state["bias_candidate"] = None
            state["bias_observations"] = 0
            state["last_bias_decision_at"] = None

        state["updated_at"] = pd.Timestamp(observed_at).isoformat()
        return state, events
