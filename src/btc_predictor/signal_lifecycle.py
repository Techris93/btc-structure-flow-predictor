from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


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

    This controls notifications only. It does not change predictor decisions,
    paper fills, stops, targets, or position accounting.
    """

    VERSION = 1

    def __init__(self, confirm_observations=2, invalidation_observations=3, bias_observations=2):
        self.confirm_observations = max(1, int(confirm_observations))
        self.invalidation_observations = max(1, int(invalidation_observations))
        self.bias_observations = max(1, int(bias_observations))

    @staticmethod
    def initial_state():
        return {
            "version": SignalLifecycle.VERSION,
            "candidate": None,
            "active": None,
            "missing_observations": 0,
            "stable_bias": None,
            "bias_candidate": None,
            "bias_observations": 0,
            "event_sequence": 0,
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
        )
        snapshot = {field: _value(prediction, field) for field in fields}
        for field in ("timestamp", "sweep_time", "reclaim_time"):
            snapshot[field] = _timestamp(snapshot.get(field))
        return snapshot

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
    def _completed_candle(snapshot, observed_at):
        decision_at = snapshot.get("timestamp")
        if not decision_at:
            return False
        decision_at = pd.Timestamp(decision_at)
        observed_at = pd.Timestamp(observed_at)
        return decision_at < observed_at.floor("min")

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
            state = self.initial_state()
        events = []
        snapshot = self.snapshot(prediction)
        signal_id = self.signal_id(snapshot)
        actionable = signal_id is not None
        active = state.get("active")

        # Definitive paper closures terminate the matching active lifecycle.
        for trade in (paper_status or {}).get("newly_closed") or []:
            reason = str(trade.get("exit_reason") or "")
            if not active:
                continue
            if reason in ("target", "stop"):
                active = None
                state["active"] = None
                state["missing_observations"] = 0
            elif reason in ("signal_flipped", "superseded_by_new_setup"):
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
                active = None
                state["active"] = None
                state["missing_observations"] = 0

        # Candidate confirmation: two observations, or one already-completed 1m candle.
        if actionable:
            candidate = state.get("candidate")
            if candidate and candidate.get("signal_id") == signal_id:
                candidate["observations"] = int(candidate.get("observations") or 0) + 1
                candidate["snapshot"] = snapshot
                candidate["last_seen_at"] = pd.Timestamp(observed_at).isoformat()
            else:
                candidate = {
                    "signal_id": signal_id,
                    "observations": 1,
                    "first_seen_at": pd.Timestamp(observed_at).isoformat(),
                    "last_seen_at": pd.Timestamp(observed_at).isoformat(),
                    "snapshot": snapshot,
                }
            state["candidate"] = candidate
            confirmed = (
                candidate["observations"] >= self.confirm_observations
                or self._completed_candle(snapshot, observed_at)
            )
            if active and active.get("signal_id") == signal_id:
                active["last_seen_at"] = pd.Timestamp(observed_at).isoformat()
                state["active"] = active
                state["missing_observations"] = 0
                state["candidate"] = None
            elif confirmed:
                previous_bias = state.get("stable_bias")
                bias_reversal = bool(
                    previous_bias in ("bullish", "bearish")
                    and snapshot.get("bias") != previous_bias
                )
                replaced_signal_id = active.get("signal_id") if active else None
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
                    "snapshot": snapshot,
                }
                active = state["active"]
                state["candidate"] = None
                state["missing_observations"] = 0
                state["stable_bias"] = snapshot.get("bias")
                state["bias_candidate"] = None
                state["bias_observations"] = 0
        else:
            state["candidate"] = None
            if active:
                state["missing_observations"] = int(state.get("missing_observations") or 0) + 1
                if state["missing_observations"] >= self.invalidation_observations:
                    event_type = self._classify_terminal(snapshot)
                    events.append(self._event(
                        state,
                        event_type,
                        active.get("signal_id"),
                        observed_at,
                        title=("BTC setup expired" if event_type == "setup_expired" else "BTC setup invalidated"),
                        body=(
                            f"{str(active['snapshot'].get('bias')).capitalize()} · "
                            f"{_human(snapshot.get('no_trade_reason') or snapshot.get('sweep_status'))}"
                        ),
                        reason=snapshot.get("no_trade_reason") or snapshot.get("sweep_status"),
                        snapshot=active.get("snapshot"),
                    ))
                    state["active"] = None
                    active = None
                    state["missing_observations"] = 0

        # Bias reversals are notified only after the predictor's non-neutral bias
        # persists. A setup-confirmed event above already carries the reversal.
        bias = snapshot.get("bias")
        if state.get("stable_bias") is None and bias in ("bullish", "bearish"):
            state["stable_bias"] = bias
        elif bias in ("bullish", "bearish") and bias != state.get("stable_bias"):
            if state.get("bias_candidate") == bias:
                state["bias_observations"] = int(state.get("bias_observations") or 0) + 1
            else:
                state["bias_candidate"] = bias
                state["bias_observations"] = 1
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
        elif bias == state.get("stable_bias"):
            state["bias_candidate"] = None
            state["bias_observations"] = 0

        state["updated_at"] = pd.Timestamp(observed_at).isoformat()
        return state, events
