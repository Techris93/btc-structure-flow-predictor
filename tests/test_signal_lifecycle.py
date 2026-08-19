from dataclasses import replace

import pandas as pd

from btc_predictor.models import PredictorOutput
from btc_predictor.signal_lifecycle import SignalLifecycle


OBSERVED = pd.Timestamp("2026-07-28T12:00:30Z")


def prediction(**updates):
    base = PredictorOutput(
        timestamp=OBSERVED.floor("min"),
        bias="bullish",
        setup_type="reversal",
        zone="equal_lows:abc",
        zone_kind="equal_lows",
        sweep_status="confirmed",
        sweep_time="2026-07-28T11:58:00+00:00",
        reclaim_time="2026-07-28T11:59:00+00:00",
        orderflow_confirmation=True,
        orderflow_reason="confirmed",
        entry=65000.0,
        stop=64750.0,
        target=65500.0,
        reward_risk=2.0,
        probability_tp_before_sl=0.66,
        position_size=0.4,
        no_trade_reason=None,
        regime_4h="bullish",
        regime_1h="bullish",
        setup_15m="bullish",
        orderflow_score=0.8,
        setup_atr=250.0,
    )
    return replace(base, **updates)


def activate(engine, setup=None):
    setup = setup or prediction()
    state, first = engine.evaluate(engine.initial_state(), setup, {}, OBSERVED)
    assert first == []
    next_bar = replace(setup, timestamp=pd.Timestamp(setup.timestamp) + pd.Timedelta(minutes=1))
    state, events = engine.evaluate(state, next_bar, {}, OBSERVED + pd.Timedelta(minutes=1, seconds=30))
    assert [event["event_type"] for event in events] == ["setup_confirmed"]
    return state, events[0]


def test_signal_identity_ignores_volatile_trade_levels_and_scores():
    engine = SignalLifecycle()
    first = engine.snapshot(prediction())
    recalculated = engine.snapshot(prediction(
        timestamp=OBSERVED + pd.Timedelta(seconds=45),
        entry=65012.0,
        stop=64762.0,
        target=65540.0,
        reward_risk=2.11,
        probability_tp_before_sl=0.71,
        orderflow_reason="absorption_confirmed",
    ))

    assert engine.signal_id(first) == engine.signal_id(recalculated)
    assert engine.signal_id(first) != engine.signal_id(
        engine.snapshot(prediction(zone="equal_lows:different"))
    )
    assert engine.signal_id(first) != engine.signal_id(
        engine.snapshot(prediction(sweep_time="2026-07-28T11:57:00Z"))
    )


def test_candidate_counts_unique_closed_bars_and_notifies_once():
    engine = SignalLifecycle(confirm_observations=2)
    state, events = engine.evaluate(engine.initial_state(), prediction(), {}, OBSERVED)
    assert events == []
    assert state["candidate"]["observations"] == 1

    state, events = engine.evaluate(
        state, prediction(entry=65005.0), {}, OBSERVED + pd.Timedelta(seconds=20)
    )
    assert events == []
    assert state["candidate"]["observations"] == 1

    state, events = engine.evaluate(
        state,
        prediction(timestamp=OBSERVED.floor("min") + pd.Timedelta(minutes=1), entry=65005.0),
        {},
        OBSERVED + pd.Timedelta(minutes=1, seconds=20),
    )
    assert [event["event_type"] for event in events] == ["setup_confirmed"]
    signal_id = state["active"]["signal_id"]

    state, events = engine.evaluate(
        state, prediction(entry=65020.0), {}, OBSERVED + pd.Timedelta(minutes=1)
    )
    assert events == []
    assert state["active"]["signal_id"] == signal_id


def test_old_completed_candle_cannot_bypass_confirmation():
    engine = SignalLifecycle(confirm_observations=2)
    completed = prediction(timestamp=OBSERVED.floor("min") - pd.Timedelta(minutes=1))

    state, events = engine.evaluate(engine.initial_state(), completed, {}, OBSERVED)

    assert state["active"] is None
    assert events == []
    assert state["candidate"]["observations"] == 1


def test_same_bias_scanner_drift_does_not_invalidate_active_setup():
    engine = SignalLifecycle(invalidation_observations=3)
    state, _ = activate(engine)
    unconfirmed = prediction(
        orderflow_confirmation=False,
        orderflow_reason="weak_absorption",
        entry=None,
        stop=None,
        target=None,
        position_size=None,
        no_trade_reason="orderflow_not_confirmed",
    )

    for offset in (2, 3, 4, 5):
        state, events = engine.evaluate(
            state,
            replace(unconfirmed, timestamp=OBSERVED.floor("min") + pd.Timedelta(minutes=offset)),
            {},
            OBSERVED + pd.Timedelta(minutes=offset, seconds=30),
        )
        assert events == []
        assert state["active"] is not None
    assert state["missing_observations"] == 0


def test_neutral_invalidation_counts_unique_closed_bars_only():
    engine = SignalLifecycle(invalidation_observations=3)
    state, _ = activate(engine)
    neutral = prediction(
        timestamp=OBSERVED.floor("min") + pd.Timedelta(minutes=2),
        bias="neutral",
        setup_type=None,
        zone=None,
        sweep_status="none",
        sweep_time=None,
        reclaim_time=None,
        orderflow_confirmation=False,
        entry=None,
        stop=None,
        target=None,
        position_size=None,
        no_trade_reason="timeframe_conflict",
    )
    state, events = engine.evaluate(state, neutral, {}, OBSERVED + pd.Timedelta(minutes=2, seconds=5))
    assert events == [] and state["missing_observations"] == 1
    state, events = engine.evaluate(state, neutral, {}, OBSERVED + pd.Timedelta(minutes=2, seconds=50))
    assert events == [] and state["missing_observations"] == 1
    for minute in (3, 4):
        state, events = engine.evaluate(
            state,
            replace(neutral, timestamp=OBSERVED.floor("min") + pd.Timedelta(minutes=minute)),
            {},
            OBSERVED + pd.Timedelta(minutes=minute, seconds=30),
        )
    assert [event["event_type"] for event in events] == ["setup_invalidated"]
    assert events[0]["reason"] == "signal_neutralized"


def test_nearby_different_zone_cannot_replace_active_setup():
    engine = SignalLifecycle(confirm_observations=2, replacement_distance_atr=.25)
    state, _ = activate(engine)
    nearby = prediction(
        timestamp=OBSERVED.floor("min") + pd.Timedelta(minutes=2),
        zone="vwap_lower:nearby",
        sweep_time="2026-07-28T12:01:00Z",
        entry=65020.0,
        reward_risk=3.0,
        probability_tp_before_sl=.8,
    )
    for minute in (2, 3, 4):
        state, events = engine.evaluate(
            state,
            replace(nearby, timestamp=OBSERVED.floor("min") + pd.Timedelta(minutes=minute)),
            {},
            OBSERVED + pd.Timedelta(minutes=minute, seconds=30),
        )
        assert events == []
    assert state["active"]["snapshot"]["zone"] == "equal_lows:abc"
    assert state["candidate"] is None


def test_same_direction_setup_cannot_replace_active_setup():
    """Option A: Same-direction setups are diagnostic only while an active signal is held."""
    engine = SignalLifecycle(confirm_observations=2, replacement_distance_atr=.25)
    state, _ = activate(engine)
    replacement = prediction(
        timestamp=OBSERVED.floor("min") + pd.Timedelta(minutes=2),
        zone="previous_day_low:far",
        sweep_time="2026-07-28T12:01:00Z",
        entry=65100.0,
        reward_risk=3.0,
        probability_tp_before_sl=.8,
    )
    state, events = engine.evaluate(state, replacement, {}, OBSERVED + pd.Timedelta(minutes=2, seconds=30))
    assert events == []
    state, events = engine.evaluate(state, replacement, {}, OBSERVED + pd.Timedelta(minutes=2, seconds=50))
    assert events == []
    state, events = engine.evaluate(
        state,
        replace(replacement, timestamp=OBSERVED.floor("min") + pd.Timedelta(minutes=3)),
        {},
        OBSERVED + pd.Timedelta(minutes=3, seconds=30),
    )
    # Under Option A, same direction does NOT emit setup_confirmed and does NOT replace active
    assert events == []
    assert state["active"]["snapshot"]["zone"] == "equal_lows:abc"


def test_recovery_before_threshold_resets_invalidation_counter():
    engine = SignalLifecycle(invalidation_observations=3)
    state, _ = activate(engine)
    unavailable = prediction(
        entry=None,
        stop=None,
        target=None,
        position_size=None,
        orderflow_confirmation=False,
        no_trade_reason="orderflow_not_confirmed",
    )
    state, _ = engine.evaluate(
        state,
        replace(unavailable, timestamp=OBSERVED.floor("min") + pd.Timedelta(minutes=2), bias="neutral"),
        {},
        OBSERVED + pd.Timedelta(minutes=2, seconds=30),
    )
    state, events = engine.evaluate(
        state,
        prediction(timestamp=OBSERVED.floor("min") + pd.Timedelta(minutes=3), entry=65015.0),
        {},
        OBSERVED + pd.Timedelta(minutes=3, seconds=30),
    )

    assert events == []
    assert state["missing_observations"] == 0
    assert state["active"] is not None


def test_expired_reason_uses_expired_terminal_event():
    engine = SignalLifecycle(invalidation_observations=1)
    state, _ = activate(engine)
    expired = prediction(
        timestamp=OBSERVED.floor("min") + pd.Timedelta(minutes=2),
        sweep_status="expired",
        entry=None,
        stop=None,
        target=None,
        position_size=None,
        orderflow_confirmation=False,
        no_trade_reason="setup_expired",
    )

    state, events = engine.evaluate(
        state, expired, {}, OBSERVED + pd.Timedelta(minutes=2)
    )

    assert [event["event_type"] for event in events] == ["setup_expired"]


def test_bias_reversal_requires_two_non_neutral_observations():
    engine = SignalLifecycle(bias_observations=2)
    state = engine.initial_state()
    neutral = prediction(
        bias="neutral",
        regime_4h="bearish",
        regime_1h="bullish",
        setup_15m="bullish",
        setup_type=None,
        zone=None,
        sweep_status="none",
        sweep_time=None,
        reclaim_time=None,
        orderflow_confirmation=False,
        entry=None,
        stop=None,
        target=None,
        position_size=None,
        no_trade_reason="timeframe_conflict",
    )
    bearish = replace(neutral, bias="bearish", regime_1h="bearish")
    state["stable_bias"] = "bullish"

    state, events = engine.evaluate(state, neutral, {}, OBSERVED)
    assert events == []
    assert state["stable_bias"] == "bullish"
    state, events = engine.evaluate(
        state,
        replace(bearish, timestamp=OBSERVED.floor("min") + pd.Timedelta(minutes=1)),
        {},
        OBSERVED + pd.Timedelta(minutes=1),
    )
    assert events == []
    state, events = engine.evaluate(
        state,
        replace(bearish, timestamp=OBSERVED.floor("min") + pd.Timedelta(minutes=2)),
        {},
        OBSERVED + pd.Timedelta(minutes=2),
    )
    assert [event["event_type"] for event in events] == ["bias_reversal"]
    assert state["stable_bias"] == "bearish"
    state, events = engine.evaluate(
        state,
        replace(bearish, timestamp=OBSERVED.floor("min") + pd.Timedelta(minutes=3)),
        {},
        OBSERVED + pd.Timedelta(minutes=3),
    )
    assert events == []


def test_opposite_confirmed_setup_combines_reversal_into_one_event():
    engine = SignalLifecycle(confirm_observations=2, bias_observations=2)
    state, _ = activate(engine)
    bearish = prediction(
        timestamp=OBSERVED.floor("min") + pd.Timedelta(minutes=3),
        bias="bearish",
        zone="equal_highs:def",
        sweep_time="2026-07-28T12:02:00Z",
        regime_4h="bearish",
        regime_1h="bearish",
        setup_15m="bearish",
        entry=64900.0,
        stop=65150.0,
        target=64400.0,
    )

    state, events = engine.evaluate(
        state, bearish, {}, OBSERVED + pd.Timedelta(minutes=3)
    )
    assert events == []
    state, events = engine.evaluate(
        state,
        replace(bearish, timestamp=pd.Timestamp(bearish.timestamp) + pd.Timedelta(minutes=1)),
        {},
        OBSERVED + pd.Timedelta(minutes=4, seconds=45),
    )

    assert [event["event_type"] for event in events] == ["setup_confirmed"]
    assert events[0]["bias_reversal"] is True
    assert events[0]["replaced_signal_id"] is not None


def test_tp_or_sl_closure_clears_lifecycle_without_duplicate_terminal_event():
    engine = SignalLifecycle()
    state, setup_event = activate(engine)
    paper = {"newly_closed": [{"exit_reason": "target"}]}

    state, events = engine.evaluate(
        state,
        prediction(entry=None, stop=None, target=None, position_size=None),
        paper,
        OBSERVED + pd.Timedelta(minutes=5),
    )

    assert setup_event["event_type"] == "setup_confirmed"
    assert state["active"] is None
    assert events == []

    # The same completed setup cannot immediately re-enter after its terminal
    # market outcome. A fresh sweep episode must create a new signal identity.
    for minute in (6, 7, 8):
        state, events = engine.evaluate(
            state,
            prediction(timestamp=OBSERVED.floor("min") + pd.Timedelta(minutes=minute)),
            {},
            OBSERVED + pd.Timedelta(minutes=minute, seconds=30),
        )
        assert events == []
        assert state["candidate"] is None


def test_state_migration_preserves_event_sequence_to_avoid_dedupe_collision():
    engine = SignalLifecycle()
    old = SignalLifecycle.initial_state()
    old.update({"version": 1, "event_sequence": 41, "stable_bias": "bearish"})

    state, events = engine.evaluate(old, prediction(), {}, OBSERVED)

    assert events == []
    assert state["version"] == SignalLifecycle.VERSION
    assert state["event_sequence"] == 41
    assert state["stable_bias"] == "bearish"


def test_legacy_open_position_is_adopted_without_notification_or_reentry():
    engine = SignalLifecycle()
    position = {
        "side": "long",
        "entry_time": "2026-07-28T12:00:00Z",
        "entry": 65000.0,
        "stop": 64750.0,
        "target": 65500.0,
        "size": .4,
        "zone": "equal_lows:abc",
        "sweep_time": "2026-07-28T11:58:00Z",
    }

    state, signal_id = engine.adopt_open_position(engine.initial_state(), position, OBSERVED)

    assert signal_id is not None
    assert state["active"]["signal_id"] == signal_id
    assert state["active"]["adopted"] is True
    assert state["active"]["snapshot"]["entry"] == 65000.0


def test_signal_flip_emits_one_invalidation_event():
    engine = SignalLifecycle()
    state, _ = activate(engine)
    paper = {"newly_closed": [{"exit_reason": "signal_flipped"}]}

    state, events = engine.evaluate(
        state,
        prediction(entry=None, stop=None, target=None, position_size=None),
        paper,
        OBSERVED + pd.Timedelta(minutes=5),
    )

    assert state["active"] is None
    assert [event["event_type"] for event in events] == ["setup_invalidated"]


def test_signal_neutralized_emits_one_invalidation_event():
    engine = SignalLifecycle()
    state, _ = activate(engine)
    paper = {"newly_closed": [{"exit_reason": "signal_neutralized"}]}

    state, events = engine.evaluate(
        state,
        prediction(bias="neutral", entry=None, stop=None, target=None, position_size=None),
        paper,
        OBSERVED + pd.Timedelta(minutes=5),
    )

    assert state["active"] is None
    assert [event["event_type"] for event in events] == ["setup_invalidated"]
    assert events[0]["reason"] == "signal_neutralized"
    assert events[0]["body"] == "Bullish · Signal neutralized"
