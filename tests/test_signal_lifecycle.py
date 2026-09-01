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


def test_continuation_micro_drifting_in_trend_produces_single_initial_confirmation():
    """Verify that micro-drifting price action in a trend produces exactly 1 clean confirmation instead of 10 rapid duplicate notifications."""
    engine = SignalLifecycle(
        confirm_observations=2,
        continuation_rearm_seconds=1800,  # 30 minutes cooldown
        continuation_rearm_atr=1.0,       # 1.0 ATR spacing
    )
    state = engine.initial_state()
    flat_paper = {"open_position": None, "pending_order": None, "closed_trades": 0}
    base_time = pd.Timestamp("2026-08-21 10:00:00", tz="UTC")
    all_events = []

    # 1. Initial continuation setup at 10:00 UTC (Entry $70,000, ATR $250)
    p0 = prediction(
        timestamp=base_time,
        setup_type="continuation",
        zone="zone_cont_0",
        sweep_time="2026-08-21T09:58:00Z",
        entry=70000.0,
        stop=69300.0,
        target=71400.0,
        setup_atr=250.0,
    )
    state, ev = engine.evaluate(state, p0, flat_paper, base_time + pd.Timedelta(seconds=30))
    assert ev == []
    all_events.extend(ev)

    p1 = replace(p0, timestamp=base_time + pd.Timedelta(minutes=1))
    state, ev = engine.evaluate(state, p1, flat_paper, base_time + pd.Timedelta(minutes=1, seconds=30))
    assert len(ev) == 1
    assert ev[0]["event_type"] == "setup_confirmed"
    assert ev[0]["snapshot"]["entry"] == 70000.0
    all_events.extend(ev)

    # 2. Simulate 9 consecutive 1-minute bars of micro price drift ($70,010 -> $70,090, drift < 1.0 ATR)
    for minute in range(2, 11):
        bar_time = base_time + pd.Timedelta(minutes=minute)
        obs_time = bar_time + pd.Timedelta(seconds=30)
        drift_price = 70000.0 + minute * 10.0  # Drift is only $20..$100, which is < 1.0 ATR ($250)
        p_drift = prediction(
            timestamp=bar_time,
            setup_type="continuation",
            zone=f"zone_cont_{minute}",
            sweep_time=f"2026-08-21T10:{minute:02d}:00Z",
            reclaim_time=f"2026-08-21T10:{minute:02d}:15Z",
            entry=drift_price,
            stop=drift_price - 700.0,
            target=drift_price + 1400.0,
            setup_atr=250.0,
        )
        state, ev = engine.evaluate(state, p_drift, flat_paper, obs_time)
        assert ev == [], f"Minute {minute} should be throttled by refractory period but emitted {ev}"
        all_events.extend(ev)

    # Across all 10 initial bars + micro-drifts, EXACTLY 1 confirmation notification was generated
    assert len(all_events) == 1

    # 3. After 35 minutes (past 30m cooldown), a new continuation setup confirms cleanly
    t35 = base_time + pd.Timedelta(minutes=35)
    p35_a = prediction(
        timestamp=t35,
        setup_type="continuation",
        zone="zone_cont_35",
        sweep_time="2026-08-21T10:33:00Z",
        entry=70150.0,
        stop=69448.5,
        target=71553.0,
        setup_atr=250.0,
    )
    state, ev = engine.evaluate(state, p35_a, flat_paper, t35 + pd.Timedelta(seconds=30))
    assert ev == []

    t36 = base_time + pd.Timedelta(minutes=36)
    p35_b = replace(p35_a, timestamp=t36)
    state, ev = engine.evaluate(state, p35_b, flat_paper, t36 + pd.Timedelta(seconds=30))
    assert len(ev) == 1
    assert ev[0]["event_type"] == "setup_confirmed"
    assert ev[0]["snapshot"]["entry"] == 70150.0
    all_events.extend(ev)
    assert len(all_events) == 2


def test_continuation_structural_expansion_rearms_before_cooldown_expires():
    """Verify that a genuine structural expansion (>= 1.0 ATR) confirms even within the 30m window."""
    engine = SignalLifecycle(
        confirm_observations=2,
        continuation_rearm_seconds=1800,
        continuation_rearm_atr=1.0,
    )
    state = engine.initial_state()
    flat_paper = {"open_position": None, "pending_order": None, "closed_trades": 0}
    base_time = pd.Timestamp("2026-08-21 10:00:00", tz="UTC")

    # 1. Setup #1 confirmed at $70,000 (ATR = 250)
    p0 = prediction(
        timestamp=base_time,
        setup_type="continuation",
        zone="zone_0",
        entry=70000.0,
        setup_atr=250.0,
    )
    state, _ = engine.evaluate(state, p0, flat_paper, base_time + pd.Timedelta(seconds=30))
    p1 = replace(p0, timestamp=base_time + pd.Timedelta(minutes=1))
    state, ev1 = engine.evaluate(state, p1, flat_paper, base_time + pd.Timedelta(minutes=1, seconds=30))
    assert len(ev1) == 1
    assert ev1[0]["event_type"] == "setup_confirmed"

    # 2. Only 8 minutes later, price expands by $300 (which is > 1.0 ATR $250)
    t8 = base_time + pd.Timedelta(minutes=8)
    p8 = prediction(
        timestamp=t8,
        setup_type="continuation",
        zone="zone_expanded",
        sweep_time="2026-08-21T10:07:00Z",
        entry=70300.0,  # 70,300 - 70,000 = 300 >= 1.0 * 250
        setup_atr=250.0,
    )
    state, ev8_a = engine.evaluate(state, p8, flat_paper, t8 + pd.Timedelta(seconds=30))
    assert ev8_a == []
    t9 = base_time + pd.Timedelta(minutes=9)
    p9 = replace(p8, timestamp=t9)
    state, ev8_b = engine.evaluate(state, p9, flat_paper, t9 + pd.Timedelta(seconds=30))
    # Confirms because structural expansion condition (>= 1.0 ATR) is satisfied
    assert len(ev8_b) == 1
    assert ev8_b[0]["event_type"] == "setup_confirmed"
    assert ev8_b[0]["snapshot"]["entry"] == 70300.0


def test_post_tp_cooldown_prevents_immediate_same_direction_reentry():
    """Verify that after a take-profit exit, same-direction setups are throttled for the TP cooldown period."""
    engine = SignalLifecycle(confirm_observations=2, tp_rearm_seconds=7200)
    state = engine.initial_state()
    base_time = pd.Timestamp("2026-09-01 21:00:00", tz="UTC")

    # 1. Simulate a trade closing at target (Take Profit)
    paper_tp = {
        "open_position": None,
        "pending_order": None,
        "newly_closed": [
            {
                "side": "short",
                "entry": 76500.0,
                "exit": 74970.0,
                "exit_reason": "target",
                "zone": "vwap_lower:short1",
                "exit_time": base_time.isoformat(),
            }
        ],
    }
    # Pass a neutral update while closing
    state, ev = engine.evaluate(
        state,
        prediction(timestamp=base_time, bias="neutral", entry=None, stop=None, target=None, position_size=None),
        paper_tp,
        base_time + pd.Timedelta(seconds=30),
    )
    assert ev == []
    assert state["last_exit_by_bias"]["bearish"]["exit_reason"] == "target"

    # 2. Next bar (21:01): Scanner outputs a new Bearish setup at the bottom of the move ($74,980)
    flat_paper = {"open_position": None, "pending_order": None, "closed_trades": 1}
    p_next = prediction(
        timestamp=base_time + pd.Timedelta(minutes=1),
        bias="bearish",
        setup_type="continuation",
        zone="session_low:short2",
        sweep_time="2026-09-01T20:59:00Z",
        entry=74980.0,
        stop=75729.8,
        target=73480.4,
        setup_atr=250.0,
    )
    state, ev = engine.evaluate(state, p_next, flat_paper, base_time + pd.Timedelta(minutes=1, seconds=30))
    # Must be blocked by post-TP cooldown!
    assert ev == []
    assert state["candidate"] is None

    # Even with repeated 1-minute polls in the next 30 minutes, it remains blocked
    for m in range(2, 30):
        p_drift = replace(p_next, timestamp=base_time + pd.Timedelta(minutes=m), zone=f"zone_{m}")
        state, ev = engine.evaluate(state, p_drift, flat_paper, base_time + pd.Timedelta(minutes=m, seconds=30))
        assert ev == []
        assert state["candidate"] is None

    # 3. An opposite Bullish setup (reversal bounce) is NOT blocked
    t_bounce = base_time + pd.Timedelta(minutes=35)
    p_bull = prediction(
        timestamp=t_bounce,
        bias="bullish",
        setup_type="reversal",
        zone="equal_lows:bottom",
        sweep_time="2026-09-01T21:34:00Z",
        entry=75100.0,
        stop=74349.0,
        target=76602.0,
        setup_atr=250.0,
    )
    state, ev = engine.evaluate(state, p_bull, flat_paper, t_bounce + pd.Timedelta(seconds=30))
    assert ev == []
    p_bull2 = replace(p_bull, timestamp=t_bounce + pd.Timedelta(minutes=1))
    state, ev = engine.evaluate(state, p_bull2, flat_paper, t_bounce + pd.Timedelta(minutes=1, seconds=30))
    assert len(ev) == 1
    assert ev[0]["event_type"] == "setup_confirmed"
    assert ev[0]["snapshot"]["bias"] == "bullish"

    # 4. After 2.5 hours (elapsed > 7200s), a fresh Bearish setup is allowed to confirm
    t_later = base_time + pd.Timedelta(hours=2, minutes=30)
    p_later = prediction(
        timestamp=t_later,
        bias="bearish",
        setup_type="reversal",
        zone="previous_day_high:top",
        sweep_time="2026-09-01T23:28:00Z",
        entry=76800.0,
        stop=77568.0,
        target=75264.0,
        setup_atr=250.0,
    )
    state, ev = engine.evaluate(state, p_later, flat_paper, t_later + pd.Timedelta(seconds=30))
    assert ev == []
    p_later2 = replace(p_later, timestamp=t_later + pd.Timedelta(minutes=1))
    state, ev = engine.evaluate(state, p_later2, flat_paper, t_later + pd.Timedelta(minutes=1, seconds=30))
    assert len(ev) == 1
    assert ev[0]["event_type"] == "setup_confirmed"
    assert ev[0]["snapshot"]["bias"] == "bearish"


