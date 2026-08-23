"""Live governance: economics, soft filters, risk caps, fail-closed, shadow."""

from pathlib import Path

import pandas as pd

from btc_predictor import live_policy
from btc_predictor.models import PredictorOutput
from btc_predictor.paper_position import HISTORICAL_SEEDED_TRADES, PaperLedger
from btc_predictor.persistence import JsonStore
from btc_predictor.signal_lifecycle import SignalLifecycle


def test_seeded_rescore_reports_gross_vs_net():
    report = live_policy.rescore_seeded_trades()
    assert report["summary"]["count"] == 3
    assert report["summary"]["gross_pnl"] > report["summary"]["net_pnl"]
    assert report["summary"]["fees"] > 0
    assert report["summary"]["do_not_treat_gross_as_alpha"] is True
    # Gross book ~ +681; net is lower after fee+slip.
    assert report["summary"]["gross_pnl"] == 681.24 or abs(report["summary"]["gross_pnl"] - 681.24) < 0.1


def test_superseded_churn_is_marked_and_excluded_from_performance_stats(tmp_path):
    path = tmp_path / "paper_ledger.json"
    rows = [
        {
            "entry_time": "2026-08-01T00:00:00Z",
            "exit_time": "2026-08-01T01:00:00Z",
            "side": "long",
            "entry": 100.0,
            "exit": 102.0,
            "stop": 99.0,
            "target": 102.0,
            "size": 1.0,
            "exit_reason": "target",
        },
        {
            "entry_time": "2026-08-01T02:00:00Z",
            "exit_time": "2026-08-01T02:05:00Z",
            "side": "long",
            "entry": 100.0,
            "exit": 99.5,
            "stop": 99.0,
            "target": 102.0,
            "size": 1.0,
            "exit_reason": "superseded_by_confirmed_setup",
        },
        {
            "entry_time": "2026-08-01T03:00:00Z",
            "exit_time": "2026-08-01T04:00:00Z",
            "side": "long",
            "entry": 100.0,
            "exit": 99.0,
            "stop": 99.0,
            "target": 102.0,
            "size": 1.0,
            "exit_reason": "stop",
        },
    ]
    JsonStore(path).write({"closed": rows, "open": None, "pending": None, "equity_net": 100.0})

    ledger = PaperLedger(path, soft_filters=False, use_fixed_pct_exits=False)
    status = ledger._status()

    assert status["closed_trades"] == 3  # complete audit/accounting count
    assert status["performance_closed_trades"] == 2
    assert status["performance_excluded_trades"] == 1
    assert status["performance_excluded_superseded_churn"] == 1
    assert status["wins"] == 1
    assert status["losses"] == 1
    assert status["performance"]["closed_trades"] == 2
    assert status["accounting"]["closed_trades"] == 3
    churn = next(t for t in status["recent_closed"] if t["exit_reason"] == "superseded_by_confirmed_setup")
    assert churn["performance_eligible"] is False
    assert churn["performance_exclusion_reason"] == "superseded_churn"
    persisted = JsonStore(path).read({})
    persisted_churn = next(t for t in persisted["closed"] if t["exit_reason"] == "superseded_by_confirmed_setup")
    assert persisted_churn["performance_class"] == "excluded_superseded_churn"


def test_paper_ledger_reset_is_one_shot_and_archives_previous_book(tmp_path):
    path = tmp_path / "paper_ledger.json"
    JsonStore(path).write({"closed": [{"exit_reason": "stop"}], "equity": 99_000.0})

    fresh = PaperLedger(path, reset_id="post-churn-v1")
    assert fresh._status()["closed_trades"] == 0
    assert JsonStore(path).read({})["ledger_reset_id"] == "post-churn-v1"
    archives = list(tmp_path.glob("paper_ledger.archive-post-churn-v1*.json"))
    assert len(archives) == 1

    restarted = PaperLedger(path, reset_id="post-churn-v1")
    assert restarted._status()["closed_trades"] == 0
    assert len(list(tmp_path.glob("paper_ledger.archive-post-churn-v1*.json"))) == 1


def test_paper_ledger_uses_research_economics_on_close():
    ledger = PaperLedger(soft_filters=False, apply_research_costs=True, use_fixed_pct_exits=False)
    # Clear seeded history for a pure unit trade.
    ledger._closed = []
    ledger._equity = 100_000.0
    ledger._equity_gross = 100_000.0
    pred = PredictorOutput(
        timestamp=pd.Timestamp("2025-01-01", tz="UTC"),
        bias="bullish",
        entry=100.0,
        stop=95.0,
        target=110.0,
        position_size=1.0,
        reward_risk=2.0,
        zone="equal_lows:test",
        zone_kind="equal_lows",
        setup_type="reversal",
        sweep_status="confirmed",
        orderflow_confirmation=True,
    )
    ledger.update(pred)
    fill = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0]},
        index=[pred.timestamp + pd.Timedelta(minutes=1)],
    )
    ledger.update_market(fill)
    assert ledger._open is not None
    exit_bar = pd.DataFrame(
        {"open": [110.0], "high": [111.0], "low": [109.0], "close": [110.0]},
        index=[pred.timestamp + pd.Timedelta(minutes=2)],
    )
    status = ledger.update_market(exit_bar)
    assert status["closed_trades"] == 1
    trade = status["last_closed"]
    assert trade["gross_pnl"] > trade["net_pnl"]
    assert trade["fees"] > 0
    assert status["gross_pnl"] > status["net_pnl"]
    assert status["pnl_reporting"]["do_not_treat_gross_as_alpha"] is True


def test_soft_filter_skips_hero_rr_and_round_magnet_stop():
    snap = {
        "bias": "short",
        "entry": 64729.4,
        "stop": 64999.9,
        "target": 63678.8,
        "reward_risk": 3.88,
        "zone": "untested_breakout:x",
        "zone_kind": "untested_breakout",
    }
    result = live_policy.evaluate_soft_filters(snap, enabled=True)
    assert result["allow"] is False
    assert "stop_on_major_magnet" in result["hard_skips"] or any(
        s.startswith("planned_rr_above") for s in result["hard_skips"]
    )


def test_soft_filter_blocks_on_ledger_place():
    ledger = PaperLedger(soft_filters=True, use_fixed_pct_exits=False)
    ledger._closed = []
    ledger._equity = 100_000.0
    event = {
        "event_id": "e1",
        "event_type": "setup_confirmed",
        "signal_id": "s1",
        "created_at": "2025-01-01T00:00:00+00:00",
        "snapshot": {
            "timestamp": "2025-01-01T00:00:00+00:00",
            "bias": "bearish",
            "entry": 64729.4,
            "stop": 64999.9,
            "target": 63678.8,
            "position_size": 1.0,
            "reward_risk": 3.88,
            "zone": "untested_breakout:x",
            "zone_kind": "untested_breakout",
            "entry_type": "market",
        },
    }
    status = ledger.apply_lifecycle([event])
    assert ledger._open is None and ledger._pending is None
    assert ledger._last_reject and ledger._last_reject["reason"] == "soft_filter"
    assert status["entry_status"]["state"] == "rejected"
    assert status["entry_status"]["signal_id"] == "s1"
    assert status["open_unrealized_pnl"] is None


def test_open_fill_reports_transition_and_persists_mark_to_market(tmp_path):
    path = tmp_path / "paper-ledger.json"
    ledger = PaperLedger(
        path,
        soft_filters=False,
        apply_research_costs=False,
        use_fixed_pct_exits=False,
        fill_min_rr=0.0,
    )
    ledger._closed = []
    ledger._equity = 100_000.0
    ledger._equity_gross = 100_000.0
    event = {
        "event_id": "open-transition",
        "event_type": "setup_confirmed",
        "signal_id": "signal-open-1",
        "created_at": "2026-08-23T00:00:00+00:00",
        "snapshot": {
            "timestamp": "2026-08-23T00:00:00+00:00",
            "bias": "bullish",
            "entry": 100.0,
            "stop": 95.0,
            "target": 110.0,
            "position_size": 1.0,
            "reward_risk": 2.0,
            "zone": "equal_lows:test",
            "zone_kind": "equal_lows",
            "entry_type": "market",
        },
    }

    accepted = ledger.apply_lifecycle([event])
    assert accepted["entry_status"]["state"] == "pending"
    assert accepted["open_unrealized_pnl"] is None

    bar = pd.DataFrame(
        {"open": [101.0], "high": [103.0], "low": [100.0], "close": [102.0]},
        index=[pd.Timestamp("2026-08-23T00:01:00Z")],
    )
    filled = ledger.update_market(bar)

    assert filled["entry_status"]["state"] == "open"
    assert filled["open_position"]["signal_id"] == "signal-open-1"
    assert [row["signal_id"] for row in filled["newly_opened"]] == ["signal-open-1"]
    assert filled["open_unrealized_pnl"] is not None

    restarted = PaperLedger(
        path,
        soft_filters=False,
        apply_research_costs=False,
        use_fixed_pct_exits=False,
        fill_min_rr=0.0,
    )
    restored = restarted._status()
    assert restored["entry_status"]["state"] == "open"
    assert restored["open_unrealized_pnl"] == filled["open_unrealized_pnl"]


def test_notional_cap_reduces_size():
    caps = live_policy.apply_risk_caps(
        entry=50_000.0,
        stop=49_999.0,  # $1 risk/unit → huge size without cap
        size=10.0,
        equity=100_000.0,
        max_notional_multiple=1.5,
    )
    assert caps["allow"] is True
    assert caps["notional"] <= 100_000.0 * 1.5 + 1e-6
    assert "notional_capped" in caps["reasons"]


def test_one_open_risk_unit():
    caps = live_policy.apply_risk_caps(
        entry=100.0, stop=95.0, size=1.0, equity=100_000.0, has_open_or_pending=True
    )
    assert caps["allow"] is False
    assert "one_open_risk_unit" in caps["reasons"]


def test_data_quality_fail_closed_on_stale_or_spot():
    ok = live_policy.evaluate_data_quality(
        market_type="linear",
        binance_feed_mode="linear",
        stale_exchanges=[],
        collectors={"binance": {"mode": "linear"}, "bybit": {"mode": "linear"}},
    )
    assert ok["tradable"] is True
    bad = live_policy.evaluate_data_quality(
        market_type="linear",
        binance_feed_mode="spot_market_data",
        stale_exchanges=["binance"],
        collectors={"binance": {"mode": "spot"}, "bybit": {"mode": "linear"}},
    )
    assert bad["tradable"] is False
    assert bad["research_only"] is True


def test_decision_snapshot_enrichment_fields():
    snap = live_policy.enrich_decision_snapshot({
        "timestamp": "2026-07-28T12:00:00+00:00",
        "bias": "bearish",
        "entry": 65000.0,
        "stop": 65200.0,
        "target": 64600.0,
        "reward_risk": 2.0,
        "zone": "vwap_lower:abc",
        "zone_kind": "vwap_lower",
        "regime_4h": "bearish",
        "regime_1h": "bearish",
        "setup_15m": "bearish",
        "sweep_depth_atr": 0.4,
        "market_flow_score": 0.5,
        "raw_footprint_score": 0.55,
        "flow_gate_mode": "independent",
        "probability_tp_before_sl": 0.6,
        "entry_type": "market",
    })
    assert snap["stop_distance_pct"] is not None
    assert "stop_magnets" in snap
    assert snap["probability_source"] == live_policy.PROBABILITY_SOURCE
    assert snap["probability_use"] == live_policy.PROBABILITY_USE
    assert snap["soft_expectancy_is_diagnostic"] is True
    assert snap["decision_bar"] == snap["timestamp"]


def test_lifecycle_snapshot_includes_geometry():
    engine = SignalLifecycle()
    pred = PredictorOutput(
        timestamp=pd.Timestamp("2026-07-28T12:00:00Z"),
        bias="bullish",
        setup_type="reversal",
        zone="equal_lows:abc",
        zone_kind="equal_lows",
        sweep_status="confirmed",
        entry=65000.0,
        stop=64750.0,
        target=65500.0,
        reward_risk=2.0,
        probability_tp_before_sl=0.66,
        position_size=0.4,
        regime_4h="bullish",
        regime_1h="bullish",
        setup_15m="bullish",
        sweep_depth_atr=0.3,
        market_flow_score=0.5,
        raw_footprint_score=0.5,
        flow_gate_mode="independent",
        setup_atr=250.0,
    )
    snap = engine.snapshot(pred)
    assert snap["zone_kind"] == "equal_lows"
    assert snap["sweep_depth_atr"] == 0.3
    assert snap["stop_distance"] is not None
    assert snap["probability_tp_before_sl_is_heuristic"] is True


def test_lifecycle_rejects_same_direction_replacement_while_active():
    """Option A: Same-direction setups are diagnostic only while an active signal is held."""
    engine = SignalLifecycle(confirm_observations=2, replacement_distance_atr=0.25)
    base = PredictorOutput(
        timestamp=pd.Timestamp("2026-07-28T12:00:00Z"),
        bias="bullish",
        setup_type="reversal",
        zone="equal_lows:abc",
        zone_kind="equal_lows",
        sweep_status="confirmed",
        sweep_time="2026-07-28T11:58:00+00:00",
        reclaim_time="2026-07-28T11:59:00+00:00",
        orderflow_confirmation=True,
        entry=65000.0,
        stop=64750.0,
        target=65500.0,
        reward_risk=2.0,
        probability_tp_before_sl=0.9,
        position_size=0.4,
        setup_atr=250.0,
    )
    state, _ = engine.evaluate(engine.initial_state(), base, {}, pd.Timestamp("2026-07-28T12:00:30Z"))
    state, events = engine.evaluate(
        state,
        PredictorOutput(**{**base.__dict__, "timestamp": pd.Timestamp("2026-07-28T12:01:00Z")}),
        {},
        pd.Timestamp("2026-07-28T12:01:30Z"),
    )
    assert any(e["event_type"] == "setup_confirmed" for e in events)
    # Same direction (bullish) setup does NOT replace active signal
    far = PredictorOutput(
        **{
            **base.__dict__,
            "timestamp": pd.Timestamp("2026-07-28T12:02:00Z"),
            "zone": "previous_day_low:far",
            "sweep_time": "2026-07-28T12:01:00+00:00",
            "entry": 65100.0,
            "probability_tp_before_sl": 0.2,
            "reward_risk": 1.6,
        }
    )
    state, events = engine.evaluate(state, far, {}, pd.Timestamp("2026-07-28T12:02:30Z"))
    assert events == []
    far2 = PredictorOutput(**{**far.__dict__, "timestamp": pd.Timestamp("2026-07-28T12:03:00Z")})
    state, events = engine.evaluate(state, far2, {}, pd.Timestamp("2026-07-28T12:03:30Z"))
    assert events == []
    assert state["active"]["snapshot"]["zone"] == "equal_lows:abc"

    # An opposite-direction (bearish) confirmed setup DOES replace and flips
    bear = PredictorOutput(
        **{
            **base.__dict__,
            "timestamp": pd.Timestamp("2026-07-28T12:04:00Z"),
            "bias": "bearish",
            "zone": "previous_day_high:top",
            "sweep_time": "2026-07-28T12:03:00+00:00",
            "entry": 65600.0,
            "stop": 65850.0,
            "target": 65100.0,
        }
    )
    state, events = engine.evaluate(state, bear, {}, pd.Timestamp("2026-07-28T12:04:30Z"))
    assert events == []
    bear2 = PredictorOutput(**{**bear.__dict__, "timestamp": pd.Timestamp("2026-07-28T12:05:00Z")})
    state, events = engine.evaluate(state, bear2, {}, pd.Timestamp("2026-07-28T12:05:30Z"))
    assert len(events) == 1
    assert events[0]["event_type"] == "setup_confirmed"
    assert events[0]["bias_reversal"] is True


def test_funnel_and_shadow_stores(tmp_path):
    funnel = live_policy.FunnelDiary(JsonStore(tmp_path / "funnel.json"))
    pred = PredictorOutput(
        timestamp=pd.Timestamp("2026-07-28T12:00:00Z"),
        bias="neutral",
        no_trade_reason="timeframe_conflict",
    )
    funnel.record_prediction(pred)
    status = funnel.status()
    assert status["counts"]["bias_neutral"] >= 1
    assert status["counts"]["decision_bars"] >= 1

    shadow = live_policy.ShadowBook(JsonStore(tmp_path / "shadow.json"), rule="skip_planned_rr_above_2_5")
    shadow.observe_confirmed({
        "signal_id": "s1",
        "event_id": "e1",
        "snapshot": {
            "entry": 100.0, "stop": 95.0, "target": 110.0,
            "reward_risk": 3.0, "bias": "bullish", "zone": "z",
        },
    })
    st = shadow.status()
    assert st["counts"]["book_b_skipped"] == 1
    assert st["forward_only"] is True


def test_calibration_status_without_artifact():
    status = live_policy.calibration_status(None, {"gate_mode": "independent", "market_threshold": 0.4, "raw_threshold": 0.4})
    assert status["present"] is False
    assert "stay_on_independent" in status["action"]


def test_retune_discipline_blocks_early():
    status = live_policy.retune_discipline_status(3, policy_effective_at="2026-08-12T00:00:00+00:00")
    assert status["parameter_changes_allowed"] is False
    assert status["trades_remaining"] > 0
    assert "expectancy_r_after_costs" in status["review_metrics_only"]


def test_write_seeded_rescore_report(tmp_path):
    path = tmp_path / "seeded_trade_rescore.json"
    report = live_policy.write_seeded_rescore_report(path)
    assert path.exists()
    assert report["summary"]["count"] == 3
    assert Path(path).read_text()


def test_seeded_trades_list_unchanged_count():
    assert len(HISTORICAL_SEEDED_TRADES) == 3
