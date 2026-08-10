import app as web_app
import gunicorn_config


def test_post_worker_init_starts_live_loop_in_worker(monkeypatch):
    started = []
    monkeypatch.setenv("START_LIVE_LOOP_ON_BOOT", "1")
    monkeypatch.setattr(web_app, "start_live_boot_supervisor", lambda: started.append(True))

    gunicorn_config.post_worker_init(None)

    assert started == [True]


def test_post_worker_init_respects_disabled_boot(monkeypatch):
    started = []
    monkeypatch.setenv("START_LIVE_LOOP_ON_BOOT", "0")
    monkeypatch.setattr(web_app, "start_live_boot_supervisor", lambda: started.append(True))

    gunicorn_config.post_worker_init(None)

    assert started == []


def test_closed_bar_decision_is_immutable_and_emits_no_duplicate_events():
    gate = web_app.ClosedBarDecisionGate()
    bar_at = web_app.pd.Timestamp("2026-01-01 00:05:00Z")
    emitted = []

    def evaluate_strategy_and_lifecycle():
        emitted.append("setup_confirmed")
        return {"decision": "first"}

    first_is_new = gate.should_evaluate(bar_at)
    first = evaluate_strategy_and_lifecycle()
    gate.commit(bar_at, first)
    second_is_new = gate.should_evaluate(bar_at)
    second = gate.value

    assert first_is_new is True
    assert second_is_new is False
    assert second is first
    assert emitted == ["setup_confirmed"]
