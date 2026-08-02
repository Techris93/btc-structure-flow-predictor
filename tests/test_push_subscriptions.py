import app as web_app
from app import (
    _delivery_ack_token,
    _is_allowed_push_endpoint,
    _remove_subscription,
    _upsert_subscription,
)
import base64
import json
from pathlib import Path
import time
import pandas as pd


def _encoded(value):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _subscription(endpoint="https://web.push.apple.com/subscription"):
    return {
        "endpoint": endpoint,
        "keys": {
            "auth": _encoded(b"a" * 16),
            "p256dh": _encoded(b"\x04" + b"p" * 64),
        },
        "installation_id": "installation-1",
    }


def test_existing_push_endpoint_replaces_rotated_encryption_keys():
    subscriptions = [
        {"endpoint": "https://push.example/subscription", "keys": {"auth": "old", "p256dh": "old"}}
    ]
    replacement = {
        "endpoint": "https://push.example/subscription",
        "keys": {"auth": "new", "p256dh": "new"},
    }

    assert _upsert_subscription(subscriptions, replacement) is False
    assert len(subscriptions) == 1
    assert subscriptions[0]["endpoint"] == replacement["endpoint"]
    assert subscriptions[0]["keys"] == replacement["keys"]
    assert subscriptions[0]["last_seen_at"]


def test_new_push_endpoint_is_appended():
    subscriptions = []
    subscription = {"endpoint": "https://push.example/new", "keys": {"auth": "a", "p256dh": "p"}}

    assert _upsert_subscription(subscriptions, subscription) is True
    assert len(subscriptions) == 1
    assert subscriptions[0]["endpoint"] == subscription["endpoint"]
    assert subscriptions[0]["keys"] == subscription["keys"]


def test_new_endpoint_replaces_same_installation_instead_of_accumulating():
    subscriptions = []
    first = _subscription("https://web.push.apple.com/old")
    replacement = _subscription("https://web.push.apple.com/new")

    assert _upsert_subscription(subscriptions, first) is True
    assert _upsert_subscription(subscriptions, replacement) is False

    assert len(subscriptions) == 1
    assert subscriptions[0]["endpoint"] == replacement["endpoint"]


def test_remove_subscription_deletes_matching_endpoint_only():
    subscriptions = [
        {"endpoint": "https://push.example/keep", "keys": {"auth": "a", "p256dh": "p"}},
        {"endpoint": "https://push.example/drop", "keys": {"auth": "b", "p256dh": "q"}},
    ]

    assert _remove_subscription(subscriptions, "https://push.example/drop") == 1
    assert subscriptions == [
        {"endpoint": "https://push.example/keep", "keys": {"auth": "a", "p256dh": "p"}}
    ]
    assert _remove_subscription(subscriptions, "https://push.example/missing") == 0


def test_webpush_receives_persistent_pem_file_path(monkeypatch):
    captured = {}

    def fake_webpush(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(web_app, "webpush", fake_webpush)
    subscription = {"endpoint": "https://push.example/new", "keys": {"auth": "a", "p256dh": "p"}}

    assert web_app._send_push({"title": "test"}, [subscription]) == (1, 0)
    assert captured["vapid_private_key"] == str(web_app.vapid_path)
    assert captured["vapid_private_key"].endswith("vapid_private.pem")
    assert captured["ttl"] == 900
    assert captured["timeout"] == 10
    assert captured["headers"] == {"Urgency": "high"}
    payload = json.loads(captured["data"])
    assert payload["web_push"] == 8030
    assert payload["notification"]["title"] == "test"
    assert payload["notification"]["navigate"].startswith("https://")
    assert payload["notification"]["silent"] is False


def test_dashboard_restores_enabled_push_state_after_reload():
    dashboard = (Path(__file__).parents[1] / "templates" / "dashboard.html").read_text()

    assert "async function syncPushButton()" in dashboard
    assert 'await navigator.serviceWorker.register("/sw.js")' in dashboard
    assert "await reg.update()" in dashboard
    assert 'fetch("/push/subscribe"' in dashboard
    assert 'id="lastpush"' in dashboard
    assert '"Repair notifications"' in dashboard
    assert '"Pause notifications"' in dashboard
    assert "async function togglePush()" in dashboard
    assert 'fetch("/push/unsubscribe"' in dashboard
    assert "syncPushButton();" in dashboard
    assert "subscriptionPayload(sub)" in dashboard
    assert "window.pushManager || registration.pushManager" in dashboard
    assert 'localStorage.getItem("btc-flow-installation-id")' in dashboard
    assert "const lastDelivery = push.last_automatic_delivery || {};" in dashboard
    assert "Last push" in dashboard
    assert "display not confirmed" not in dashboard
    assert "Apple accepted" not in dashboard
    assert 'lastDelivery.delivery_type === "test"' not in dashboard
    assert '" · delivered"' not in dashboard


def test_healthz_is_a_side_effect_free_liveness_probe(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("/healthz must not start loops or inspect stores")

    monkeypatch.setattr(web_app, "start_live_loop", fail_if_called)
    monkeypatch.setattr(web_app, "trade_store", fail_if_called)

    response = web_app.app.test_client().get("/healthz")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "ok",
        "service": "btc-structure-flow-predictor",
        "process_alive": True,
    }


def test_live_loop_diagnostics_detects_a_stalled_poll(monkeypatch):
    now = time.monotonic()
    monkeypatch.setenv("LIVE_POLL_SECONDS", "20")
    monkeypatch.setattr(web_app, "live_loop_started_at", "2026-08-02T00:00:00+00:00")
    monkeypatch.setattr(web_app, "live_loop_started_monotonic", now - 100)
    monkeypatch.setattr(web_app, "live_loop_last_completed_monotonic", now - 100)
    monkeypatch.setattr(web_app, "live_thread", None)

    diagnostics = web_app._live_loop_diagnostics()

    assert diagnostics["stale"] is True
    assert diagnostics["stale_after_seconds"] == 60
    assert diagnostics["last_completed_age_seconds"] >= 99


def test_latest_delivery_summaries_separate_automatic_and_test(monkeypatch, tmp_path):
    events_store = web_app.JsonStore(tmp_path / "events.json")
    events_store.write([
        {
            "delivery_id": "auto-delivery",
            "batch_id": "auto-batch",
            "delivery_type": "automatic",
            "created_at": "2026-07-26T12:00:00+00:00",
            "accepted_at": "2026-07-26T12:00:01+00:00",
            "failed_at": None,
            "received_at": None,
            "notification_created_at": None,
            "retry_count": 0,
        },
        {
            "delivery_id": "test-delivery",
            "batch_id": "test-batch",
            "delivery_type": "test",
            "created_at": "2026-07-26T13:00:00+00:00",
            "accepted_at": "2026-07-26T13:00:01+00:00",
            "failed_at": None,
            "received_at": None,
            "notification_created_at": None,
            "retry_count": 0,
        },
    ])
    monkeypatch.setattr(web_app, "push_delivery_events_store", events_store)

    automatic = web_app._latest_delivery_summary("automatic")
    test = web_app._latest_delivery_summary("test")

    assert automatic["batch_id"] == "auto-batch"
    assert automatic["delivery_type"] == "automatic"
    assert automatic["subscriptions"] == 1
    assert test["batch_id"] == "test-batch"
    assert test["delivery_type"] == "test"


def test_subscription_sync_preserves_retired_endpoint_delivery_history(monkeypatch, tmp_path):
    current = _subscription("https://web.push.apple.com/current")
    retired = _subscription("https://web.push.apple.com/retired")
    events_store = web_app.JsonStore(tmp_path / "events.json")
    events_store.write([
        {
            "delivery_id": "keep",
            "endpoint_hash": web_app._endpoint_hash(current["endpoint"]),
        },
        {
            "delivery_id": "drop",
            "endpoint_hash": web_app._endpoint_hash(retired["endpoint"]),
        },
    ])
    monkeypatch.setattr(web_app, "push_subscriptions", [current])
    monkeypatch.setattr(web_app, "push_delivery_events_store", events_store)
    monkeypatch.setattr(
        web_app,
        "subscription_store",
        web_app.JsonStore(tmp_path / "subscriptions.json"),
    )
    monkeypatch.setattr(web_app, "PUSH_SINGLE_INSTALLATION", True)

    response = web_app.app.test_client().post("/push/subscribe", json=current)

    assert response.status_code == 200
    assert [event["delivery_id"] for event in events_store.read([])] == ["keep", "drop"]


def test_tp_notification_is_immediate_durable_and_deduplicated(monkeypatch, tmp_path):
    exit_store = web_app.JsonStore(tmp_path / "paper-exit-push.json")
    decision_store = web_app.JsonStore(tmp_path / "push-decisions.json")
    monkeypatch.setattr(web_app, "paper_exit_push_store", exit_store)
    monkeypatch.setattr(web_app, "push_decision_events_store", decision_store)
    # Paper exits must remain immediate even while generic state pushes are throttled.
    monkeypatch.setattr(web_app, "PUSH_STATE_COOLDOWN_SECONDS", 3600)
    submissions = []

    def fake_send(payload, subscriptions=None, delivery_type="automatic"):
        submissions.append((payload, delivery_type))
        return 1, 0

    monkeypatch.setattr(web_app, "_send_push", fake_send)
    trade = {
        "entry_time": "2026-07-26T10:00:00+00:00",
        "exit_time": "2026-07-26T11:00:00+00:00",
        "side": "long",
        "entry": 65000.0,
        "exit": 66000.0,
        "size": 0.25,
        "pnl": 250.0,
        "r_multiple": 2.0,
        "exit_reason": "target",
    }

    assert web_app._notify_paper_exits([trade]) == 1
    assert web_app._notify_paper_exits([trade]) == 0
    assert len(submissions) == 1
    payload, delivery_type = submissions[0]
    assert payload["title"] == "BTC paper trade · Target hit"
    assert "P&L +$250.00" in payload["body"]
    assert delivery_type == "automatic"
    assert exit_store.read({})["pending"] == []
    assert len(exit_store.read({})["notified_ids"]) == 1
    assert decision_store.read([])[-1]["status"] == "accepted"


def test_failed_sl_notification_remains_pending_for_next_poll(monkeypatch, tmp_path):
    exit_store = web_app.JsonStore(tmp_path / "paper-exit-push.json")
    decision_store = web_app.JsonStore(tmp_path / "push-decisions.json")
    monkeypatch.setattr(web_app, "paper_exit_push_store", exit_store)
    monkeypatch.setattr(web_app, "push_decision_events_store", decision_store)
    outcomes = iter([(0, 1), (1, 0)])
    monkeypatch.setattr(
        web_app,
        "_send_push",
        lambda payload, subscriptions=None, delivery_type="automatic": next(outcomes),
    )
    trade = {
        "entry_time": "2026-07-26T10:00:00+00:00",
        "exit_time": "2026-07-26T10:15:00+00:00",
        "side": "short",
        "entry": 65000.0,
        "exit": 65250.0,
        "size": 1.0,
        "pnl": -250.0,
        "r_multiple": -1.0,
        "exit_reason": "stop",
    }

    assert web_app._notify_paper_exits([trade]) == 0
    assert len(exit_store.read({})["pending"]) == 1
    assert web_app._notify_paper_exits([]) == 1
    assert exit_store.read({})["pending"] == []


def test_lifecycle_push_queue_deduplicates_and_respects_safety_cooldown(
    monkeypatch, tmp_path
):
    queue_store = web_app.JsonStore(tmp_path / "signal-events.json")
    decision_store = web_app.JsonStore(tmp_path / "push-decisions.json")
    monkeypatch.setattr(web_app, "signal_event_queue_store", queue_store)
    monkeypatch.setattr(web_app, "push_decision_events_store", decision_store)
    monkeypatch.setattr(
        web_app,
        "_latest_delivery_summary",
        lambda delivery_type: {
            "batch_id": "batch-1",
            "attempted": 1,
            "accepted": 1,
            "failed": 0,
        },
    )
    sent = []
    monkeypatch.setattr(
        web_app,
        "_send_push",
        lambda payload, subscriptions=None, delivery_type="automatic": (
            sent.append(payload) or 1,
            0,
        ),
    )
    event = {
        "event_id": "lifecycle-1-setup_confirmed-abc",
        "event_type": "setup_confirmed",
        "signal_id": "abc",
        "title": "BTC setup confirmed",
        "body": "Bullish reversal confirmed",
    }
    assert web_app._enqueue_signal_events([event, event]) == 1
    assert web_app.PUSH_STATE_COOLDOWN_SECONDS == 60

    now = pd.Timestamp("2026-07-28T00:01:00Z")
    queue_store.write({
        **queue_store.read({}),
        "last_generic_push_at": (now - pd.Timedelta(seconds=30)).isoformat(),
    })
    assert web_app._dispatch_signal_event(now) == 0
    assert len(queue_store.read({})["pending"]) == 1
    assert sent == []

    assert web_app._dispatch_signal_event(now + pd.Timedelta(seconds=30)) == 1
    assert queue_store.read({})["pending"] == []
    assert queue_store.read({})["notified_ids"] == [event["event_id"]]
    assert len(sent) == 1


def test_failed_lifecycle_push_remains_durable(monkeypatch, tmp_path):
    queue_store = web_app.JsonStore(tmp_path / "signal-events.json")
    decision_store = web_app.JsonStore(tmp_path / "push-decisions.json")
    monkeypatch.setattr(web_app, "signal_event_queue_store", queue_store)
    monkeypatch.setattr(web_app, "push_decision_events_store", decision_store)
    monkeypatch.setattr(
        web_app,
        "_latest_delivery_summary",
        lambda delivery_type: {
            "batch_id": "failed-batch",
            "attempted": 1,
            "accepted": 0,
            "failed": 1,
        },
    )
    monkeypatch.setattr(
        web_app,
        "_send_push",
        lambda payload, subscriptions=None, delivery_type="automatic": (0, 1),
    )
    event = {
        "event_id": "lifecycle-2-setup_invalidated-abc",
        "event_type": "setup_invalidated",
        "signal_id": "abc",
        "title": "BTC setup invalidated",
        "body": "Bullish setup invalidated",
    }
    web_app._enqueue_signal_events([event])

    assert web_app._dispatch_signal_event(pd.Timestamp("2026-07-28T00:01:00Z")) == 0
    state = queue_store.read({})
    assert [item["event_id"] for item in state["pending"]] == [event["event_id"]]
    assert state["last_generic_push_at"] is None


def test_terminal_lifecycle_event_supersedes_stale_pending_setup(
    monkeypatch, tmp_path
):
    queue_store = web_app.JsonStore(tmp_path / "signal-events.json")
    decision_store = web_app.JsonStore(tmp_path / "push-decisions.json")
    monkeypatch.setattr(web_app, "signal_event_queue_store", queue_store)
    monkeypatch.setattr(web_app, "push_decision_events_store", decision_store)
    setup = {
        "event_id": "setup-abc",
        "event_type": "setup_confirmed",
        "signal_id": "abc",
        "title": "BTC setup confirmed",
        "body": "Bullish reversal confirmed",
    }
    invalidated = {
        "event_id": "invalidated-abc",
        "event_type": "setup_invalidated",
        "signal_id": "abc",
        "title": "BTC setup invalidated",
        "body": "Bullish setup invalidated",
    }
    web_app._enqueue_signal_events([setup])

    assert web_app._enqueue_signal_events([invalidated]) == 1
    state = queue_store.read({})
    assert [item["event_id"] for item in state["pending"]] == ["invalidated-abc"]
    decisions = decision_store.read([])
    old = next(item for item in decisions if item["decision_id"] == "setup-abc")
    assert old["status"] == "superseded"


def test_replacement_setup_supersedes_old_pending_setup(monkeypatch, tmp_path):
    queue_store = web_app.JsonStore(tmp_path / "signal-events.json")
    decision_store = web_app.JsonStore(tmp_path / "push-decisions.json")
    monkeypatch.setattr(web_app, "signal_event_queue_store", queue_store)
    monkeypatch.setattr(web_app, "push_decision_events_store", decision_store)
    old = {
        "event_id": "setup-old",
        "event_type": "setup_confirmed",
        "signal_id": "old",
        "title": "BTC setup confirmed",
        "body": "Bullish reversal confirmed",
    }
    new = {
        "event_id": "setup-new",
        "event_type": "setup_confirmed",
        "signal_id": "new",
        "replaced_signal_id": "old",
        "title": "BTC setup confirmed",
        "body": "Bearish reversal confirmed",
    }
    web_app._enqueue_signal_events([old])

    assert web_app._enqueue_signal_events([new]) == 1
    assert [
        item["event_id"] for item in queue_store.read({})["pending"]
    ] == ["setup-new"]


def test_push_unsubscribe_endpoint_removes_subscription(monkeypatch):
    subscription = _subscription("https://web.push.apple.com/toggle")
    monkeypatch.setattr(web_app, "push_subscriptions", [subscription])
    client = web_app.app.test_client()

    registered = client.post("/push/subscribe", json=subscription).get_json()
    assert registered["ok"] is True

    paused = client.post(
        "/push/unsubscribe",
        json={"endpoint": subscription["endpoint"], "test_token": registered["test_token"]},
    ).get_json()

    assert paused["ok"] is True
    assert paused["removed"] == 1
    assert web_app.push_subscriptions == []


def test_subscribe_rejects_non_push_endpoint():
    client = web_app.app.test_client()
    subscription = _subscription("https://127.0.0.1/internal")

    response = client.post("/push/subscribe", json=subscription)

    assert response.status_code == 400
    assert response.get_json()["error"] == "unsupported push endpoint"


def test_single_installation_registration_removes_legacy_endpoints(monkeypatch, tmp_path):
    current = _subscription("https://web.push.apple.com/current")
    legacy = _subscription("https://web.push.apple.com/legacy")
    legacy["installation_id"] = "legacy-installation"
    monkeypatch.setattr(web_app, "push_subscriptions", [legacy])
    monkeypatch.setattr(web_app, "subscription_store", web_app.JsonStore(tmp_path / "subscriptions.json"))
    monkeypatch.setattr(web_app, "PUSH_SINGLE_INSTALLATION", True)

    response = web_app.app.test_client().post("/push/subscribe", json=current)

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["single_installation"] is True
    assert payload["subscriptions"] == 1
    assert payload["current_verified"] is False
    assert [item["endpoint"] for item in web_app.push_subscriptions] == [current["endpoint"]]


def test_known_push_service_hosts_are_allowed():
    assert _is_allowed_push_endpoint("https://web.push.apple.com/x")
    assert _is_allowed_push_endpoint("https://fcm.googleapis.com/x")
    assert _is_allowed_push_endpoint("https://updates.push.services.mozilla.com/x")
    assert not _is_allowed_push_endpoint("https://push.apple.com.attacker.example/x")


def test_acknowledgement_marks_delivery_and_subscription_verified(monkeypatch, tmp_path):
    subscription = _subscription("https://web.push.apple.com/ack")
    subscription["last_ack_at"] = None
    monkeypatch.setattr(web_app, "push_subscriptions", [subscription])
    monkeypatch.setattr(web_app, "push_delivery_events_store", web_app.JsonStore(tmp_path / "events.json"))
    monkeypatch.setattr(web_app, "push_delivery_store", web_app.JsonStore(tmp_path / "last.json"))
    monkeypatch.setattr(web_app, "subscription_store", web_app.JsonStore(tmp_path / "subscriptions.json"))
    delivery_id = "delivery-1"
    endpoint_hash = web_app._endpoint_hash(subscription["endpoint"])
    web_app.push_delivery_events_store.write([{
        "delivery_id": delivery_id,
        "batch_id": "batch-1",
        "endpoint_hash": endpoint_hash,
        "accepted_at": "2026-07-25T00:00:00+00:00",
        "failed_at": None,
        "received_at": None,
        "notification_created_at": None,
        "retry_count": 0,
    }])
    web_app.push_delivery_store.write({"batch_id": "batch-1"})
    token = _delivery_ack_token(delivery_id, endpoint_hash)

    response = web_app.app.test_client().post("/push/ack", json={
        "delivery_id": delivery_id,
        "ack_token": token,
        "status": "notification_created",
    })

    assert response.status_code == 200
    event = web_app.push_delivery_events_store.read([])[0]
    assert event["received_at"]
    assert event["notification_created_at"]
    assert web_app.push_subscriptions[0]["status"] == "verified"


def test_broadcast_test_requires_admin_token(monkeypatch):
    monkeypatch.delenv("PUSH_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("ADMIN_API_TOKEN", raising=False)

    response = web_app.app.test_client().post("/push/broadcast-test")

    assert response.status_code == 401


def test_dashboard_does_not_create_foreground_only_notifications():
    dashboard = (Path(__file__).parents[1] / "templates" / "dashboard.html").read_text()

    assert 'new Notification("BTC Predictor update"' not in dashboard


def test_service_worker_supports_background_subscription_rotation():
    client = web_app.app.test_client()
    response = client.get("/sw.js")
    script = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "pushsubscriptionchange" in script
    assert "showNotification" in script
    assert "clients.matchAll" in script
    assert "fetch('/push/ack'" in script
    assert "acknowledge('received')" in script
    assert "acknowledge('notification_created')" in script
    assert "previous_endpoint" in script
    assert "const proposed = data.notification || {}" in script


def test_delivery_record_exists_before_fast_device_ack(monkeypatch, tmp_path):
    subscription = _subscription("https://web.push.apple.com/fast-ack")
    monkeypatch.setattr(web_app, "push_subscriptions", [subscription])
    monkeypatch.setattr(web_app, "push_delivery_events_store", web_app.JsonStore(tmp_path / "events.json"))
    monkeypatch.setattr(web_app, "push_delivery_store", web_app.JsonStore(tmp_path / "last.json"))
    monkeypatch.setattr(web_app, "subscription_store", web_app.JsonStore(tmp_path / "subscriptions.json"))

    def fake_webpush(**kwargs):
        payload = json.loads(kwargs["data"])
        ok, error = web_app._ack_delivery(
            payload["delivery_id"],
            payload["ack_token"],
            "received",
        )
        assert ok is True, error

    monkeypatch.setattr(web_app, "webpush", fake_webpush)

    assert web_app._send_push({"title": "test"}, [subscription]) == (1, 0)
    assert web_app.push_delivery_store.read({})["received"] == 1


def test_manifest_has_stable_pwa_identity():
    manifest = web_app.app.test_client().get("/manifest.json").get_json()

    assert manifest["id"] == "/"
    assert manifest["display"] == "standalone"


def test_transient_gateway_failure_is_retried(monkeypatch, tmp_path):
    subscription = _subscription("https://web.push.apple.com/retry")
    now = pd.Timestamp("2026-07-25T20:00:00Z")
    endpoint_hash = web_app._endpoint_hash(subscription["endpoint"])
    event = {
        "delivery_id": "retry-delivery",
        "batch_id": "retry-batch",
        "delivery_type": "automatic",
        "endpoint_hash": endpoint_hash,
        "payload": {"title": "retry"},
        "accepted_at": None,
        "failed_at": (now - pd.Timedelta(minutes=1)).isoformat(),
        "received_at": None,
        "notification_created_at": None,
        "retry_count": 0,
        "next_retry_at": (now - pd.Timedelta(seconds=1)).isoformat(),
        "expires_at": (now + pd.Timedelta(minutes=10)).isoformat(),
        "http_status": 503,
        "error": "temporary failure",
    }
    monkeypatch.setattr(web_app, "push_subscriptions", [subscription])
    monkeypatch.setattr(web_app, "push_delivery_events_store", web_app.JsonStore(tmp_path / "events.json"))
    monkeypatch.setattr(web_app, "push_delivery_store", web_app.JsonStore(tmp_path / "last.json"))
    monkeypatch.setattr(web_app, "subscription_store", web_app.JsonStore(tmp_path / "subscriptions.json"))
    web_app.push_delivery_events_store.write([event])
    web_app.push_delivery_store.write({"batch_id": "retry-batch"})
    monkeypatch.setattr(web_app, "webpush", lambda **kwargs: None)

    assert web_app._retry_unacknowledged_pushes(now) == 1

    retried = web_app.push_delivery_events_store.read([])[0]
    assert retried["accepted_at"]
    assert retried["failed_at"] is None
    assert retried["retry_count"] == 1
