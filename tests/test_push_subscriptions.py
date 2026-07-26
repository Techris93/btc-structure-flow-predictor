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
    assert "accepted · awaiting device" in dashboard
    assert '" · delivered"' not in dashboard


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
