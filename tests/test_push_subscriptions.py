import app as web_app
from app import _remove_subscription, _upsert_subscription
from pathlib import Path


def test_existing_push_endpoint_replaces_rotated_encryption_keys():
    subscriptions = [
        {"endpoint": "https://push.example/subscription", "keys": {"auth": "old", "p256dh": "old"}}
    ]
    replacement = {
        "endpoint": "https://push.example/subscription",
        "keys": {"auth": "new", "p256dh": "new"},
    }

    assert _upsert_subscription(subscriptions, replacement) is False
    assert subscriptions == [replacement]


def test_new_push_endpoint_is_appended():
    subscriptions = []
    subscription = {"endpoint": "https://push.example/new", "keys": {"auth": "a", "p256dh": "p"}}

    assert _upsert_subscription(subscriptions, subscription) is True
    assert subscriptions == [subscription]


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
    assert captured["ttl"] == 86_400
    assert captured["timeout"] == 10
    assert captured["headers"] == {"Urgency": "high", "Topic": "btc-structure-flow"}


def test_dashboard_restores_enabled_push_state_after_reload():
    dashboard = (Path(__file__).parents[1] / "templates" / "dashboard.html").read_text()

    assert "async function syncPushButton()" in dashboard
    assert 'await navigator.serviceWorker.register("/sw.js")' in dashboard
    assert "await reg.update()" in dashboard
    assert 'fetch("/push/subscribe"' in dashboard
    assert 'id="lastpush"' in dashboard
    assert 'b.textContent = sub ? "Pause notifications" : "Enable push notifications"' in dashboard
    assert "async function togglePush()" in dashboard
    assert 'fetch("/push/unsubscribe"' in dashboard
    assert "syncPushButton();" in dashboard


def test_push_unsubscribe_endpoint_removes_subscription(monkeypatch):
    subscription = {
        "endpoint": "https://push.example/toggle",
        "keys": {"auth": "a", "p256dh": "p"},
    }
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
