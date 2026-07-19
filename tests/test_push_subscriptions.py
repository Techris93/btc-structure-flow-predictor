import app as web_app
from app import _upsert_subscription
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


def test_webpush_receives_persistent_pem_file_path(monkeypatch):
    captured = {}

    def fake_webpush(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(web_app, "webpush", fake_webpush)
    subscription = {"endpoint": "https://push.example/new", "keys": {"auth": "a", "p256dh": "p"}}

    assert web_app._send_push({"title": "test"}, [subscription]) == (1, 0)
    assert captured["vapid_private_key"] == str(web_app.vapid_path)
    assert captured["vapid_private_key"].endswith("vapid_private.pem")


def test_dashboard_restores_enabled_push_state_after_reload():
    dashboard = (Path(__file__).parents[1] / "templates" / "dashboard.html").read_text()

    assert "async function syncPushButton()" in dashboard
    assert "await navigator.serviceWorker.getRegistration()" in dashboard
    assert 'b.textContent = sub ? "Notifications enabled"' in dashboard
    assert "syncPushButton();" in dashboard
