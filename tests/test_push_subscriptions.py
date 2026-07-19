from app import _upsert_subscription


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
