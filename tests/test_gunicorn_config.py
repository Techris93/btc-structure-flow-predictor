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
