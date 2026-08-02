"""Gunicorn lifecycle hooks for the stateful live predictor worker."""

import os


def post_worker_init(worker):
    """Start collectors only after Gunicorn has forked and initialized Flask.

    Starting threads at module import time is unsafe under Gunicorn: the app
    can be imported in the parent and only the calling thread survives fork.
    This hook runs inside the actual worker process.
    """
    if os.getenv("START_LIVE_LOOP_ON_BOOT", "1").lower() not in (
        "1", "true", "yes", "on"
    ):
        return
    from app import start_live_boot_supervisor

    start_live_boot_supervisor()
