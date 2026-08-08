from __future__ import annotations

import pandas as pd


class BinanceBackfillController:
    """Allow one startup backfill and one recovery backfill per WS outage."""

    def __init__(
        self,
        startup_enabled: bool = True,
        gap_recovery_enabled: bool = True,
        gap_seconds: float = 45.0,
    ):
        self.startup_enabled = bool(startup_enabled)
        self.gap_recovery_enabled = bool(gap_recovery_enabled)
        self.gap_seconds = max(1.0, float(gap_seconds))
        self.startup_pending = self.startup_enabled
        self.was_fresh = False
        self.outage_started_at = None
        self.outage_backfilled = False
        self.retry_at = pd.Timestamp(0, tz="UTC")
        self.last_reason = None
        self.last_attempt_at = None

    @staticmethod
    def _timestamp(value):
        result = pd.Timestamp(value)
        return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")

    def decide(self, now, websocket_fresh: bool):
        now = self._timestamp(now)
        websocket_fresh = bool(websocket_fresh)
        if websocket_fresh:
            self.was_fresh = True
            self.outage_started_at = None
            self.outage_backfilled = False
            if self.startup_pending and now >= self.retry_at:
                return "startup"
            return None

        if self.startup_pending:
            return "startup" if now >= self.retry_at else None

        if not self.gap_recovery_enabled:
            return None
        if self.outage_started_at is None:
            self.outage_started_at = now
            self.outage_backfilled = False
        age = (now - self.outage_started_at).total_seconds()
        if (
            self.was_fresh
            and not self.outage_backfilled
            and age >= self.gap_seconds
            and now >= self.retry_at
        ):
            return "gap_recovery"
        return None

    def mark_success(self, reason, now, websocket_fresh: bool):
        now = self._timestamp(now)
        self.last_reason = str(reason)
        self.last_attempt_at = now
        self.retry_at = pd.Timestamp(0, tz="UTC")
        if reason == "startup":
            self.startup_pending = False
            if not websocket_fresh:
                # Startup backfill covers the current initial outage exactly
                # once. A later fresh->stale transition creates a new gap.
                self.outage_started_at = now
                self.outage_backfilled = True
        elif reason == "gap_recovery":
            self.outage_backfilled = True

    def mark_failure(self, reason, now, retry_after_seconds):
        now = self._timestamp(now)
        self.last_reason = str(reason)
        self.last_attempt_at = now
        self.retry_at = now + pd.Timedelta(seconds=max(0.0, float(retry_after_seconds)))

    def snapshot(self, now=None):
        now = self._timestamp(now or pd.Timestamp.now(tz="UTC"))
        outage_age = (
            max(0.0, (now - self.outage_started_at).total_seconds())
            if self.outage_started_at is not None
            else None
        )
        return {
            "startup_pending": self.startup_pending,
            "was_websocket_fresh": self.was_fresh,
            "outage_started_at": (
                self.outage_started_at.isoformat()
                if self.outage_started_at is not None
                else None
            ),
            "outage_age_seconds": (
                round(outage_age, 1) if outage_age is not None else None
            ),
            "outage_backfilled": self.outage_backfilled,
            "retry_at": self.retry_at.isoformat(),
            "last_reason": self.last_reason,
            "last_attempt_at": (
                self.last_attempt_at.isoformat()
                if self.last_attempt_at is not None
                else None
            ),
        }
