from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import threading
import time


class BinanceRestDeferred(RuntimeError):
    """A local rate guard deferred a Binance REST request before transmission."""

    def __init__(self, retry_after_seconds: float, reason: str = "local_rate_limit"):
        self.retry_after_seconds = max(0.0, float(retry_after_seconds))
        self.reason = str(reason)
        super().__init__(f"{self.reason}; retry_after={self.retry_after_seconds:.3f}s")


class BinanceRateLimited(RuntimeError):
    """Binance returned an actual HTTP 429 or 418 response."""

    def __init__(self, status_code: int, retry_after_seconds: float):
        self.status_code = int(status_code)
        self.retry_after_seconds = max(0.0, float(retry_after_seconds))
        super().__init__(
            f"binance_http_{self.status_code}; retry_after={self.retry_after_seconds:.3f}s"
        )


@dataclass(frozen=True)
class BinanceRestConfig:
    # Binance publishes 2,400 request-weight/min for the relevant futures API.
    # This service deliberately caps itself at half that value.
    capacity_weight: float = 1200.0
    window_seconds: float = 60.0
    default_429_cooldown_seconds: float = 60.0
    default_418_cooldown_seconds: float = 600.0


class BinanceRestLimiter:
    """Thread-safe token bucket plus Binance response-header accounting."""

    def __init__(self, config=None, monotonic_fn=None, wall_time_fn=None):
        self.config = config or BinanceRestConfig()
        self._monotonic = monotonic_fn or time.monotonic
        self._wall_time = wall_time_fn or time.time
        self._lock = threading.RLock()
        self._tokens = float(self.config.capacity_weight)
        self._updated_at = float(self._monotonic())
        self._cooldown_until = 0.0
        self._cooldown_until_epoch = None
        self._server_used_weight_1m = None
        self._server_weight_seen_at = None
        self._attempted = 0
        self._succeeded = 0
        self._deferred = 0
        self._rate_limited = 0
        self._last_attempt_at = None
        self._last_success_at = None
        self._last_status_code = None
        self._last_error = None

    @staticmethod
    def _iso(epoch_seconds):
        if epoch_seconds is None:
            return None
        return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc).isoformat()

    def _refill(self, now):
        elapsed = max(0.0, float(now) - self._updated_at)
        refill_rate = self.config.capacity_weight / self.config.window_seconds
        self._tokens = min(
            self.config.capacity_weight,
            self._tokens + elapsed * refill_rate,
        )
        self._updated_at = float(now)
        if (
            self._server_weight_seen_at is not None
            and float(now) - self._server_weight_seen_at >= self.config.window_seconds
        ):
            self._server_used_weight_1m = None
            self._server_weight_seen_at = None

    def acquire(self, weight):
        weight = max(0.0, float(weight))
        now = float(self._monotonic())
        wall_now = float(self._wall_time())
        with self._lock:
            self._refill(now)
            if now < self._cooldown_until:
                self._deferred += 1
                raise BinanceRestDeferred(
                    self._cooldown_until - now,
                    "binance_cooldown",
                )

            server_would_exceed = (
                self._server_used_weight_1m is not None
                and self._server_used_weight_1m + weight
                > self.config.capacity_weight
            )
            if self._tokens < weight or server_would_exceed:
                refill_rate = self.config.capacity_weight / self.config.window_seconds
                token_wait = max(0.0, weight - self._tokens) / refill_rate
                server_wait = (
                    max(0.0, self.config.window_seconds - (now - self._server_weight_seen_at))
                    if server_would_exceed and self._server_weight_seen_at is not None
                    else 0.0
                )
                self._deferred += 1
                raise BinanceRestDeferred(
                    max(token_wait, server_wait, 0.05),
                    "binance_weight_budget",
                )

            self._tokens -= weight
            self._attempted += 1
            self._last_attempt_at = wall_now

    @staticmethod
    def _header(headers, name):
        target = name.lower()
        for key, value in (headers or {}).items():
            if str(key).lower() == target:
                return value
        return None

    def _retry_after_seconds(self, headers, status_code):
        value = self._header(headers, "Retry-After")
        if value is not None:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                try:
                    when = parsedate_to_datetime(str(value))
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=timezone.utc)
                    return max(0.0, when.timestamp() - float(self._wall_time()))
                except (TypeError, ValueError, OverflowError):
                    pass
        return (
            self.config.default_418_cooldown_seconds
            if int(status_code) == 418
            else self.config.default_429_cooldown_seconds
        )

    def observe_response(self, status_code, headers):
        status_code = int(status_code)
        now = float(self._monotonic())
        wall_now = float(self._wall_time())
        with self._lock:
            used_values = []
            for key, value in (headers or {}).items():
                if str(key).lower().startswith("x-mbx-used-weight"):
                    try:
                        used_values.append(float(value))
                    except (TypeError, ValueError):
                        continue
            if used_values:
                self._server_used_weight_1m = max(used_values)
                self._server_weight_seen_at = now

            self._last_status_code = status_code
            if status_code in (418, 429):
                retry_after = self._retry_after_seconds(headers, status_code)
                self._cooldown_until = now + retry_after
                self._cooldown_until_epoch = wall_now + retry_after
                self._rate_limited += 1
                self._last_error = f"http_{status_code}"
                raise BinanceRateLimited(status_code, retry_after)

            if 200 <= status_code < 400:
                self._succeeded += 1
                self._last_success_at = wall_now
                self._last_error = None

    def observe_error(self, exc):
        with self._lock:
            self._last_error = f"{type(exc).__name__}: {str(exc)[:160]}"

    def snapshot(self):
        now = float(self._monotonic())
        with self._lock:
            self._refill(now)
            cooldown_remaining = max(0.0, self._cooldown_until - now)
            return {
                "capacity_weight_per_minute": self.config.capacity_weight,
                "available_weight": round(self._tokens, 1),
                "server_used_weight_1m": self._server_used_weight_1m,
                "cooldown": cooldown_remaining > 0,
                "cooldown_remaining_seconds": round(cooldown_remaining, 3),
                "cooldown_until": self._iso(self._cooldown_until_epoch),
                "requests_attempted": self._attempted,
                "requests_succeeded": self._succeeded,
                "requests_deferred": self._deferred,
                "rate_limit_responses": self._rate_limited,
                "last_attempt_at": self._iso(self._last_attempt_at),
                "last_success_at": self._iso(self._last_success_at),
                "last_status_code": self._last_status_code,
                "last_error": self._last_error,
            }
