from __future__ import annotations

import math
from pathlib import Path
import threading

import numpy as np
import pandas as pd

from .persistence import JsonStore


SESSION_WINDOWS = (
    (0, 8, "asia"),
    (8, 16, "london"),
    (16, 24, "new_york"),
)


def session_identity(at) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    at = pd.Timestamp(at)
    at = at.tz_localize("UTC") if at.tzinfo is None else at.tz_convert("UTC")
    day = at.floor("D")
    for start_hour, end_hour, name in SESSION_WINDOWS:
        if start_hour <= at.hour < end_hour:
            return name, day + pd.Timedelta(hours=start_hour), day + pd.Timedelta(hours=end_hour)
    raise AssertionError("UTC hour must belong to one session")


class FlowStateStore:
    """Durable closed-minute flow aggregates and per-sweep diagnostics.

    Existing closed bars are immutable: late trades can only affect a later
    decision bar. The store keeps compact price buckets rather than raw events.
    """

    VERSION = 1

    def __init__(self, path: str | Path, price_bucket: float = 25.0, retention_minutes: int = 540):
        self.store = JsonStore(path)
        self.lock = threading.RLock()
        self.price_bucket = max(float(price_bucket), 0.01)
        self.retention_minutes = max(120, int(retention_minutes))
        self.state = self._load()

    def _empty(self):
        return {
            "version": self.VERSION,
            "price_bucket": self.price_bucket,
            "bars": [],
            "sweeps": {},
            "last_processed_closed_bar": {},
            "updated_at": None,
        }

    def _load(self):
        state = self.store.read(self._empty())
        if state.get("version") != self.VERSION:
            return self._empty()
        state={**self._empty(),**state}
        if float(state.get("price_bucket",self.price_bucket))!=self.price_bucket:
            # Preserve venue deltas/CVD across a calibrated bucket change, but
            # never reinterpret old price buckets under the new geometry.
            for bar in state.get("bars") or []:bar["buckets"]=[]
            state["price_bucket"]=self.price_bucket
        return state

    def _save(self):
        self.store.write(self.state)

    def update(self, trades: pd.DataFrame, decision_time) -> bool:
        """Insert unseen, fully closed 1m aggregates and ignore later revisions."""
        decision_time = pd.Timestamp(decision_time)
        decision_time = decision_time.tz_localize("UTC") if decision_time.tzinfo is None else decision_time.tz_convert("UTC")
        if trades is None or trades.empty or "exchange" not in trades:
            return False
        frame = trades.copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[frame.time < decision_time]
        if frame.empty:
            return False
        frame["bar_close"] = frame.time.dt.ceil("1min")
        frame = frame.loc[frame.bar_close <= decision_time]
        frame["notional"] = pd.to_numeric(frame.price) * pd.to_numeric(frame.qty)
        frame["signed"] = np.where(frame.side.astype(str).str.lower().eq("buy"), frame.notional, -frame.notional)
        frame["price_level"] = (pd.to_numeric(frame.price) / self.price_bucket).round() * self.price_bucket

        with self.lock:
            known = {(str(row["exchange"]), str(row["bar_close"])) for row in self.state["bars"]}
            changed = False
            for (exchange, bar_close), group in frame.groupby(["exchange", "bar_close"], sort=True):
                bar_close = pd.Timestamp(bar_close).isoformat()
                key = (str(exchange), bar_close)
                if key in known:
                    continue
                buckets = []
                for level, bucket in group.groupby("price_level", sort=True):
                    buy = float(bucket.loc[bucket.signed > 0, "notional"].sum())
                    sell = float(bucket.loc[bucket.signed < 0, "notional"].sum())
                    buckets.append({"price_level": float(level), "buy": buy, "sell": sell})
                self.state["bars"].append({
                    "exchange": str(exchange),
                    "bar_close": bar_close,
                    "buy": float(group.loc[group.signed > 0, "notional"].sum()),
                    "sell": float(-group.loc[group.signed < 0, "signed"].sum()),
                    "volume": float(group.notional.sum()),
                    "trades": int(len(group)),
                    "buckets": buckets,
                })
                self.state["last_processed_closed_bar"][str(exchange)] = bar_close
                known.add(key)
                changed = True
            cutoff = decision_time - pd.Timedelta(minutes=self.retention_minutes)
            retained = [row for row in self.state["bars"] if pd.Timestamp(row["bar_close"]) > cutoff]
            if len(retained) != len(self.state["bars"]):
                self.state["bars"] = retained
                changed = True
            if changed:
                self.state["bars"].sort(key=lambda row: (row["bar_close"], row["exchange"]))
                self.state["updated_at"] = decision_time.isoformat()
                self._save()
            return changed

    def footprint_bars(self, start=None, end=None) -> pd.DataFrame:
        with self.lock:
            rows = []
            for bar in self.state["bars"]:
                close = pd.Timestamp(bar["bar_close"])
                if start is not None and close <= pd.Timestamp(start):
                    continue
                if end is not None and close > pd.Timestamp(end):
                    continue
                for bucket in bar.get("buckets") or []:
                    rows.append({
                        "bar_close": close,
                        "exchange": bar["exchange"],
                        "price_level": float(bucket["price_level"]),
                        "buy": float(bucket["buy"]),
                        "sell": float(bucket["sell"]),
                    })
        return pd.DataFrame(rows, columns=["bar_close", "exchange", "price_level", "buy", "sell"])

    def session_cvd(self, decision_time) -> dict:
        decision_time = pd.Timestamp(decision_time)
        decision_time = decision_time.tz_localize("UTC") if decision_time.tzinfo is None else decision_time.tz_convert("UTC")
        name, start, end = session_identity(decision_time)
        with self.lock:
            bars = pd.DataFrame(self.state["bars"])
        result = {
            "session": name,
            "session_start": start.isoformat(),
            "session_end": end.isoformat(),
            "decision_time": decision_time.isoformat(),
            "venues": {},
            "combined": None,
            "combined_slope": None,
            "complete": False,
        }
        expected = max(1, int(math.ceil((decision_time - start).total_seconds() / 60.0)))
        for exchange in ("binance", "bybit"):
            if bars.empty:
                venue = pd.DataFrame()
            else:
                closes = pd.to_datetime(bars.bar_close, utc=True)
                venue = bars.loc[(bars.exchange == exchange) & (closes > start) & (closes <= decision_time)].copy()
            if venue.empty:
                result["venues"][exchange] = {"cvd": None, "slope": None, "bars": 0, "coverage": 0.0, "complete": False}
                continue
            venue["bar_close"] = pd.to_datetime(venue.bar_close, utc=True)
            venue = venue.sort_values("bar_close")
            delta = pd.to_numeric(venue.buy) - pd.to_numeric(venue.sell)
            coverage = min(1.0, len(venue) / expected)
            starts_on_time = venue.bar_close.iloc[0] <= start + pd.Timedelta(minutes=2)
            complete = bool(starts_on_time and coverage >= 0.98)
            result["venues"][exchange] = {
                "cvd": float(delta.sum()),
                "slope": float(delta.tail(5).sum()),
                "bars": int(len(venue)),
                "coverage": round(float(coverage), 3),
                "complete": complete,
            }
        binance = result["venues"]["binance"]
        bybit = result["venues"]["bybit"]
        if binance["complete"] and bybit["complete"]:
            result["combined"] = float(binance["cvd"] + bybit["cvd"])
            result["combined_slope"] = float(binance["slope"] + bybit["slope"])
            result["complete"] = True
        return result

    def record_sweeps(self, observations, decision_time):
        decision_time = pd.Timestamp(decision_time)
        decision_time = decision_time.tz_localize("UTC") if decision_time.tzinfo is None else decision_time.tz_convert("UTC")
        with self.lock:
            changed = False
            for observation in observations or []:
                zone = observation.get("zone")
                sweep_time = observation.get("sweep_time")
                if not zone or not sweep_time:
                    continue
                key = f"{zone}|{pd.Timestamp(sweep_time).isoformat()}"
                previous = self.state["sweeps"].get(key)
                if previous and previous.get("flow_state") == "frozen":
                    continue
                value = {**observation, "last_seen_at": decision_time.isoformat()}
                self.state["sweeps"][key] = value
                changed = True
            cutoff = decision_time - pd.Timedelta(minutes=60)
            retained = {
                key: value for key, value in self.state["sweeps"].items()
                if pd.Timestamp(value.get("sweep_time")) >= cutoff or value.get("flow_state") == "frozen"
            }
            # Frozen diagnostics need not live forever; retain one session.
            retained = {
                key: value for key, value in retained.items()
                if pd.Timestamp(value.get("last_seen_at", decision_time)) >= decision_time - pd.Timedelta(hours=8)
            }
            if retained != self.state["sweeps"]:
                self.state["sweeps"] = retained
                changed = True
            if changed:
                self.state["updated_at"] = decision_time.isoformat()
                self._save()

    def sweep_states(self):
        with self.lock:
            return {key: dict(value) for key, value in self.state["sweeps"].items()}
