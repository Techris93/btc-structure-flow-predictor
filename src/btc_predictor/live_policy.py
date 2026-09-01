"""Live paper governance: research economics, snapshots, risk, soft filters.

These controls are process/risk instrumentation — not OOS-validated alpha rules.
Flow thresholds must not be retuned from small live samples.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# Match research backtest defaults (backtest.py).
RESEARCH_FEE_BPS = 5.0
RESEARCH_SLIPPAGE_BPS = 2.0
RESEARCH_INITIAL_EQUITY = 100_000.0
RISK_FRACTION = 0.0025

# Risk governance (unproven sizes; process only).
MAX_NOTIONAL_MULTIPLE = float(os.getenv("PAPER_MAX_NOTIONAL_MULTIPLE", "1.0"))
DAILY_LOSS_R = float(os.getenv("PAPER_DAILY_LOSS_R", "2.0"))
WEEKLY_LOSS_R = float(os.getenv("PAPER_WEEKLY_LOSS_R", "4.0"))
MAX_HOLD_HOURS = float(os.getenv("PAPER_MAX_HOLD_HOURS", "12"))
SAME_SIDE_COOLDOWN_HOURS = float(os.getenv("PAPER_SAME_SIDE_COOLDOWN_HOURS", "8"))
SAME_SIDE_TP_COOLDOWN_HOURS = float(os.getenv("PAPER_SAME_SIDE_TP_COOLDOWN_HOURS", "2.0"))
FILL_MIN_RR = float(os.getenv("PAPER_FILL_MIN_RR", "1.5"))

# Fixed percent exits: 1% stop / 2% target → 2R from the fill.
FIXED_STOP_PCT = float(os.getenv("FIXED_STOP_PCT", "0.01"))
FIXED_TARGET_PCT = float(os.getenv("FIXED_TARGET_PCT", "0.02"))
USE_FIXED_PCT_EXITS = os.getenv("USE_FIXED_PCT_EXITS", "1").lower() in ("1", "true", "yes", "on")
HUNDRED_PUSH_BUFFER = float(os.getenv("PAPER_HUNDRED_PUSH_BUFFER", "1.0"))  # dollars beyond the 100

# Soft filters from 3-trade postmortem — labeled unproven.
SOFT_FILTERS_ENABLED = os.getenv("PAPER_SOFT_FILTERS", "1").lower() in ("1", "true", "yes", "on")
SOFT_MAX_PLANNED_RR = float(os.getenv("PAPER_SOFT_MAX_RR", "2.5"))
SOFT_HERO_RR = float(os.getenv("PAPER_SOFT_HERO_RR", "3.5"))
SOFT_PREFERRED_RR_MIN = 1.5
SOFT_PREFERRED_RR_MAX = 2.2
SOFT_ROUND_MAGNET_PCT = float(os.getenv("PAPER_ROUND_MAGNET_PCT", "0.001"))  # 0.1%
SOFT_WIDE_STOP_PCT = float(os.getenv("PAPER_WIDE_STOP_PCT", "0.0035"))  # 0.35%
SOFT_WIDE_BREAKOUT_KINDS = frozenset({"untested_breakout"})

# Retune discipline.
MIN_CLOSED_BEFORE_RETUNE = int(os.getenv("PAPER_MIN_CLOSED_BEFORE_RETUNE", "40"))
RETUNE_CALENDAR_DAYS = int(os.getenv("PAPER_RETUNE_CALENDAR_DAYS", "90"))
POLICY_EFFECTIVE_AT = os.getenv("PAPER_POLICY_EFFECTIVE_AT", "2026-08-12T00:00:00+00:00")
REVIEW_METRICS = (
    "expectancy_r_after_costs",
    "profit_factor_net",
    "max_drawdown_net",
    "closed_trade_count",
    "reject_funnel",
)

# Shadow book: Book A = production paper; Book B adds one extra skip rule.
# Fixed 1%/2% makes RR≈2 always, so Book B tracks untested_breakout skips.
SHADOW_RULE = os.getenv("PAPER_SHADOW_RULE", "skip_untested_breakout")
SHADOW_RR_CAP = float(os.getenv("PAPER_SHADOW_RR_CAP", "2.5"))
SHADOW_MAGNET_DOLLARS = float(os.getenv("PAPER_SHADOW_MAGNET_DOLLARS", "50"))

PROBABILITY_SOURCE = "heuristic_uncalibrated"
PROBABILITY_USE = "display_and_log_only"


def research_economics() -> dict[str, Any]:
    return {
        "fee_bps": RESEARCH_FEE_BPS,
        "slippage_bps": RESEARCH_SLIPPAGE_BPS,
        "initial_equity": RESEARCH_INITIAL_EQUITY,
        "risk_fraction": RISK_FRACTION,
        "aligned_with": "backtest.run_event_backtest defaults",
        "note": "Paper P&L must report gross vs approx-net; do not treat gross as alpha.",
    }


def policy_manifest() -> dict[str, Any]:
    effective = pd.Timestamp(POLICY_EFFECTIVE_AT)
    if effective.tzinfo is None:
        effective = effective.tz_localize("UTC")
    review_by = (effective + pd.Timedelta(days=RETUNE_CALENDAR_DAYS)).isoformat()
    return {
        "version": 1,
        "label": "do_now_no_historical_replay",
        "effective_at": effective.isoformat(),
        "review_by": review_by,
        "min_closed_trades_before_retune": MIN_CLOSED_BEFORE_RETUNE,
        "review_metrics_only": list(REVIEW_METRICS),
        "economics": research_economics(),
        "risk": {
            "risk_fraction": RISK_FRACTION,
            "max_notional_multiple": MAX_NOTIONAL_MULTIPLE,
            "daily_loss_r": DAILY_LOSS_R,
            "weekly_loss_r": WEEKLY_LOSS_R,
            "one_open_risk_unit": True,
            "max_hold_hours": MAX_HOLD_HOURS,
            "same_side_cooldown_hours": SAME_SIDE_COOLDOWN_HOURS,
            "same_side_tp_cooldown_hours": SAME_SIDE_TP_COOLDOWN_HOURS,
            "fill_min_rr": FILL_MIN_RR,
        },
        "exits": {
            "mode": "fixed_pct" if USE_FIXED_PCT_EXITS else "structural_atr",
            "stop_pct": FIXED_STOP_PCT,
            "target_pct": FIXED_TARGET_PCT,
            "planned_rr": (FIXED_TARGET_PCT / FIXED_STOP_PCT) if FIXED_STOP_PCT else None,
            "push_stop_through_100": False,
        },
        "soft_filters": {
            "enabled": SOFT_FILTERS_ENABLED,
            "validated": False,
            "label": "unproven_postmortem_heuristics",
            "max_planned_rr": SOFT_MAX_PLANNED_RR,
            "hero_rr": SOFT_HERO_RR,
            "preferred_rr": [SOFT_PREFERRED_RR_MIN, SOFT_PREFERRED_RR_MAX],
            "round_magnet_pct": SOFT_ROUND_MAGNET_PCT,
            "wide_stop_pct": SOFT_WIDE_STOP_PCT,
            "wide_breakout_kinds": sorted(SOFT_WIDE_BREAKOUT_KINDS),
            "require_new_zone_after_magnet_stop": True,
            "do_not_retune_flow_thresholds": True,
        },
        "probability": {
            "source": PROBABILITY_SOURCE,
            "use": PROBABILITY_USE,
            "sizing": "fixed_risk_fraction_only",
            "lifecycle_ranking": "soft_diagnostic_only",
        },
        "data_quality": {
            "fail_closed": True,
            "required_market_type": "linear",
            "required_venues": ["binance", "bybit"],
            "reject_spot_or_mixed": True,
            "research_only_if_degraded": True,
        },
        "shadow_book": {
            "book_a": "production_paper",
            "book_b_rule": SHADOW_RULE,
            "forward_only": True,
            "do_not_promote_early": True,
        },
    }


def nearest_round_levels(price: float) -> dict[str, Any]:
    """Distance of a price to nearest 100 / 500 / 1000 magnets."""
    if price is None or not math.isfinite(float(price)) or float(price) <= 0:
        return {
            "nearest_100": None,
            "nearest_500": None,
            "nearest_1000": None,
            "dist_100": None,
            "dist_500": None,
            "dist_1000": None,
            "dist_100_pct": None,
            "dist_500_pct": None,
            "dist_1000_pct": None,
            "min_dist_pct": None,
        }
    p = float(price)
    out: dict[str, Any] = {}
    for step, name in ((100.0, "100"), (500.0, "500"), (1000.0, "1000")):
        nearest = round(p / step) * step
        dist = abs(p - nearest)
        out[f"nearest_{name}"] = nearest
        out[f"dist_{name}"] = round(dist, 4)
        out[f"dist_{name}_pct"] = round(dist / p, 8)
    out["min_dist_pct"] = min(out["dist_100_pct"], out["dist_500_pct"], out["dist_1000_pct"])
    return out


def stop_geometry(entry: float | None, stop: float | None, side: str | None = None) -> dict[str, Any]:
    if entry is None or stop is None:
        return {
            "stop_distance": None,
            "stop_distance_pct": None,
            "stop_magnets": nearest_round_levels(None),
            "stop_on_round_magnet": None,
            "stop_on_major_magnet": None,
        }
    entry_f = float(entry)
    stop_f = float(stop)
    dist = abs(entry_f - stop_f)
    pct = dist / entry_f if entry_f else None
    magnets = nearest_round_levels(stop_f)
    # $100 prints are ubiquitous; hard-skip only major 500/1000 magnets (T1 ~65k).
    major_pct = None
    for key in ("dist_500_pct", "dist_1000_pct"):
        val = magnets.get(key)
        if val is not None:
            major_pct = val if major_pct is None else min(major_pct, val)
    on_major = bool(major_pct is not None and major_pct <= SOFT_ROUND_MAGNET_PCT)
    on_any = bool(
        magnets.get("min_dist_pct") is not None
        and magnets["min_dist_pct"] <= SOFT_ROUND_MAGNET_PCT
    )
    return {
        "stop_distance": round(dist, 6),
        "stop_distance_pct": round(pct, 8) if pct is not None else None,
        "stop_magnets": magnets,
        "stop_on_round_magnet": on_any,
        "stop_on_major_magnet": on_major,
        "side": side,
    }


def push_stop_beyond_hundred(entry: float, stop: float, side: str, buffer: float | None = None) -> tuple[float, bool]:
    """If the stop sits just in front of a $100 print, push it through.

    Long: stop is below entry. If a 100 lies between stop and entry, the stop
    is in front of that magnet — move it to magnet-buffer. Short is symmetric.
    """
    buf = HUNDRED_PUSH_BUFFER if buffer is None else float(buffer)
    entry_f, stop_f = float(entry), float(stop)
    side = "long" if side in ("long", "bullish") else "short"
    lo, hi = (stop_f, entry_f) if side == "long" else (entry_f, stop_f)
    first_hundred = math.ceil(lo / 100.0) * 100.0
    pushed = False
    if first_hundred < hi - 1e-9 and first_hundred > lo + 1e-9:
        if side == "long":
            stop_f = first_hundred - buf
            pushed = True
        else:
            stop_f = first_hundred + buf
            pushed = True
    return stop_f, pushed


def fixed_pct_exits(
    entry: float,
    side: str,
    *,
    stop_pct: float | None = None,
    target_pct: float | None = None,
    push_through_100: bool = False,
) -> dict[str, Any]:
    """1% stop / 2% target from *this* price (signal or fill). Always ~2R."""
    stop_pct = FIXED_STOP_PCT if stop_pct is None else float(stop_pct)
    target_pct = FIXED_TARGET_PCT if target_pct is None else float(target_pct)
    entry_f = float(entry)
    is_long = side in ("long", "bullish")
    if is_long:
        stop = entry_f * (1.0 - stop_pct)
        target = entry_f * (1.0 + target_pct)
    else:
        stop = entry_f * (1.0 + stop_pct)
        target = entry_f * (1.0 - target_pct)
    pushed = False
    if push_through_100:
        stop, pushed = push_stop_beyond_hundred(entry_f, stop, "long" if is_long else "short")
    risk = abs(entry_f - stop)
    reward = abs(entry_f - target)
    rr = reward / risk if risk > 0 else 0.0
    return {
        "entry": entry_f,
        "stop": float(stop),
        "target": float(target),
        "risk": float(risk),
        "reward": float(reward),
        "reward_risk": float(rr),
        "stop_pct": stop_pct,
        "target_pct": target_pct,
        "stop_pushed_through_100": pushed,
    }


def fill_min_rr_ok(entry: float, stop: float, target: float, side: str, min_rr: float | None = None) -> bool:
    floor = FILL_MIN_RR if min_rr is None else float(min_rr)
    risk = (entry - stop) if side in ("long", "bullish") else (stop - entry)
    reward = (target - entry) if side in ("long", "bullish") else (entry - target)
    if risk <= 0 or reward <= 0:
        return False
    return (reward / risk) >= floor


def enrich_decision_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Attach geometry and probability metadata to a lifecycle/predictor snapshot."""
    snap = dict(snapshot or {})
    entry = snap.get("entry")
    stop = snap.get("stop")
    bias = snap.get("bias")
    side = "long" if bias == "bullish" else "short" if bias == "bearish" else None
    geometry = stop_geometry(entry, stop, side)
    snap.update({
        "stop_distance": geometry["stop_distance"],
        "stop_distance_pct": geometry["stop_distance_pct"],
        "stop_on_round_magnet": geometry["stop_on_round_magnet"],
        "stop_on_major_magnet": geometry["stop_on_major_magnet"],
        "stop_magnets": geometry["stop_magnets"],
        "probability_source": PROBABILITY_SOURCE,
        "probability_use": PROBABILITY_USE,
        "probability_tp_before_sl_is_heuristic": True,
        "policy_version": 1,
        "decision_bar": snap.get("timestamp"),
    })
    rr = snap.get("reward_risk")
    if rr is not None and math.isfinite(float(rr)):
        rr_f = float(rr)
        snap["planned_rr"] = rr_f
        snap["rr_in_preferred_band"] = SOFT_PREFERRED_RR_MIN <= rr_f <= SOFT_PREFERRED_RR_MAX
        snap["rr_hero"] = rr_f >= SOFT_HERO_RR
    zone = str(snap.get("zone") or "")
    kind = snap.get("zone_kind") or (zone.split(":", 1)[0] if zone else None)
    if kind and not snap.get("zone_kind"):
        snap["zone_kind"] = kind
    # With fixed 1% stops, width is uniform. Flag the zone kind only.
    snap["wide_untested_breakout"] = bool(kind in SOFT_WIDE_BREAKOUT_KINDS)
    # Soft expectancy diagnostic only — never size or hard-gate from this.
    p = snap.get("probability_tp_before_sl")
    if p is not None and rr is not None:
        try:
            p_f = float(p)
            rr_f = float(rr)
            snap["soft_expectancy_r"] = round(p_f * rr_f - (1.0 - p_f), 6)
            snap["soft_expectancy_is_diagnostic"] = True
        except (TypeError, ValueError):
            snap["soft_expectancy_r"] = None
            snap["soft_expectancy_is_diagnostic"] = True
    else:
        snap["soft_expectancy_r"] = None
        snap["soft_expectancy_is_diagnostic"] = True
    return snap


def evaluate_soft_filters(
    snapshot: dict[str, Any],
    *,
    last_closed: dict | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Return skip/warn flags. Unproven heuristics from paper postmortem."""
    use = SOFT_FILTERS_ENABLED if enabled is None else bool(enabled)
    snap = enrich_decision_snapshot(snapshot)
    hard_skips: list[str] = []
    warnings: list[str] = []
    if not use:
        return {
            "enabled": False,
            "validated": False,
            "allow": True,
            "hard_skips": [],
            "warnings": ["soft_filters_disabled"],
            "snapshot": snap,
        }

    rr = snap.get("planned_rr") or snap.get("reward_risk")
    if rr is not None and float(rr) > SOFT_MAX_PLANNED_RR:
        hard_skips.append(f"planned_rr_above_{SOFT_MAX_PLANNED_RR:g}")
    if rr is not None and float(rr) >= SOFT_HERO_RR:
        warnings.append("hero_rr_target")

    if snap.get("stop_on_major_magnet"):
        hard_skips.append("stop_on_major_magnet")
        warnings.append("stop_near_500_or_1000")
    elif snap.get("stop_on_round_magnet"):
        warnings.append("stop_near_100_print")

    if snap.get("wide_untested_breakout"):
        # Book A: warning only — 1% SL is now uniform. Book B shadows the skip.
        warnings.append("untested_breakout_zone")

    # After a stop, require a new zone and block same-side re-entry for a cooldown.
    if last_closed and str(last_closed.get("exit_reason") or "").lower() == "stop":
        prev_geom = stop_geometry(last_closed.get("entry"), last_closed.get("stop"), last_closed.get("side"))
        if prev_geom.get("stop_on_major_magnet"):
            prev_zone = last_closed.get("zone")
            if prev_zone and snap.get("zone") == prev_zone:
                hard_skips.append("same_zone_after_magnet_stop")
        last_side = last_closed.get("side")
        next_side = "long" if snap.get("bias") == "bullish" else "short" if snap.get("bias") == "bearish" else None
        if last_side and next_side == last_side and last_closed.get("exit_time") and snap.get("timestamp"):
            try:
                gap_h = (
                    pd.Timestamp(snap["timestamp"]) - pd.Timestamp(last_closed["exit_time"])
                ).total_seconds() / 3600.0
                if gap_h < SAME_SIDE_COOLDOWN_HOURS:
                    hard_skips.append("same_side_cooldown_after_stop")
            except (TypeError, ValueError):
                pass

    # After a take profit (target), require a new zone and block same-side re-entry for a cooldown.
    if last_closed and str(last_closed.get("exit_reason") or "").lower() in ("target", "tp"):
        prev_zone = last_closed.get("zone")
        if prev_zone and snap.get("zone") == prev_zone:
            hard_skips.append("same_zone_after_target")
        last_side = last_closed.get("side")
        next_side = "long" if snap.get("bias") == "bullish" else "short" if snap.get("bias") == "bearish" else None
        if last_side and next_side == last_side and last_closed.get("exit_time") and snap.get("timestamp"):
            try:
                gap_h = (
                    pd.Timestamp(snap["timestamp"]) - pd.Timestamp(last_closed["exit_time"])
                ).total_seconds() / 3600.0
                if gap_h < SAME_SIDE_TP_COOLDOWN_HOURS:
                    hard_skips.append("same_side_cooldown_after_target")
            except (TypeError, ValueError):
                pass

    return {
        "enabled": True,
        "validated": False,
        "label": "unproven_postmortem_heuristics",
        "allow": not hard_skips,
        "hard_skips": hard_skips,
        "warnings": warnings,
        "snapshot": snap,
    }


def apply_risk_caps(
    entry: float,
    stop: float,
    size: float,
    equity: float,
    *,
    closed: list[dict] | None = None,
    has_open_or_pending: bool = False,
    now: Any = None,
    max_notional_multiple: float | None = None,
    daily_loss_r: float | None = None,
    weekly_loss_r: float | None = None,
    risk_fraction: float | None = None,
) -> dict[str, Any]:
    """Cap notional/leverage and enforce daily/weekly R loss + one open unit."""
    max_mult = MAX_NOTIONAL_MULTIPLE if max_notional_multiple is None else float(max_notional_multiple)
    day_cap = DAILY_LOSS_R if daily_loss_r is None else float(daily_loss_r)
    week_cap = WEEKLY_LOSS_R if weekly_loss_r is None else float(weekly_loss_r)
    risk_frac = RISK_FRACTION if risk_fraction is None else float(risk_fraction)

    reasons: list[str] = []
    if has_open_or_pending:
        return {
            "allow": False,
            "reasons": ["one_open_risk_unit"],
            "size": 0.0,
            "notional": 0.0,
            "risk_cash": 0.0,
        }

    risk_per_unit = abs(float(entry) - float(stop))
    if risk_per_unit <= 0 or not math.isfinite(risk_per_unit):
        return {"allow": False, "reasons": ["invalid_risk"], "size": 0.0, "notional": 0.0, "risk_cash": 0.0}

    # Fixed fractional risk (already production default).
    risk_cash = float(equity) * risk_frac
    size_from_risk = risk_cash / risk_per_unit
    size_f = min(float(size), size_from_risk) if size and float(size) > 0 else size_from_risk

    notional = abs(float(entry) * size_f)
    max_notional = float(equity) * max_mult
    if notional > max_notional and float(entry) > 0:
        size_f = max_notional / float(entry)
        notional = abs(float(entry) * size_f)
        reasons.append("notional_capped")

    # Recompute risk cash after cap; if risk falls below tiny dust, reject.
    risk_cash = risk_per_unit * size_f
    if size_f <= 0 or risk_cash <= 0:
        return {"allow": False, "reasons": reasons + ["size_non_positive"], "size": 0.0, "notional": 0.0, "risk_cash": 0.0}

    ts = pd.Timestamp(now or datetime.now(timezone.utc))
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    day_start = ts.floor("D")
    week_start = ts - pd.Timedelta(days=int(ts.dayofweek))
    week_start = week_start.floor("D")

    def _r_sum(since: pd.Timestamp) -> float:
        total = 0.0
        for trade in closed or []:
            exit_t = trade.get("exit_time")
            if not exit_t:
                continue
            et = pd.Timestamp(exit_t)
            if et.tzinfo is None:
                et = et.tz_localize("UTC")
            if et < since:
                continue
            r = trade.get("r_multiple_net")
            if r is None:
                r = trade.get("r_multiple")
            if r is not None:
                total += float(r)
        return total

    day_r = _r_sum(day_start)
    week_r = _r_sum(week_start)
    if day_r <= -abs(day_cap):
        return {
            "allow": False,
            "reasons": ["daily_loss_stop"],
            "size": 0.0,
            "notional": 0.0,
            "risk_cash": 0.0,
            "day_r": day_r,
            "week_r": week_r,
        }
    if week_r <= -abs(week_cap):
        return {
            "allow": False,
            "reasons": ["weekly_loss_stop"],
            "size": 0.0,
            "notional": 0.0,
            "risk_cash": 0.0,
            "day_r": day_r,
            "week_r": week_r,
        }

    return {
        "allow": True,
        "reasons": reasons,
        "size": float(size_f),
        "notional": float(notional),
        "risk_cash": float(risk_cash),
        "risk_fraction": risk_frac,
        "max_notional_multiple": max_mult,
        "day_r": day_r,
        "week_r": week_r,
    }


def evaluate_data_quality(
    *,
    market_type: str | None,
    binance_feed_mode: str | None,
    stale_exchanges: list | None,
    collectors: dict | None = None,
    binance_data_path: str | None = None,
) -> dict[str, Any]:
    """Fail closed: no new paper entries unless linear futures dual-venue is healthy."""
    reasons: list[str] = []
    market = str(market_type or "").lower()
    if market != "linear":
        reasons.append(f"market_type_{market or 'unknown'}_not_linear")

    mode = str(binance_feed_mode or "").lower()
    # Accept linear / futures labels; reject spot mixes.
    if mode in ("spot", "spot_market_data", "spot_fallback"):
        reasons.append(f"binance_feed_mode_{mode}")
    if mode and mode not in ("linear", "futures", "unknown", ""):
        if "spot" in mode:
            reasons.append(f"binance_feed_mode_{mode}")

    stale = list(stale_exchanges or [])
    if "binance" in stale:
        reasons.append("binance_stale")
    if "bybit" in stale:
        reasons.append("bybit_stale")

    collectors = collectors or {}
    for venue in ("binance", "bybit"):
        status = collectors.get(venue) or {}
        venue_mode = str(status.get("mode") or "").lower()
        if venue_mode in ("spot", "spot_market_data"):
            reasons.append(f"{venue}_collector_mode_{venue_mode}")

    tradable = not reasons
    return {
        "tradable": tradable,
        "research_only": not tradable,
        "status": "tradable" if tradable else "research_only_no_trade",
        "reasons": reasons,
        "required": {
            "market_type": "linear",
            "venues_fresh": ["binance", "bybit"],
            "reject": "spot_or_mixed_or_stale",
        },
        "observed": {
            "market_type": market_type,
            "binance_feed_mode": binance_feed_mode,
            "binance_data_path": binance_data_path,
            "stale_exchanges": stale,
        },
        "policy": "if data path != linear futures dual-venue → research-only / no trade",
    }


def adverse_slip_price(price: float, side: str, is_entry: bool, slippage_bps: float) -> float:
    """Push fill against the trader (same spirit as research backtest)."""
    slip = float(slippage_bps) / 10_000.0
    # Long entry / short exit buy; short entry / long exit sell.
    if side == "long":
        return float(price) * (1 + slip) if is_entry else float(price) * (1 - slip)
    return float(price) * (1 - slip) if is_entry else float(price) * (1 + slip)


def trade_costs(
    side: str,
    entry: float,
    exit_price: float,
    size: float,
    *,
    fee_bps: float = RESEARCH_FEE_BPS,
    slippage_bps: float = RESEARCH_SLIPPAGE_BPS,
    apply_slip_to_prices: bool = True,
) -> dict[str, Any]:
    """Gross vs fee/slippage-adjusted net for one closed trade."""
    side = str(side).lower()
    entry_f = float(entry)
    exit_f = float(exit_price)
    size_f = float(size)
    if apply_slip_to_prices:
        fill_entry = adverse_slip_price(entry_f, side, True, slippage_bps)
        fill_exit = adverse_slip_price(exit_f, side, False, slippage_bps)
    else:
        fill_entry, fill_exit = entry_f, exit_f

    if side == "long":
        gross = (exit_f - entry_f) * size_f
        net_from_slipped = (fill_exit - fill_entry) * size_f
    else:
        gross = (entry_f - exit_f) * size_f
        net_from_slipped = (fill_entry - fill_exit) * size_f

    fees = (fill_entry + fill_exit) * size_f * float(fee_bps) / 10_000.0
    slippage_cost = gross - net_from_slipped
    net = net_from_slipped - fees
    risk = abs(entry_f - float(exit_price))  # placeholder; caller may pass stop risk
    return {
        "gross_pnl": round(gross, 6),
        "fees": round(fees, 6),
        "slippage_cost": round(slippage_cost, 6),
        "net_pnl": round(net, 6),
        "slipped_entry": round(fill_entry, 6),
        "slipped_exit": round(fill_exit, 6),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
    }


def rescore_closed_trade(
    trade: dict[str, Any],
    *,
    fee_bps: float = RESEARCH_FEE_BPS,
    slippage_bps: float = RESEARCH_SLIPPAGE_BPS,
) -> dict[str, Any]:
    """Recompute gross/net economics and geometry for a closed paper trade."""
    entry = float(trade["entry"])
    exit_price = float(trade["exit"])
    stop = float(trade["stop"])
    target = float(trade.get("target") or exit_price)
    size = float(trade["size"])
    side = str(trade["side"]).lower()
    costs = trade_costs(side, entry, exit_price, size, fee_bps=fee_bps, slippage_bps=slippage_bps)
    risk = abs(entry - stop)
    planned_reward = abs(entry - target)
    planned_rr = planned_reward / risk if risk else None
    gross_r = costs["gross_pnl"] / (risk * size) if risk and size else None
    net_r = costs["net_pnl"] / (risk * size) if risk and size else None
    geom = stop_geometry(entry, stop, side)
    hold_h = None
    if trade.get("entry_time") and trade.get("exit_time"):
        hold_h = (
            pd.Timestamp(trade["exit_time"]) - pd.Timestamp(trade["entry_time"])
        ).total_seconds() / 3600.0
    zone = str(trade.get("zone") or "")
    zone_kind = trade.get("zone_kind") or (zone.split(":", 1)[0] if zone else None)
    return {
        **{k: trade.get(k) for k in (
            "entry_time", "exit_time", "side", "entry", "exit", "stop", "target",
            "size", "exit_reason", "zone", "signal_id",
        )},
        "zone_kind": zone_kind,
        "planned_rr": round(planned_rr, 4) if planned_rr is not None else None,
        "stop_distance": geom["stop_distance"],
        "stop_distance_pct": geom["stop_distance_pct"],
        "stop_on_round_magnet": geom["stop_on_round_magnet"],
        "stop_magnets": geom["stop_magnets"],
        "hold_hours": round(hold_h, 4) if hold_h is not None else None,
        "gross_pnl": costs["gross_pnl"],
        "fees": costs["fees"],
        "slippage_cost": costs["slippage_cost"],
        "net_pnl": costs["net_pnl"],
        "r_multiple_gross": round(gross_r, 4) if gross_r is not None else None,
        "r_multiple_net": round(net_r, 4) if net_r is not None else None,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "ledger_pnl_field": trade.get("pnl"),
        "ledger_r_field": trade.get("r_multiple"),
    }


def rescore_seeded_trades(
    trades: list[dict] | None = None,
    *,
    fee_bps: float = RESEARCH_FEE_BPS,
    slippage_bps: float = RESEARCH_SLIPPAGE_BPS,
) -> dict[str, Any]:
    if trades is None:
        # Lazy import avoids circular dependency with paper_position.
        from btc_predictor.paper_position import HISTORICAL_SEEDED_TRADES
        trades = HISTORICAL_SEEDED_TRADES
    rows = [rescore_closed_trade(t, fee_bps=fee_bps, slippage_bps=slippage_bps) for t in trades]
    gross = sum(r["gross_pnl"] for r in rows)
    net = sum(r["net_pnl"] for r in rows)
    fees = sum(r["fees"] for r in rows)
    slip = sum(r["slippage_cost"] for r in rows)
    r_gross = sum(r["r_multiple_gross"] or 0 for r in rows)
    r_net = sum(r["r_multiple_net"] or 0 for r in rows)
    sequence = " → ".join(
        f"{r['side'][0].upper()}@{r['entry']:.0f}/{r['exit_reason']}" for r in rows
    )
    return {
        "economics": research_economics(),
        "trades": rows,
        "summary": {
            "count": len(rows),
            "gross_pnl": round(gross, 2),
            "fees": round(fees, 2),
            "slippage_cost": round(slip, 2),
            "net_pnl": round(net, 2),
            "sum_r_gross": round(r_gross, 4),
            "sum_r_net": round(r_net, 4),
            "expectancy_r_net": round(r_net / len(rows), 4) if rows else None,
            "do_not_treat_gross_as_alpha": True,
            "sequence": sequence,
            "narrative": (
                "stop_at_magnet → re-entry_new_zone → continuation_tight_breakout"
                if len(rows) >= 3
                else None
            ),
        },
        "note": "n=3 case study only — not OOS edge.",
    }


def funnel_category(prediction: Any) -> str:
    """Map a predictor output to a live funnel bucket."""
    bias = _attr(prediction, "bias")
    if bias not in ("bullish", "bearish"):
        return "bias_neutral"
    sweep = str(_attr(prediction, "sweep_status") or "")
    reason = str(_attr(prediction, "no_trade_reason") or "")
    of_conf = _attr(prediction, "orderflow_confirmation")
    if reason == "insufficient_reward_risk":
        return "insufficient_rr"
    if reason == "orderflow_not_confirmed" or (sweep == "confirmed" and of_conf is False):
        return "flow_reject"
    if reason == "sweep_not_confirmed" or sweep in (
        "none", "waiting_reclaim", "approaching", "shallow_excursion",
        "excessive_excursion", "expired_reclaim",
    ):
        if sweep == "confirmed":
            pass
        else:
            return "sweep_waiting"
    if of_conf is True and _attr(prediction, "entry") is not None and not reason:
        return "flow_pass_actionable"
    if reason:
        return f"other:{reason}"
    if sweep == "confirmed":
        return "sweep_confirmed_other"
    return "bias_directional_other"


def _attr(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


FUNNEL_KEYS = (
    "bias_neutral",
    "bias_directional_other",
    "sweep_waiting",
    "sweep_confirmed_other",
    "flow_reject",
    "flow_pass_actionable",
    "insufficient_rr",
    "paper_entries",
    "paper_exits",
    "soft_filter_skips",
    "risk_cap_skips",
    "data_quality_blocks",
    "decision_bars",
)


def empty_funnel_counts() -> dict[str, int]:
    return {k: 0 for k in FUNNEL_KEYS}


def week_id(ts=None) -> str:
    t = pd.Timestamp(ts or datetime.now(timezone.utc))
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    iso = t.isocalendar()
    return f"{iso.year}-W{int(iso.week):02d}"


def shadow_rule_skip(snapshot: dict[str, Any], rule: str | None = None) -> dict[str, Any]:
    """Book B extra skip on top of production (Book A). Forward-only."""
    rule = rule or SHADOW_RULE
    snap = enrich_decision_snapshot(snapshot)
    rr = snap.get("planned_rr") or snap.get("reward_risk")
    skip = False
    detail = None
    if rule == "skip_planned_rr_above_2_5":
        if rr is not None and float(rr) > SHADOW_RR_CAP:
            skip = True
            detail = f"planned_rr={float(rr):.3f}>{SHADOW_RR_CAP:g}"
    elif rule == "skip_stop_within_magnet_dollars":
        magnets = (snap.get("stop_magnets") or {})
        dists = [magnets.get(k) for k in ("dist_100", "dist_500", "dist_1000") if magnets.get(k) is not None]
        if dists and min(dists) <= SHADOW_MAGNET_DOLLARS:
            skip = True
            detail = f"stop_magnet_dist={min(dists):.2f}<={SHADOW_MAGNET_DOLLARS:g}"
    elif rule == "skip_untested_breakout":
        kind = snap.get("zone_kind") or str(snap.get("zone") or "").split(":", 1)[0]
        if kind == "untested_breakout":
            skip = True
            detail = "zone_kind=untested_breakout"
    else:
        detail = f"unknown_rule:{rule}"
    return {
        "rule": rule,
        "skip": skip,
        "detail": detail,
        "forward_only": True,
        "validated": False,
        "snapshot_zone": snap.get("zone"),
        "planned_rr": rr,
    }


def calibration_status(artifact: dict | None, config: dict | None = None) -> dict[str, Any]:
    """Read-only summary of any existing flow_calibration artifact."""
    config = config or {}
    if not artifact:
        return {
            "present": False,
            "promotion_passed": False,
            "action": "stay_on_independent_do_not_invent_thresholds",
            "gate_mode": config.get("gate_mode"),
            "thresholds": {
                "legacy": config.get("legacy_threshold"),
                "market": config.get("market_threshold"),
                "raw": config.get("raw_threshold"),
            },
            "note": "No calibration artifact loaded. Keep FLOW_GATE_MODE=independent; do not retune 0.40 from n=3.",
        }
    passed = bool(artifact.get("promotion_passed"))
    selected = artifact.get("selected_config") or {}
    oos = artifact.get("oos_summary") or artifact.get("out_of_sample") or artifact.get("promotion")
    return {
        "present": True,
        "promotion_passed": passed,
        "run_hash": artifact.get("run_hash"),
        "selected_config": selected,
        "oos_summary": oos,
        "fallback_reason": config.get("fallback_reason"),
        "gate_mode": config.get("gate_mode"),
        "thresholds": {
            "legacy": config.get("legacy_threshold"),
            "market": config.get("market_threshold"),
            "raw": config.get("raw_threshold"),
        },
        "action": (
            "consider_promotion_only_after_manual_report_review"
            if passed
            else "stay_on_independent_or_shadow_do_not_invent_thresholds"
        ),
        "note": (
            "Artifact reports promotion_passed; review OOS report before changing live mode."
            if passed
            else "Artifact missing or failed promotion. Stay independent; do not invent new 0.40s."
        ),
    }


def write_seeded_rescore_report(path: str | Path) -> dict[str, Any]:
    report = rescore_seeded_trades()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def retune_discipline_status(closed_trades: int, policy_effective_at: str | None = None) -> dict[str, Any]:
    effective = pd.Timestamp(policy_effective_at or POLICY_EFFECTIVE_AT)
    if effective.tzinfo is None:
        effective = effective.tz_localize("UTC")
    now = pd.Timestamp.now(tz="UTC")
    days_elapsed = max(0, int((now - effective).total_seconds() // 86400))
    days_remaining = max(0, RETUNE_CALENDAR_DAYS - days_elapsed)
    trades_remaining = max(0, MIN_CLOSED_BEFORE_RETUNE - int(closed_trades or 0))
    allowed = int(closed_trades or 0) >= MIN_CLOSED_BEFORE_RETUNE or days_elapsed >= RETUNE_CALENDAR_DAYS
    return {
        "parameter_changes_allowed": allowed,
        "closed_trades": int(closed_trades or 0),
        "min_closed_before_retune": MIN_CLOSED_BEFORE_RETUNE,
        "trades_remaining": trades_remaining,
        "policy_effective_at": effective.isoformat(),
        "review_by": (effective + pd.Timedelta(days=RETUNE_CALENDAR_DAYS)).isoformat(),
        "days_elapsed": days_elapsed,
        "days_remaining_on_calendar": days_remaining,
        "review_metrics_only": list(REVIEW_METRICS),
        "do_not_retune_flow_thresholds_from_small_n": True,
    }


class FunnelDiary:
    """Weekly live funnel counters (friction map, not counterfactual edge)."""

    def __init__(self, store):
        self.store = store

    def _load(self) -> dict:
        data = self.store.read({"weeks": {}, "updated_at": None}) if self.store else {"weeks": {}, "updated_at": None}
        data.setdefault("weeks", {})
        return data

    def record(self, category: str, *, ts=None, n: int = 1) -> dict:
        if not self.store:
            return {}
        data = self._load()
        wid = week_id(ts)
        week = data["weeks"].setdefault(wid, empty_funnel_counts())
        for key in FUNNEL_KEYS:
            week.setdefault(key, 0)
        if category.startswith("other:"):
            week["bias_directional_other"] = int(week.get("bias_directional_other") or 0) + n
            other = week.setdefault("other_reasons", {})
            other[category[6:]] = int(other.get(category[6:], 0) or 0) + n
        elif category in week:
            week[category] = int(week.get(category) or 0) + n
        else:
            week["bias_directional_other"] = int(week.get("bias_directional_other") or 0) + n
        data["updated_at"] = pd.Timestamp(ts or datetime.now(timezone.utc)).isoformat()
        data["weeks"][wid] = week
        # Retain ~26 weeks.
        if len(data["weeks"]) > 26:
            for old in sorted(data["weeks"])[:-26]:
                data["weeks"].pop(old, None)
        self.store.write(data)
        return week

    def record_prediction(self, prediction, *, ts=None, blocked_data: bool = False) -> dict:
        if blocked_data:
            self.record("data_quality_blocks", ts=ts)
        cat = funnel_category(prediction)
        week = self.record(cat, ts=ts)
        self.record("decision_bars", ts=ts)
        return week

    def status(self) -> dict:
        data = self._load()
        wid = week_id()
        current = data.get("weeks", {}).get(wid) or empty_funnel_counts()
        return {
            "current_week": wid,
            "counts": current,
            "weeks": data.get("weeks") or {},
            "updated_at": data.get("updated_at"),
            "note": "Funnel shows where live decisions die; rejects are not proven edge filters.",
        }


class DecisionSnapshotLog:
    """Append-only recent decision snapshots for future calibration."""

    def __init__(self, store, max_rows: int = 2000):
        self.store = store
        self.max_rows = max(100, int(max_rows))

    def append(self, snapshot: dict, *, meta: dict | None = None) -> None:
        if not self.store:
            return
        data = self.store.read({"rows": []})
        rows = list(data.get("rows") or [])
        row = {
            "recorded_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "snapshot": enrich_decision_snapshot(snapshot),
            "meta": meta or {},
        }
        rows.append(row)
        data["rows"] = rows[-self.max_rows:]
        data["count"] = len(data["rows"])
        data["updated_at"] = row["recorded_at"]
        self.store.write(data)

    def status(self) -> dict:
        data = self.store.read({"rows": [], "count": 0}) if self.store else {"rows": [], "count": 0}
        return {
            "count": data.get("count") or len(data.get("rows") or []),
            "updated_at": data.get("updated_at"),
            "latest": (data.get("rows") or [None])[-1],
        }


class ShadowBook:
    """Forward-only Book B: production rules + one extra skip rule."""

    def __init__(self, store, rule: str | None = None):
        self.store = store
        self.rule = rule or SHADOW_RULE

    def _load(self) -> dict:
        if not self.store:
            return self._empty()
        data = self.store.read(self._empty())
        data.setdefault("signals", [])
        data.setdefault("skipped", [])
        data.setdefault("taken", [])
        return data

    def _empty(self) -> dict:
        return {
            "rule": self.rule,
            "forward_only": True,
            "validated": False,
            "book_a": "production_paper",
            "signals": [],
            "skipped": [],
            "taken": [],
            "updated_at": None,
        }

    def observe_confirmed(self, event: dict) -> dict:
        """Record a Book-A confirmed setup and whether Book B would skip it."""
        if not self.store:
            return {}
        snapshot = enrich_decision_snapshot((event or {}).get("snapshot") or {})
        decision = shadow_rule_skip(snapshot, self.rule)
        row = {
            "observed_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "signal_id": (event or {}).get("signal_id"),
            "lifecycle_event_id": (event or {}).get("event_id"),
            "zone": snapshot.get("zone"),
            "planned_rr": snapshot.get("planned_rr") or snapshot.get("reward_risk"),
            "entry": snapshot.get("entry"),
            "stop": snapshot.get("stop"),
            "target": snapshot.get("target"),
            "book_b": decision,
        }
        data = self._load()
        data["rule"] = self.rule
        data["signals"] = (list(data.get("signals") or []) + [row])[-500:]
        if decision.get("skip"):
            data["skipped"] = (list(data.get("skipped") or []) + [row])[-500:]
        else:
            data["taken"] = (list(data.get("taken") or []) + [row])[-500:]
        data["updated_at"] = row["observed_at"]
        data["counts"] = {
            "signals": len(data["signals"]),
            "book_b_taken": len(data["taken"]),
            "book_b_skipped": len(data["skipped"]),
        }
        self.store.write(data)
        return row

    def status(self) -> dict:
        data = self._load()
        return {
            "rule": data.get("rule") or self.rule,
            "forward_only": True,
            "validated": False,
            "do_not_promote_early": True,
            "book_a": "production_paper",
            "counts": data.get("counts") or {
                "signals": len(data.get("signals") or []),
                "book_b_taken": len(data.get("taken") or []),
                "book_b_skipped": len(data.get("skipped") or []),
            },
            "latest": (data.get("signals") or [None])[-1],
            "updated_at": data.get("updated_at"),
            "note": "Shadow comparison is forward-only; low power until N grows.",
        }


class OpsReliability:
    """Lightweight ops snapshot for weekly review."""

    def __init__(self, store):
        self.store = store

    def record(self, event: str, detail: dict | None = None) -> None:
        if not self.store:
            return
        data = self.store.read({"events": [], "seed_or_repair": []})
        row = {
            "at": pd.Timestamp.now(tz="UTC").isoformat(),
            "event": event,
            "detail": detail or {},
        }
        data["events"] = (list(data.get("events") or []) + [row])[-200:]
        if event in ("ledger_seeded", "ledger_repaired", "ledger_loaded"):
            data["seed_or_repair"] = (list(data.get("seed_or_repair") or []) + [row])[-50:]
        data["updated_at"] = row["at"]
        self.store.write(data)

    def status(self, *, paper_status: dict | None = None, live_loop: dict | None = None, data_quality: dict | None = None) -> dict:
        data = self.store.read({"events": [], "seed_or_repair": []}) if self.store else {"events": [], "seed_or_repair": []}
        paper_status = paper_status or {}
        return {
            "live_loop": live_loop or {},
            "paper": {
                "open": bool(paper_status.get("open_position")),
                "pending": bool(paper_status.get("pending_order")),
                "closed_trades": paper_status.get("closed_trades"),
                "equity_net": paper_status.get("equity_net") or paper_status.get("equity"),
                "equity_gross": paper_status.get("equity_gross"),
                "last_closed": paper_status.get("last_closed"),
            },
            "data_quality": data_quality or {},
            "seed_or_repair_events": data.get("seed_or_repair") or [],
            "recent_events": (data.get("events") or [])[-20:],
            "updated_at": data.get("updated_at"),
            "checklist": [
                "single_live_loop_lock",
                "ledger_durability",
                "exit_push",
                "health_endpoints",
                "feed_uptime",
                "open_vs_closed_paper_state",
            ],
        }
