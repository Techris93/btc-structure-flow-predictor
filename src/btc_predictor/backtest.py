from __future__ import annotations

from collections import Counter
import math
import pandas as pd

from .strategy import Predictor
from .structure import structure_events
from .timeframes import completed_timeframes


def _close_trade(open_trade, exit_price, exit_time, equity, fee_bps, reason):
    side_sign = 1 if open_trade["side"] == "long" else -1
    gross = (exit_price - open_trade["entry"]) * open_trade["size"] * side_sign
    entry_fee_bps=float(open_trade.get("entry_fee_bps",fee_bps))
    fees = (open_trade["entry"]*entry_fee_bps + exit_price*fee_bps) * open_trade["size"] / 10_000
    pnl = gross - fees
    equity += pnl
    risk_cash = abs(open_trade["entry"] - open_trade["stop"]) * open_trade["size"]
    row = {**open_trade, "exit": exit_price, "pnl": pnl, "equity": equity, "exit_time": exit_time,
           "exit_reason": reason, "r_multiple": pnl / risk_cash if risk_cash else None,
           "hold_minutes": (pd.Timestamp(exit_time) - pd.Timestamp(open_trade["entry_time"])).total_seconds() / 60}
    return row, equity


def run_event_backtest(
    ohlc: pd.DataFrame,
    trades: pd.DataFrame,
    predictor=None,
    initial_equity=100_000,
    fee_bps=5,
    maker_fee_bps=2,
    slippage_bps=2,
    decision_stride: int = 1,
    analysis_lookback_bars: int = 400,
    mode: str = "reactive",
    same_bar_policy: str = "conservative",
    force_close: bool = True,
    decision_start=None,
    progress=None,
    resume_state: dict | None = None,
    checkpoint=None,
    features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Causal replay: decide after close, fill at the next open, never expose future trades."""
    if mode not in {"reactive", "mtf"}:
        raise ValueError("mode must be reactive or mtf")
    if same_bar_policy not in {"conservative", "optimistic"}:
        raise ValueError("same_bar_policy must be conservative or optimistic")
    predictor = predictor or Predictor()
    state = resume_state or {}
    equity = float(state.get("equity", initial_equity))
    records = list(state.get("records", [])); open_trade = state.get("open_trade"); pending = state.get("pending")
    rejections, collisions = Counter(state.get("rejections", {})), int(state.get("collisions", 0))
    candidate_rejections = Counter(state.get("candidate_rejections", {}))
    attempted_setups = set(state.get("attempted_setups", []))
    if state.get("held_bias") in {"bullish","bearish","neutral"}: predictor._held_bias = state["held_bias"]
    bars = ohlc.sort_index().copy()
    bars.index = pd.to_datetime(bars.index, utc=True)
    if decision_start is not None:
        decision_start = pd.Timestamp(decision_start)
        decision_start = decision_start.tz_localize("UTC") if decision_start.tzinfo is None else decision_start.tz_convert("UTC")
    trade_data = trades.copy()
    trade_data["time"] = pd.to_datetime(trade_data.time, utc=True)
    trade_data = trade_data.sort_values("time")
    # Keep datetime values for searchsorted: integer representations can have
    # different units (us/ns) across pandas inputs and admit future rows.
    trade_times = trade_data["time"].reset_index(drop=True)
    feature_index = None
    if features is not None and len(features):
        features = features.sort_index()
        features.index = pd.to_datetime(features.index, utc=True)
        feature_index = features.index
    # Zone/risk construction is always 15m. Reactive mode keeps its 1m bias,
    # while MTF additionally receives the completed 1h/4h regime frames.
    derived = completed_timeframes(bars)
    frame_slice_cache = {}
    reactive_events=structure_events(bars) if mode=="reactive" and isinstance(predictor,Predictor) else None
    reactive_event_times=reactive_events.index if reactive_events is not None and not reactive_events.empty else None

    for i in range(max(80, int(state.get("next_i", 80))), len(bars)):
        now, b = bars.index[i], bars.iloc[i]
        # A signal from the prior close is filled only now, at this bar's open.
        if pending is not None and open_trade is None:
            limit_price = pending.get("limit_price")
            if limit_price is None:
                sign = 1 if pending["side"] == "long" else -1
                fill = float(b.open) * (1 + sign * slippage_bps / 10_000)
                open_trade = {**pending, "entry_time": now, "entry": fill, "entry_fee_bps":fee_bps}
                pending = None
            else:
                valid_until = pending.get("valid_until")
                if valid_until is not None and now > pd.Timestamp(valid_until):
                    rejections["limit_expired"] += 1
                    pending = None
                else:
                    is_long = pending["side"] == "long"
                    touched = float(b.low) <= limit_price if is_long else float(b.high) >= limit_price
                    if touched:
                        # Passive limits receive price improvement on gaps and no
                        # adverse entry slippage; maker fees are charged separately.
                        raw_fill = min(float(b.open), limit_price) if is_long else max(float(b.open), limit_price)
                        open_trade = {**pending, "entry_time": now, "entry":raw_fill, "entry_fee_bps":maker_fee_bps}
                        pending = None

        if open_trade is not None:
            side = open_trade["side"]
            hit_stop = b.low <= open_trade["stop"] if side == "long" else b.high >= open_trade["stop"]
            hit_target = b.high >= open_trade["target"] if side == "long" else b.low <= open_trade["target"]
            max_hold = open_trade.get("max_holding_minutes")
            held = (now - pd.Timestamp(open_trade["entry_time"])).total_seconds() / 60
            time_exit = max_hold is not None and held >= max_hold
            if time_exit and not (hit_stop or hit_target):
                sign = -1 if side == "long" else 1
                exit_price = float(b.close) * (1 + sign * slippage_bps / 10_000)
                row, equity = _close_trade(open_trade, exit_price, now, equity, fee_bps, "time_exit")
                records.append(row); open_trade = None
        if open_trade is not None:
            side = open_trade["side"]
            hit_stop = b.low <= open_trade["stop"] if side == "long" else b.high >= open_trade["stop"]
            hit_target = b.high >= open_trade["target"] if side == "long" else b.low <= open_trade["target"]
            if hit_stop or hit_target:
                if hit_stop and hit_target:
                    collisions += 1
                    use_stop = same_bar_policy == "conservative"
                else:
                    use_stop = hit_stop
                raw_exit = open_trade["stop"] if use_stop else open_trade["target"]
                sign = -1 if side == "long" else 1
                exit_price = float(raw_exit) * (1 + sign * slippage_bps / 10_000)
                row, equity = _close_trade(open_trade, exit_price, now, equity, fee_bps, "stop" if use_stop else "target")
                records.append(row); open_trade = None

        decisions_open = decision_start is None or now >= pd.Timestamp(decision_start)
        if decisions_open and open_trade is None and i % decision_stride == 0 and i < len(bars) - 1:
            history = bars.iloc[max(0, i-analysis_lookback_bars+1):i+1]
            # Strict cutoff is the decision timestamp. Exchange events after it are invisible.
            trade_end = trade_times.searchsorted(pd.Timestamp(now), side="left")
            tt = trade_data.iloc[max(0, trade_end-3000):trade_end]
            context = None
            if feature_index is not None:
                fpos = feature_index.searchsorted(pd.Timestamp(now), side="right") - 1
                if fpos >= 0:
                    context = {k: (None if pd.isna(v) else float(v))
                               for k, v in features.iloc[fpos].items()}
            frames = {}
            frame_names=("15m",) if mode=="reactive" else ("15m","1h","4h")
            for name in frame_names:
                frame=derived[name]
                pos=frame.index.searchsorted(now,side="right")
                ckey=(name,pos)
                if ckey not in frame_slice_cache:
                    frame_slice_cache[ckey]=frame.iloc[max(0,pos-400):pos]
                frames[name]=frame_slice_cache[ckey]
            if reactive_event_times is not None:
                event_pos=reactive_event_times.searchsorted(now,side="right")
                reactive_bias=str(reactive_events.iloc[event_pos-1].bias) if event_pos else "neutral"
                extra={"bias_override":reactive_bias}
            else:
                extra={}
            if context is not None:
                extra["context"]=context
            out=predictor.predict(history,tt,equity,frames=frames,**extra)
            if out.entry is not None and out.position_size:
                side = "long" if out.bias == "bullish" else "short"
                if getattr(out, "setup_type", None) == "continuation":
                    # Continuation setups share zone/sweep fields; bucket by 4h so
                    # re-entry is possible after a cooldown but not every bar.
                    setup_key = f"{side}|continuation|{pd.Timestamp(now).floor('4h')}"
                else:
                    setup_key="|".join(str(v or "") for v in (side,out.zone,out.sweep_time,out.reclaim_time))
                if setup_key in attempted_setups:
                    rejections["duplicate_setup"] += 1
                else:
                    if pending is not None and (pending["side"], pending["zone"]) != (side, out.zone):
                        # Live-ledger parity: a new distinct setup supersedes the working order.
                        rejections["pending_superseded"] += 1
                        pending = None
                    if pending is None:
                        attempted_setups.add(setup_key)
                        pending = {"decision_time": now, "side": side,
                                   "signal_entry": out.entry, "stop": out.stop, "target": out.target,
                               "size": out.position_size, "zone": out.zone, "setup_type": out.setup_type,
                               "entry_type":getattr(out,"entry_type","market"),
                                   "limit_price": out.entry if getattr(out, "entry_type", "market") == "limit" else None,
                                   "valid_until": getattr(out, "entry_expires_at", None),
                                   "max_holding_minutes": getattr(out, "max_holding_minutes", None)}
            elif pending is None:
                rejections[out.no_trade_reason or "unknown"] += 1
                for entry_type, reason in (getattr(out,"candidate_rejections",None) or {}).items():
                    candidate_rejections[f"{entry_type}:{reason}"] += 1
        if progress and (i % 1000 == 0 or i == len(bars)-1):
            progress(i + 1, len(bars), records)
        if checkpoint and i % 1000 == 0:
            checkpoint({"next_i":i+1,"equity":equity,"records":records,"open_trade":open_trade,"pending":pending,
                        "rejections":dict(rejections),"candidate_rejections":dict(candidate_rejections),
                        "attempted_setups":sorted(attempted_setups),"collisions":collisions,"held_bias":predictor._held_bias})

    open_position_status = None
    if open_trade is not None:
        if force_close:
            final_time, final_price = bars.index[-1], float(bars.close.iloc[-1])
            sign = -1 if open_trade["side"] == "long" else 1
            final_price *= 1 + sign * slippage_bps / 10_000
            row, equity = _close_trade(open_trade, final_price, final_time, equity, fee_bps, "end_of_data")
            records.append(row)
        else:
            open_position_status = open_trade
    elif pending is not None:
        open_position_status = {**pending, "status": "unfilled_end_of_data"}

    result = pd.DataFrame(records)
    wins = int((result.pnl > 0).sum()) if len(result) else 0
    losses = int((result.pnl <= 0).sum()) if len(result) else 0
    gross_profit = float(result.loc[result.pnl > 0, "pnl"].sum()) if len(result) else 0.0
    gross_loss = float(-result.loc[result.pnl < 0, "pnl"].sum()) if len(result) else 0.0
    equity_curve = pd.concat([pd.Series([initial_equity]), result.equity.reset_index(drop=True)]) if len(result) else pd.Series([initial_equity])
    drawdown = equity_curve.cummax() - equity_curve
    stats = {
        "initial_equity": initial_equity, "final_equity": equity, "trades": len(result),
        "wins": wins, "losses": losses, "win_rate": wins / len(result) if len(result) else 0.0,
        "gross_profit": gross_profit, "gross_loss": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss else (math.inf if gross_profit else None),
        "net_pnl": equity - initial_equity, "maximum_drawdown": float(drawdown.max()),
        "average_r": float(result.r_multiple.mean()) if len(result) else None,
        "average_hold_minutes": float(result.hold_minutes.mean()) if len(result) else None,
        "same_bar_collisions": collisions, "same_bar_policy": same_bar_policy,
        "open_position": open_position_status, "rejection_counts": dict(rejections),
        "candidate_rejection_counts": dict(candidate_rejections),
        "causality": "close-time decision; next-open fill; trades < decision time",
    }
    return result, stats


def walk_forward_splits(index, train_bars=500, test_bars=100, step=None):
    step = step or test_bars; i = train_bars
    while i + test_bars <= len(index):
        yield index[:i], index[i:i+test_bars]
        i += step
