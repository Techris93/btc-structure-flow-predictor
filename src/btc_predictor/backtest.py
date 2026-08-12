from __future__ import annotations

from collections import Counter
import math
import pandas as pd

from .strategy import Predictor
from .timeframes import completed_timeframes


def _adapt_predictor_for_available_flow(predictor, trades):
    """Independent/calibrated gates need two venues. Proxy klines do not have them."""
    if predictor is None:
        from .research import predictor_for_replay
        return predictor_for_replay(trades), "shadow_proxy_default"
    mode = str(getattr(predictor, "flow_gate_mode", "") or "")
    has_exchange = trades is not None and hasattr(trades, "columns") and "exchange" in trades.columns
    venues = set()
    if has_exchange and not getattr(trades, "empty", True):
        venues = {str(value).lower() for value in trades.exchange.dropna().unique()}
    if mode in ("independent", "calibrated") and not {"binance", "bybit"}.issubset(venues):
        predictor.flow_gate_mode = "shadow"
        return predictor, "shadow_because_two_venue_flow_unavailable"
    return predictor, None


def _close_trade(open_trade, exit_price, exit_time, equity, fee_bps, reason):
    side_sign = 1 if open_trade["side"] == "long" else -1
    gross = (exit_price - open_trade["entry"]) * open_trade["size"] * side_sign
    fees = (open_trade["entry"] + exit_price) * open_trade["size"] * fee_bps / 10_000
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
    slippage_bps=2,
    decision_stride: int = 1,
    analysis_lookback_bars: int = 400,
    mode: str = "reactive",
    same_bar_policy: str = "conservative",
    force_close: bool = True,
    progress=None,
    resume_state: dict | None = None,
    checkpoint=None,
) -> tuple[pd.DataFrame, dict]:
    """Causal replay: decide after close, fill at the next open, never expose future trades."""
    if mode not in {"reactive", "mtf"}:
        raise ValueError("mode must be reactive or mtf")
    if same_bar_policy not in {"conservative", "optimistic"}:
        raise ValueError("same_bar_policy must be conservative or optimistic")
    predictor, gate_adapt = _adapt_predictor_for_available_flow(predictor, trades)
    state = resume_state or {}
    equity = float(state.get("equity", initial_equity))
    records = list(state.get("records", [])); open_trade = state.get("open_trade"); pending = state.get("pending")
    rejections, collisions = Counter(state.get("rejections", {})), int(state.get("collisions", 0))
    if state.get("held_bias") in {"bullish","bearish","neutral"}: predictor._held_bias = state["held_bias"]
    bars = ohlc.sort_index().copy()
    bars.index = pd.to_datetime(bars.index, utc=True)
    trade_data = trades.copy()
    trade_data["time"] = pd.to_datetime(trade_data.time, utc=True)
    trade_data = trade_data.sort_values("time")
    # Keep datetime values for searchsorted: integer representations can have
    # different units (us/ns) across pandas inputs and admit future rows.
    trade_times = trade_data["time"].reset_index(drop=True)
    derived = completed_timeframes(bars) if mode == "mtf" else None
    # searchsorted frame windows: avoid O(n) boolean masks every decision bar
    # (that path slowed from ~300 bars/s to ~20 bars/s as history grew).
    frame_cache = {name: {"position": None, "frame": None} for name in (derived or {})}

    for i in range(max(80, int(state.get("next_i", 80))), len(bars)):
        now, b = bars.index[i], bars.iloc[i]
        # A signal from the prior close is filled only now, at this bar's open.
        if pending is not None and open_trade is None:
            sign = 1 if pending["side"] == "long" else -1
            fill = float(b.open) * (1 + sign * slippage_bps / 10_000)
            open_trade = {**pending, "entry_time": now, "entry": fill}
            pending = None

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

        if open_trade is None and pending is None and i % decision_stride == 0 and i < len(bars) - 1:
            history = bars.iloc[max(0, i-analysis_lookback_bars+1):i+1]
            # Strict cutoff is the decision timestamp. Exchange events after it are invisible.
            trade_end = trade_times.searchsorted(pd.Timestamp(now), side="left")
            tt = trade_data.iloc[max(0, trade_end-3000):trade_end]
            frames = None
            if mode == "mtf" and derived is not None:
                frames = {}
                for name, frame in derived.items():
                    position = int(frame.index.searchsorted(now, side="right"))
                    cached = frame_cache[name]
                    if cached["position"] != position:
                        cached["position"] = position
                        cached["frame"] = frame.iloc[max(0, position - 400):position]
                    frames[name] = cached["frame"]
            flow_bars = history if "taker_buy_volume" in history.columns else None
            out = predictor.predict(
                history,
                tt,
                equity,
                frames=frames,
                flow_bars=flow_bars,
                flow_source="historical_binance_kline" if flow_bars is not None else None,
            )
            if out.entry is not None and out.position_size:
                pending = {"decision_time": now, "side": "long" if out.bias == "bullish" else "short",
                           "signal_entry": out.entry, "stop": out.stop, "target": out.target,
                           "size": out.position_size, "zone": out.zone, "setup_type": out.setup_type}
            else:
                rejections[out.no_trade_reason or "unknown"] += 1
        if progress and (i % 500 == 0 or i == len(bars)-1):
            progress(i + 1, len(bars), records)
        # Checkpoint less often once large — JSON of full records is a bottleneck.
        if checkpoint and (i % 5000 == 0 or i == len(bars) - 1):
            checkpoint({"next_i":i+1,"equity":equity,"records":records,"open_trade":open_trade,"pending":pending,
                        "rejections":dict(rejections),"collisions":collisions,"held_bias":predictor._held_bias})

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
        "causality": "close-time decision; next-open fill; trades < decision time",
        "flow_gate_mode": getattr(predictor, "flow_gate_mode", None),
        "flow_gate_adapt": gate_adapt,
    }
    return result, stats


def walk_forward_splits(index, train_bars=500, test_bars=100, step=None):
    step = step or test_bars; i = train_bars
    while i + test_bars <= len(index):
        yield index[:i], index[i:i+test_bars]
        i += step
