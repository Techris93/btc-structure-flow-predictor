from __future__ import annotations
import pandas as pd
from .strategy import Predictor

def run_event_backtest(ohlc: pd.DataFrame, trades: pd.DataFrame, predictor=None, initial_equity=100_000, fee_bps=5, slippage_bps=2, decision_stride: int = 5) -> tuple[pd.DataFrame, dict]:
    predictor=predictor or Predictor(); equity=initial_equity; records=[]; open_trade=None
    bars=ohlc.sort_index()
    for i in range(80,len(bars)):
        now=bars.index[i]; history=bars.iloc[:i+1]; tt=trades[pd.to_datetime(trades.time,utc=True)<=now]
        if open_trade:
            b=bars.iloc[i]; side=open_trade["side"]; hit_stop=(b.low<=open_trade["stop"] if side=="long" else b.high>=open_trade["stop"]); hit_target=(b.high>=open_trade["target"] if side=="long" else b.low<=open_trade["target"])
            if hit_stop or hit_target:
                exit_price=open_trade["stop"] if hit_stop else open_trade["target"]; gross=(exit_price-open_trade["entry"])*open_trade["size"]*(1 if side=="long" else -1); costs=(open_trade["entry"]+exit_price)*open_trade["size"]*(fee_bps+slippage_bps)/10000; pnl=gross-costs; equity+=pnl; records.append({**open_trade,"exit":exit_price,"pnl":pnl,"equity":equity,"exit_time":now}); open_trade=None
        if open_trade is None and i % decision_stride == 0:
            out=predictor.predict(history,tt,equity)
            if out.entry and out.position_size:
                open_trade={"entry_time":now,"side":"long" if out.bias=="bullish" else "short","entry":out.entry,"stop":out.stop,"target":out.target,"size":out.position_size,"zone":out.zone}
    result=pd.DataFrame(records); stats={"initial_equity":initial_equity,"final_equity":equity,"trades":len(result),"net_pnl":equity-initial_equity,"win_rate":float((result.pnl>0).mean()) if len(result) else 0.0}
    return result,stats

def walk_forward_splits(index, train_bars=500, test_bars=100, step=None):
    step=step or test_bars; i=train_bars
    while i+test_bars <= len(index): yield index[:i], index[i:i+test_bars]; i+=step
