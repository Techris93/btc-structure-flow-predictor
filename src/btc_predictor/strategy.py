from __future__ import annotations
import pandas as pd
import numpy as np
from .indicators import atr
from .models import PredictorOutput, Zone
from .structure import structure_events
from .zones import build_projected_zones
from .footprint import orderflow_features

class Predictor:
    def __init__(self, risk_fraction=.0025, atr_mult=1.5, min_rr=1.5, sweep_atr=(.05, 2.0), flow_freq="1min"):
        self.risk_fraction, self.atr_mult, self.min_rr, self.sweep_atr, self.flow_freq = risk_fraction, atr_mult, min_rr, sweep_atr, flow_freq
        self._held_bias = "neutral"

    def _regime_bias(self, frames: dict[str, pd.DataFrame]) -> str:
        signals, event_sets = [], []
        for name in ("4h", "1h"):
            frame = frames.get(name)
            if frame is None or len(frame) < 40: return "neutral"
            events = structure_events(frame)
            event_sets.append(events)
            signals.append(events.iloc[-1].bias if not events.empty else "neutral")
        candidate = signals[0] if signals[0] == signals[1] else "neutral"
        # A disagreement is externally neutral.  The held regime is state only.
        if candidate == "neutral":
            return "neutral"
        # A held directional regime can only reverse after an explicitly
        # confirmed opposing CHoCH on the higher-timeframe frames.
        if self._held_bias in ("bullish", "bearish") and candidate in ("bullish", "bearish") and candidate != self._held_bias:
            opposing_choch = any(not e.empty and e.iloc[-1].event == "CHoCH" and e.iloc[-1].bias == candidate for e in event_sets)
            if not opposing_choch:
                return "neutral"
        if candidate in ("bullish", "bearish"): self._held_bias = candidate
        return candidate

    def predict(self, ohlc: pd.DataFrame, trades: pd.DataFrame, equity: float = 100_000, frames: dict[str, pd.DataFrame] | None = None) -> PredictorOutput:
        if len(ohlc) < 80 or trades.empty: return PredictorOutput(ohlc.index[-1], "neutral", no_trade_reason="insufficient_history")
        o=ohlc.copy(); o.index=pd.to_datetime(o.index,utc=True); t=trades.copy(); t["time"]=pd.to_datetime(t.time,utc=True)
        frames = frames or {}
        setup = frames.get("15m", o)
        bias = self._regime_bias(frames) if frames else (structure_events(o).iloc[-1].bias if not structure_events(o).empty else "neutral")
        now=o.index[-1]; price=float(o.close.iloc[-1]); setup_atr=atr(setup)
        a=float(setup_atr.iloc[-1]) if len(setup_atr) else np.nan
        if bias == "neutral" or not np.isfinite(a): return PredictorOutput(now,bias,no_trade_reason="neutral_or_unready_structure")
        zones=build_projected_zones(setup)
        candidates=[z for z in zones if z.is_active(now) and ((bias=="bullish" and z.side=="below") or (bias=="bearish" and z.side=="above"))]
        if not candidates: return PredictorOutput(now,bias,no_trade_reason="no_projected_zone")
        # All valid zones are tested; the best swept zone wins by score then proximity.
        swept_candidates = [z for z in candidates if
            (bias=="bullish" and o.low.iloc[-1] < z.low and price > z.high) or
            (bias=="bearish" and o.high.iloc[-1] > z.high and price < z.low)]
        if not swept_candidates:
            z=max(candidates,key=lambda q:(q.score,-abs(price-q.midpoint)))
            return PredictorOutput(now,bias,zone=z.zone_id,sweep_status="approaching",no_trade_reason="sweep_not_confirmed")
        z=max(swept_candidates,key=lambda q:(q.score,-abs(price-q.midpoint)))
        t=t.loc[t.time <= now]
        if t.empty: return PredictorOutput(now,bias,zone=z.zone_id,sweep_status="swept",no_trade_reason="orderflow_not_confirmed")
        f=orderflow_features(t, freq=self.flow_freq); flow=f.iloc[-1]
        dz = float(flow.delta_z) if np.isfinite(flow.delta_z) else 0.0
        confirm=bool(flow.delta_reversal or (dz > 1 if bias=="bullish" else dz < -1))
        if not confirm: return PredictorOutput(now,bias,zone=z.zone_id,sweep_status="swept",no_trade_reason="orderflow_not_confirmed")
        if bias=="bullish":
            stop=min(float(o.low.iloc[-1])-0.1*a, price-self.atr_mult*a); target=min([q.midpoint for q in zones if q.side=="above" and q.midpoint>price] or [price+2*a]); rr=(target-price)/(price-stop)
        else:
            stop=max(float(o.high.iloc[-1])+0.1*a, price+self.atr_mult*a); target=max([q.midpoint for q in zones if q.side=="below" and q.midpoint<price] or [price-2*a]); rr=(price-target)/(stop-price)
        if rr < self.min_rr: return PredictorOutput(now,bias,"reversal",z.zone_id,"confirmed",True,price,stop,target,rr,no_trade_reason="insufficient_reward_risk")
        size=equity*self.risk_fraction/abs(price-stop)
        return PredictorOutput(now,bias,"reversal",z.zone_id,"confirmed",True,price,stop,target,rr,None,size)
