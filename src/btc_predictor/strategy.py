from __future__ import annotations
import pandas as pd
import numpy as np
from .indicators import atr
from .models import PredictorOutput, Zone
from .structure import structure_events
from .zones import build_projected_zones
from .footprint import orderflow_features

class Predictor:
    def __init__(self, risk_fraction=.0025, atr_mult=1.5, min_rr=1.5, sweep_atr=(.05, 2.0)):
        self.risk_fraction, self.atr_mult, self.min_rr, self.sweep_atr = risk_fraction, atr_mult, min_rr, sweep_atr

    def predict(self, ohlc: pd.DataFrame, trades: pd.DataFrame, equity: float = 100_000) -> PredictorOutput:
        if len(ohlc) < 80 or trades.empty: return PredictorOutput(ohlc.index[-1], "neutral", no_trade_reason="insufficient_history")
        o=ohlc.copy(); o.index=pd.to_datetime(o.index,utc=True); t=trades.copy(); t["time"]=pd.to_datetime(t.time,utc=True)
        ev=structure_events(o); bias=ev.iloc[-1].bias if not ev.empty else "neutral"; now=o.index[-1]; price=float(o.close.iloc[-1]); a=float(atr(o).iloc[-1])
        if bias == "neutral" or not np.isfinite(a): return PredictorOutput(now,bias,no_trade_reason="neutral_or_unready_structure")
        zones=build_projected_zones(o); candidates=[z for z in zones if z.available_at <= now and ((bias=="bullish" and z.side=="below") or (bias=="bearish" and z.side=="above")) and z.swept_at is None]
        if not candidates: return PredictorOutput(now,bias,no_trade_reason="no_projected_zone")
        z=min(candidates,key=lambda q: abs(price-q.midpoint)); f=orderflow_features(t); flow=f.iloc[-1]
        swept=(bias=="bullish" and o.low.iloc[-1] < z.low and price > z.high) or (bias=="bearish" and o.high.iloc[-1] > z.high and price < z.low)
        confirm=bool(flow.delta_reversal if bias=="bullish" else (flow.delta_z < -1 if np.isfinite(flow.delta_z) else False))
        if not swept: return PredictorOutput(now,bias,zone=z.zone_id,sweep_status="approaching",no_trade_reason="sweep_not_confirmed")
        if not confirm: return PredictorOutput(now,bias,zone=z.zone_id,sweep_status="swept",no_trade_reason="orderflow_not_confirmed")
        if bias=="bullish":
            stop=min(float(o.low.iloc[-1])-0.1*a, price-self.atr_mult*a); target=min([q.midpoint for q in zones if q.side=="above" and q.midpoint>price] or [price+2*a]); rr=(target-price)/(price-stop)
        else:
            stop=max(float(o.high.iloc[-1])+0.1*a, price+self.atr_mult*a); target=max([q.midpoint for q in zones if q.side=="below" and q.midpoint<price] or [price-2*a]); rr=(price-target)/(stop-price)
        if rr < self.min_rr: return PredictorOutput(now,bias,"reversal",z.zone_id,"confirmed",True,price,stop,target,rr,no_trade_reason="insufficient_reward_risk")
        size=equity*self.risk_fraction/abs(price-stop)
        return PredictorOutput(now,bias,"reversal",z.zone_id,"confirmed",True,price,stop,target,rr,.5,size)
