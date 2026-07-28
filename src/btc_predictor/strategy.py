from __future__ import annotations

import numpy as np
import pandas as pd

from .footprint import footprint_confirmation
from .indicators import atr
from .models import PredictorOutput
from .structure import structure_events
from .zones import build_projected_zones


def detect_sweep(ohlc,zone,direction,setup_atr,min_depth=.05,max_depth=2.0,reclaim_bars=15):
    """Find a recent ATR-bounded breach followed by a causal reclaim within N closed 1m bars."""
    recent=ohlc.tail(reclaim_bars+1); breaches=[]
    for ts,bar in recent.iterrows():
        breached=bar.low<zone.low if direction=="bullish" else bar.high>zone.high
        if not breached: continue
        depth=(zone.low-float(bar.low))/setup_atr if direction=="bullish" else (float(bar.high)-zone.high)/setup_atr
        breaches.append((ts,depth,float(bar.low if direction=="bullish" else bar.high)))
    if not breaches:return {"status":"none","confirmed":False}
    breach_time,depth,extreme=max(breaches,key=lambda item:item[1])
    if depth>max_depth:return {"status":"excessive_excursion","confirmed":False,"time":breach_time,"depth_atr":depth,"extreme":extreme}
    if depth<min_depth:return {"status":"shallow_excursion","confirmed":False,"time":breach_time,"depth_atr":depth,"extreme":extreme}
    after=ohlc.loc[ohlc.index>=breach_time].head(reclaim_bars+1)
    reclaimed=(after.close>zone.high) if direction=="bullish" else (after.close<zone.low)
    if not reclaimed.any():return {"status":"waiting_reclaim","confirmed":False,"time":breach_time,"depth_atr":depth,"extreme":extreme}
    reclaim_time=reclaimed[reclaimed].index[0]
    if reclaim_time<ohlc.index[-reclaim_bars-1] or reclaim_time>ohlc.index[-1]:return {"status":"expired_reclaim","confirmed":False}
    return {"status":"confirmed","confirmed":True,"time":breach_time,"reclaim_time":reclaim_time,"depth_atr":depth,"extreme":extreme}


class Predictor:
    def __init__(self,risk_fraction=.0025,atr_mult=1.5,min_rr=1.5,sweep_atr=(.05,2.0),flow_freq="1min",reclaim_bars=60,require_15m_align=True,half_life_minutes=30.0):
        self.risk_fraction,self.atr_mult,self.min_rr,self.sweep_atr,self.flow_freq=risk_fraction,atr_mult,min_rr,sweep_atr,flow_freq
        self.reclaim_bars=reclaim_bars; self.require_15m_align=require_15m_align; self.half_life_minutes=half_life_minutes
        self._held_bias="neutral"; self.last_regimes={"4h":"neutral","1h":"neutral","15m":"neutral"}

    @staticmethod
    def _last(frame):
        events=structure_events(frame)
        return (events.iloc[-1].bias,events) if not events.empty else ("neutral",events)

    def _regime_bias(self,frames):
        signals=[]; event_sets=[]
        for name in ("4h","1h"):
            frame=frames.get(name)
            if frame is None or len(frame)<40:self.last_regimes[name]="unready"; return "neutral"
            signal,events=self._last(frame); signals.append(signal); event_sets.append(events); self.last_regimes[name]=signal
        setup=frames.get("15m")
        if setup is not None:self.last_regimes["15m"]=self._last(setup)[0]
        candidate=signals[0] if signals[0]==signals[1] else "neutral"
        if candidate=="neutral":return "neutral"
        if self.require_15m_align and setup is not None:
            setup_bias=self._last(setup)[0]
            self.last_regimes["15m"]=setup_bias
            if setup_bias not in ("neutral",candidate): return "neutral"
        if self._held_bias in ("bullish","bearish") and candidate!=self._held_bias:
            opposing=any(not e.empty and e.iloc[-1].event=="CHoCH" and e.iloc[-1].bias==candidate for e in event_sets)
            if not opposing:return "neutral"
        self._held_bias=candidate; return candidate

    def _probability_estimate(self, score: float, rr: float, bias: str, decay: float = 1.0) -> float:
        """Empirical probability estimate with half-life signal decay."""
        if not (np.isfinite(score) and np.isfinite(rr)):
            return None
        base = 0.50
        score_add = 0.20 * (score - 0.50)
        rr_add = 0.10 * (rr - 1.5)
        align = 0.05 if self.last_regimes.get("15m") == bias else 0.0
        prob = (base + score_add + rr_add + align) * decay
        return float(np.clip(prob, 0.05, 0.95))
    
    def _output(self,now,bias,**kwargs):
        return PredictorOutput(now,bias,regime_4h=self.last_regimes["4h"],regime_1h=self.last_regimes["1h"],setup_15m=self.last_regimes["15m"],**kwargs)

    def predict(self,ohlc,trades,equity=100_000,frames=None,flow_bars=None):
        if len(ohlc)<80 or trades.empty:return PredictorOutput(ohlc.index[-1],"neutral",no_trade_reason="insufficient_history")
        o=ohlc.copy(); o.index=pd.to_datetime(o.index,utc=True); t=trades.copy(); t["time"]=pd.to_datetime(t.time,utc=True)
        frames=frames or {}; setup=frames.get("15m",o)
        bias=self._regime_bias(frames) if frames else self._last(o)[0]
        now=o.index[-1]; price=float(o.close.iloc[-1]); setup_atr=atr(setup); a=float(setup_atr.iloc[-1]) if len(setup_atr) else np.nan
        if bias=="neutral" or not np.isfinite(a):return self._output(now,bias,no_trade_reason="timeframe_conflict" if self.last_regimes["4h"]!=self.last_regimes["1h"] else "neutral_or_unready_structure")
        zones=build_projected_zones(setup); directional=[z for z in zones if z.is_active(now) and ((bias=="bullish" and z.side=="below") or (bias=="bearish" and z.side=="above"))]
        if not directional:return self._output(now,bias,no_trade_reason="no_projected_zone")
        evaluated=[(z,detect_sweep(o,z,bias,a,*self.sweep_atr,self.reclaim_bars)) for z in directional]
        confirmed=[pair for pair in evaluated if pair[1].get("confirmed")]
        if not confirmed:
            waiting=[p for p in evaluated if p[1].get("status")=="waiting_reclaim"]
            z,sweep=max(waiting or evaluated,key=lambda p:(p[0].score,-abs(price-p[0].midpoint)))
            return self._output(now,bias,zone=z.zone_id,zone_kind=z.kind,sweep_status=sweep.get("status","approaching"),sweep_depth_atr=sweep.get("depth_atr"),sweep_time=str(sweep.get("time")) if sweep.get("time") is not None else None,no_trade_reason="sweep_not_confirmed")
        z,sweep=max(confirmed,key=lambda p:(p[0].score,-abs(price-p[0].midpoint)))
        usable=t.loc[t.time<now]
        confirm,flow=footprint_confirmation(usable,flow_bars,bias,sweep["time"],now)
        if not confirm:return self._output(now,bias,zone=z.zone_id,zone_kind=z.kind,sweep_status="confirmed",sweep_depth_atr=sweep["depth_atr"],sweep_time=str(sweep.get("time")) if sweep.get("time") is not None else None,reclaim_time=str(sweep.get("reclaim_time")) if sweep.get("reclaim_time") is not None else None,orderflow_reason=flow["reason"],exchange_agreement=flow.get("agreement",0.0)>0.5,no_trade_reason="orderflow_not_confirmed")
        
        # Exponential state decay calculation
        reclaim_time = pd.Timestamp(sweep.get("reclaim_time", now))
        delta_t_minutes = max(0.0, float((now - reclaim_time).total_seconds() / 60.0))
        tau = self.half_life_minutes / np.log(2)
        signal_decay = float(np.exp(-delta_t_minutes / tau))

        active_targets=[q for q in zones if q.is_active(now) and q.zone_id!=z.zone_id]
        if bias=="bullish":
            stop=min(sweep["extreme"]-.5*a,price-self.atr_mult*a); options=[q.midpoint for q in active_targets if q.side=="above" and q.midpoint>price]; target=min(options or [price+2*a]); rr=(target-price)/(price-stop)
        else:
            stop=max(sweep["extreme"]+.5*a,price+self.atr_mult*a); options=[q.midpoint for q in active_targets if q.side=="below" and q.midpoint<price]; target=max(options or [price-2*a]); rr=(price-target)/(stop-price)
        prob = self._probability_estimate(flow.get("score",0.0), rr, bias, decay=signal_decay)
        base=dict(setup_type="reversal",zone=z.zone_id,zone_kind=z.kind,sweep_status="confirmed",sweep_depth_atr=sweep["depth_atr"],sweep_time=str(sweep.get("time")) if sweep.get("time") is not None else None,reclaim_time=str(sweep.get("reclaim_time")) if sweep.get("reclaim_time") is not None else None,orderflow_confirmation=True,orderflow_reason=flow["reason"],exchange_agreement=flow.get("agreement"),entry=price,stop=stop,target=target,reward_risk=rr,probability_tp_before_sl=prob)
        if rr<self.min_rr:return self._output(now,bias,**base,no_trade_reason="insufficient_reward_risk")
        return self._output(now,bias,**base,position_size=equity*self.risk_fraction/abs(price-stop))
