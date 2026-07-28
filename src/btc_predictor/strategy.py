from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .footprint import footprint_confirmation
from .indicators import atr
from .models import PredictorOutput
from .structure import structure_events
from .timeframes import completed_timeframes
from .zones import build_projected_zones

# Replay evidence (63-trade ledger): these zone families lost systematically
# (untested_breakout -0.75R/trade, volume nodes -1.3R/trade), so their score is
# demoted rather than removed — they still trade when nothing better exists.
DEFAULT_ZONE_SCORE_ADJUSTMENTS = {"untested_breakout": -0.6, "volume_lvn": -0.6, "volume_hvn": -0.6}


def detect_sweep(ohlc,zone,direction,setup_atr,min_depth=.05,max_depth=2.0,reclaim_bars=15):
    """Find the latest causal breach/reclaim event after the zone became knowable."""
    if not np.isfinite(setup_atr) or setup_atr <= 0:
        return {"status":"invalid_atr","confirmed":False}
    recent=ohlc.tail(reclaim_bars+1)
    available=pd.Timestamp(zone.available_at)
    recent=recent.loc[recent.index>=available]
    if recent.empty:return {"status":"none","confirmed":False}

    lows=recent.low.to_numpy(dtype=float,copy=False)
    highs=recent.high.to_numpy(dtype=float,copy=False)
    closes=recent.close.to_numpy(dtype=float,copy=False)
    times=recent.index
    if direction=="bullish":
        breaches=lows<float(zone.low); reclaims=closes>float(zone.high); extremes=lows
    else:
        breaches=highs>float(zone.high); reclaims=closes<float(zone.low); extremes=highs
    events=[]; active=None
    for pos in range(len(recent)):
        ts=times[pos]; breached=bool(breaches[pos]); reclaimed=bool(reclaims[pos]); extreme=float(extremes[pos])
        if breached:
            if active is None:
                active={"time":ts,"extreme":extreme}
            elif direction=="bullish":
                active["extreme"]=min(active["extreme"],extreme)
            else:
                active["extreme"]=max(active["extreme"],extreme)
        if active is not None and reclaimed:
            depth=(zone.low-active["extreme"])/setup_atr if direction=="bullish" else (active["extreme"]-zone.high)/setup_atr
            events.append({"status":"confirmed","confirmed":True,"time":active["time"],"reclaim_time":ts,"depth_atr":depth,"extreme":active["extreme"]})
            active=None
    if active is not None:
        depth=(zone.low-active["extreme"])/setup_atr if direction=="bullish" else (active["extreme"]-zone.high)/setup_atr
        event={"status":"waiting_reclaim","confirmed":False,"time":active["time"],"depth_atr":depth,"extreme":active["extreme"]}
    elif events:
        event=events[-1]
    else:
        return {"status":"none","confirmed":False}
    depth=event["depth_atr"]
    if depth>max_depth:return {**event,"status":"excessive_excursion","confirmed":False}
    if depth<min_depth:return {**event,"status":"shallow_excursion","confirmed":False}
    return event


class Predictor:
    def __init__(self,risk_fraction=.0025,atr_mult=1.5,min_rr=1.0,sweep_atr=(.05,2.0),flow_freq="1min",reclaim_bars=60,require_15m_align=True,half_life_minutes=30.0,
                 stop_buffer_atr=0.3,max_stop_atr=2.5,min_target_distance_atr=0.25,measured_move_atr=2.0,
                 cost_bps=14.0,limit_cost_bps=9.0,max_cost_fraction=0.25,min_expectancy_r=0.05,min_stop_atr=0.5,entry_mode="market",limit_expiry_minutes=30,
                 limit_fallback=True,limit_buffer_atr=0.05,allow_1h_15m_regime=False,probability_calibration=None,
                 require_drift_alignment=True,drift_lookback_bars=96,zone_score_adjustments=None):
        self.risk_fraction,self.atr_mult,self.min_rr,self.sweep_atr,self.flow_freq=risk_fraction,atr_mult,min_rr,sweep_atr,flow_freq
        self.reclaim_bars=reclaim_bars; self.require_15m_align=require_15m_align; self.half_life_minutes=half_life_minutes
        # Risk framework: structural stop + buffer, hard width cap, noise-filtered targets.
        self.stop_buffer_atr=stop_buffer_atr; self.max_stop_atr=max_stop_atr
        # Stops tighter than this fraction of the risk ATR are churned by noise:
        # the measured replay drag was ~0.3R per trade in fees and slippage.
        self.min_stop_atr=min_stop_atr
        self.min_target_distance_atr=min_target_distance_atr; self.measured_move_atr=measured_move_atr
        # Execution economics: round-trip fees + slippage, charged against the gate.
        self.cost_bps=cost_bps; self.limit_cost_bps=limit_cost_bps
        self.max_cost_fraction=max_cost_fraction; self.min_expectancy_r=min_expectancy_r
        if entry_mode not in ("market","limit_retest"): raise ValueError("entry_mode must be market or limit_retest")
        self.entry_mode=entry_mode; self.limit_expiry_minutes=limit_expiry_minutes
        self.limit_fallback=limit_fallback; self.limit_buffer_atr=limit_buffer_atr
        self.allow_1h_15m_regime=allow_1h_15m_regime
        # Drift gate: fading the prevailing 24h drift lost 0.92R/trade in replay
        # (14% win rate); aligned trades were near breakeven. Zero or unknown
        # drift imposes no constraint.
        self.require_drift_alignment=require_drift_alignment; self.drift_lookback_bars=drift_lookback_bars
        self.zone_score_adjustments=dict(DEFAULT_ZONE_SCORE_ADJUSTMENTS if zone_score_adjustments is None else zone_score_adjustments)
        self._calibration=self._load_calibration(probability_calibration)
        self._held_bias="neutral"; self.last_regimes={"4h":"neutral","1h":"neutral","15m":"neutral"}

    def _drift_sign(self, setup):
        """Sign of the setup-frame drift over drift_lookback_bars; None when unknown."""
        if setup is None or len(setup)<=self.drift_lookback_bars: return None
        closes=setup.close.to_numpy(dtype=float,copy=False)
        delta=float(closes[-1]-closes[-1-self.drift_lookback_bars])
        if not np.isfinite(delta) or delta==0.0: return None
        return 1 if delta>0 else -1

    def _zone_rank(self, z, price):
        return (z.score + self.zone_score_adjustments.get(z.kind, 0.0), -abs(price-z.midpoint))

    @staticmethod
    def _load_calibration(source):
        """Optional empirical P(TP before SL) by footprint-score bucket, fit offline from replay results."""
        if source is None: return None
        data=source
        if isinstance(source,(str,Path)):
            path=Path(source)
            if not path.exists(): return None
            data=json.loads(path.read_text())
        if not isinstance(data,dict) or not data: return None
        try:
            return sorted((float(k),float(v)) for k,v in data.items())
        except (TypeError,ValueError):
            return None

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
        setup_bias=None
        if setup is not None:
            setup_bias=self._last(setup)[0]; self.last_regimes["15m"]=setup_bias
        candidate=signals[0] if signals[0]==signals[1] else "neutral"
        if candidate=="neutral":
            if self.allow_1h_15m_regime and signals[1] in ("bullish","bearish") and signals[1]==setup_bias:
                candidate=signals[1]
            else:
                return "neutral"
        if self.require_15m_align and setup_bias is not None:
            if setup_bias not in ("neutral",candidate): return "neutral"
        if self._held_bias in ("bullish","bearish") and candidate!=self._held_bias:
            opposing=any(not e.empty and e.iloc[-1].event=="CHoCH" and e.iloc[-1].bias==candidate for e in event_sets)
            if not opposing:return "neutral"
        self._held_bias=candidate; return candidate

    def _probability_estimate(self, score: float, rr: float, bias: str, decay: float = 1.0) -> float:
        """Empirical probability estimate with half-life signal decay."""
        if not (np.isfinite(score) and np.isfinite(rr)):
            return None
        if self._calibration:
            calibrated=self._calibrated_probability(score)
            if calibrated is not None:
                return float(np.clip(0.50+(calibrated-0.50)*decay,0.05,0.95))
        base = 0.50
        score_add = 0.20 * (score - 0.50)
        align = 0.05 if self.last_regimes.get("15m") == bias else 0.0
        # Payoff size must not inflate the hit probability: rr enters through the
        # expectancy gate, never through the estimate itself.
        raw = base + score_add + align
        # Age removes edge; it must converge to an uninformed 50%, not to 0%.
        prob = 0.50 + (raw - 0.50) * decay
        return float(np.clip(prob, 0.05, 0.95))

    @staticmethod
    def _risk_atr(ohlc, frames, setup_atr):
        """Volatility horizon for stops/targets; never compare an hourly setup with 1m ATR."""
        frame=(frames or {}).get("15m")
        if frame is None or len(frame)<14:
            frame=completed_timeframes(ohlc).get("15m")
        if frame is not None and len(frame)>=14:
            values=atr(frame)
            value=float(values.iloc[-1]) if len(values) else np.nan
            if np.isfinite(value) and value>0:return max(value,setup_atr)
        return setup_atr

    def _calibrated_probability(self, score: float) -> float:
        """Piecewise-constant lookup: first bucket upper bound >= score wins."""
        for upper, prob in self._calibration:
            if score <= upper:
                return prob
        return self._calibration[-1][1]

    def _structural_stop(self, bias: str, sweep: dict, a: float, entry: float):
        """Stop beyond the sweep extreme plus a volatility buffer; never widened to fit."""
        buffer = self.stop_buffer_atr * a
        if bias == "bullish":
            stop = float(sweep["extreme"]) - buffer
            risk = entry - stop
        else:
            stop = float(sweep["extreme"]) + buffer
            risk = stop - entry
        return stop, risk

    def _project_target(self, bias: str, entry: float, a: float, active_targets):
        """Nearest opposing pool beyond the noise band; measured-move fallback when none exists."""
        min_distance = self.min_target_distance_atr * a
        if bias == "bullish":
            options = sorted(q.midpoint for q in active_targets if q.side == "above" and q.midpoint > entry + min_distance)
            return options[0] if options else entry + self.measured_move_atr * a
        options = sorted((q.midpoint for q in active_targets if q.side == "below" and q.midpoint < entry - min_distance), reverse=True)
        return options[0] if options else entry - self.measured_move_atr * a
    
    def _output(self,now,bias,**kwargs):
        return PredictorOutput(now,bias,regime_4h=self.last_regimes["4h"],regime_1h=self.last_regimes["1h"],setup_15m=self.last_regimes["15m"],**kwargs)

    def predict(self,ohlc,trades,equity=100_000,frames=None,flow_bars=None,bias_override=None):
        if len(ohlc)<80 or trades.empty:return PredictorOutput(ohlc.index[-1],"neutral",no_trade_reason="insufficient_history")
        o=ohlc.copy(); o.index=pd.to_datetime(o.index,utc=True); t=trades.copy(); t["time"]=pd.to_datetime(t.time,utc=True)
        frames=frames or {}; setup=frames.get("15m",o)
        has_regime_frames=all(name in frames for name in ("4h","1h","15m"))
        bias=bias_override if bias_override in ("bullish","bearish","neutral") else (self._regime_bias(frames) if has_regime_frames else self._last(o)[0])
        now=o.index[-1]; price=float(o.close.iloc[-1]); setup_atr=atr(setup); a=float(setup_atr.iloc[-1]) if len(setup_atr) else np.nan
        if bias=="neutral" or not np.isfinite(a):return self._output(now,bias,no_trade_reason="timeframe_conflict" if self.last_regimes["4h"]!=self.last_regimes["1h"] else "neutral_or_unready_structure")
        if self.require_drift_alignment:
            drift=self._drift_sign(setup)
            if drift is not None and ((bias=="bullish" and drift<0) or (bias=="bearish" and drift>0)):
                return self._output(now,bias,no_trade_reason="counter_drift")
        risk_atr=self._risk_atr(o,frames,a)
        zones=build_projected_zones(setup)
        recent_cutoff=now-pd.Timedelta(minutes=self.reclaim_bars)
        def _eligible(z):
            if z.is_active(now):return True
            swept=pd.Timestamp(z.swept_at) if z.swept_at is not None else None
            return swept is not None and recent_cutoff<=swept<=now
        directional=sorted((z for z in zones if _eligible(z) and ((bias=="bullish" and z.side=="below") or (bias=="bearish" and z.side=="above"))),key=lambda z:self._zone_rank(z,price),reverse=True)
        if not directional:return self._output(now,bias,no_trade_reason="no_projected_zone")
        recent=o.tail(self.reclaim_bars+1)
        window_low=float(recent.low.min()); window_high=float(recent.high.max())
        breached=[z for z in directional if (window_low<z.low if bias=="bullish" else window_high>z.high)]
        if not breached:
            z=directional[0]
            return self._output(now,bias,zone=z.zone_id,zone_kind=z.kind,sweep_status="none",no_trade_reason="sweep_not_confirmed")
        evaluated=[(z,detect_sweep(o,z,bias,a,*self.sweep_atr,self.reclaim_bars)) for z in breached]
        confirmed=[pair for pair in evaluated if pair[1].get("confirmed")]
        if not confirmed:
            waiting=[p for p in evaluated if p[1].get("status")=="waiting_reclaim"]
            z,sweep=max(waiting or evaluated,key=lambda p:self._zone_rank(p[0],price))
            return self._output(now,bias,zone=z.zone_id,zone_kind=z.kind,sweep_status=sweep.get("status","approaching"),sweep_depth_atr=sweep.get("depth_atr"),sweep_time=str(sweep.get("time")) if sweep.get("time") is not None else None,no_trade_reason="sweep_not_confirmed")
        z,sweep=max(confirmed,key=lambda p:self._zone_rank(p[0],price))
        usable=t.loc[t.time<now]
        confirm,flow=footprint_confirmation(usable,flow_bars,bias,sweep["time"],now)
        if not confirm:return self._output(now,bias,zone=z.zone_id,zone_kind=z.kind,sweep_status="confirmed",sweep_depth_atr=sweep["depth_atr"],sweep_time=str(sweep.get("time")) if sweep.get("time") is not None else None,reclaim_time=str(sweep.get("reclaim_time")) if sweep.get("reclaim_time") is not None else None,orderflow_reason=flow["reason"],exchange_agreement=flow.get("agreement",0.0)>0.5,no_trade_reason="orderflow_not_confirmed")
        
        # Exponential state decay calculation
        reclaim_time = pd.Timestamp(sweep.get("reclaim_time", now))
        delta_t_minutes = max(0.0, float((now - reclaim_time).total_seconds() / 60.0))
        tau = self.half_life_minutes / np.log(2)
        signal_decay = float(np.exp(-delta_t_minutes / tau))

        active_targets=[q for q in zones if q.is_active(now) and q.zone_id!=z.zone_id]

        def _candidate(entry,entry_type,entry_expires_at):
            stop,risk=self._structural_stop(bias,sweep,a,entry)
            target=self._project_target(bias,entry,risk_atr,active_targets)
            reward=(target-entry) if bias=="bullish" else (entry-target)
            rr=reward/risk if risk>0 else 0.0
            prob=self._probability_estimate(flow.get("score",0.0),rr,bias,decay=signal_decay)
            # Round-trip cost in R units charged against expectancy.
            estimated_cost_bps=self.limit_cost_bps if entry_type=="limit" else self.cost_bps
            cost_r=(entry*estimated_cost_bps/10_000)/risk if risk>0 else None
            expectancy_r=(prob*rr-(1.0-prob)-cost_r) if prob is not None and cost_r is not None else None
            base=dict(setup_type="reversal",zone=z.zone_id,zone_kind=z.kind,sweep_status="confirmed",sweep_depth_atr=sweep["depth_atr"],sweep_time=str(sweep.get("time")) if sweep.get("time") is not None else None,reclaim_time=str(sweep.get("reclaim_time")) if sweep.get("reclaim_time") is not None else None,orderflow_confirmation=True,orderflow_reason=flow["reason"],exchange_agreement=flow.get("agreement"),entry=entry,stop=stop,target=target,reward_risk=rr,probability_tp_before_sl=prob,entry_type=entry_type,entry_expires_at=entry_expires_at,expectancy_r=expectancy_r,risk_atr=risk_atr,stop_distance_atr=risk/risk_atr if risk_atr>0 else None,estimated_cost_bps=estimated_cost_bps,estimated_cost_r=cost_r)
            rejection=None
            if risk<=0 or not np.isfinite(risk):rejection="invalid_stop"
            elif risk<self.min_stop_atr*risk_atr:rejection="stop_too_tight"
            elif risk>self.max_stop_atr*risk_atr:rejection="stop_too_wide"
            elif cost_r is not None and cost_r>self.max_cost_fraction:rejection="costs_exceed_edge"
            elif reward<=0 or rr<self.min_rr:rejection="insufficient_reward_risk"
            elif expectancy_r is None or expectancy_r<self.min_expectancy_r:rejection="negative_expectancy"
            return base,risk,rejection

        # Entry: market at the decision close, or a resting limit at the reclaimed
        # zone edge waiting for a retest (expires after limit_expiry_minutes).
        limit_buffer=self.limit_buffer_atr*a
        limit_entry=(float(z.high)+limit_buffer) if bias=="bullish" else (float(z.low)-limit_buffer)
        limit_expires=(now+pd.Timedelta(minutes=self.limit_expiry_minutes)).isoformat()
        candidates=[(price,"market",None)] if self.entry_mode=="market" else [(limit_entry,"limit",limit_expires)]
        # Late market entries fail the location gates; rather than chase, work a
        # limit at the reclaimed edge where the same stop/target framework is viable.
        if self.limit_fallback and self.entry_mode=="market":
            candidates.append((limit_entry,"limit",limit_expires))

        first_base=first_rejection=None; candidate_rejections={}
        for entry,entry_type,expires_at in candidates:
            base,risk,rejection=_candidate(entry,entry_type,expires_at)
            if first_base is None:first_base,first_rejection=base,rejection
            if rejection is None:
                return self._output(now,bias,**base,position_size=equity*self.risk_fraction/risk,candidate_rejections=candidate_rejections or None)
            candidate_rejections[entry_type]=rejection
        return self._output(now,bias,**first_base,no_trade_reason=first_rejection,candidate_rejections=candidate_rejections)
