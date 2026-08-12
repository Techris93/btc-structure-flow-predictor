from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

from .footprint import footprint_confirmation
from .indicators import atr
from .models import PredictorOutput
from .structure import structure_events
from .zones import build_projected_zones
from . import live_policy


def detect_sweep(ohlc,zone,direction,setup_atr,min_depth=.05,max_depth=2.0,reclaim_bars=15,rearm_bars=3,rearm_atr=.5):
    """Find a causal sweep episode from closed 1m bars.

    The episode identity is its *first* breach, not whichever later candle
    happens to make the deepest extreme. A deeper low/high updates geometry
    without manufacturing a new setup. A new episode is possible only after
    a reclaim has remained at least ``rearm_atr`` away for ``rearm_bars``
    completed bars.
    """
    eligible=ohlc.copy()
    eligible.index=pd.to_datetime(eligible.index,utc=True)
    available_at=pd.Timestamp(zone.available_at)
    available_at=available_at.tz_localize("UTC") if available_at.tzinfo is None else available_at.tz_convert("UTC")
    # A zone cannot be swept before it was causally available.
    recent=eligible.loc[eligible.index>=available_at].tail(reclaim_bars+1)
    if recent.empty:return {"status":"none","confirmed":False}
    breach_mask=(recent.low.to_numpy()<zone.low) if direction=="bullish" else (recent.high.to_numpy()>zone.high)
    if not np.any(breach_mask):return {"status":"none","confirmed":False}
    episodes=[]; episode=None; clear_bars=0
    for ts,bar in recent.iterrows():
        breached=bar.low<zone.low if direction=="bullish" else bar.high>zone.high
        if breached:
            depth=(zone.low-float(bar.low))/setup_atr if direction=="bullish" else (float(bar.high)-zone.high)/setup_atr
            extreme=float(bar.low if direction=="bullish" else bar.high)
            if episode is None:
                episode={"time":ts,"depth_atr":depth,"extreme":extreme,"reclaim_time":None}
            elif depth>episode["depth_atr"]:
                episode["depth_atr"]=depth; episode["extreme"]=extreme
            # A candle may breach intrabar and reclaim on its close. Preserve
            # that causal close as the first reclaim instead of waiting for a
            # later bar that never breached.
            reclaimed=float(bar.close)>zone.high if direction=="bullish" else float(bar.close)<zone.low
            if reclaimed and episode["reclaim_time"] is None:
                episode["reclaim_time"]=ts
            clear_bars=0
            continue
        if episode is None:
            continue
        reclaimed=float(bar.close)>zone.high if direction=="bullish" else float(bar.close)<zone.low
        if reclaimed and episode["reclaim_time"] is None:
            episode["reclaim_time"]=ts
        away=(float(bar.close)>=zone.high+rearm_atr*setup_atr) if direction=="bullish" else (float(bar.close)<=zone.low-rearm_atr*setup_atr)
        clear_bars=clear_bars+1 if episode["reclaim_time"] is not None and away else 0
        if clear_bars>=max(1,int(rearm_bars)):
            episodes.append(episode); episode=None; clear_bars=0
    if episode is not None: episodes.append(episode)
    if not episodes:return {"status":"none","confirmed":False}
    selected=episodes[-1]
    breach_time=selected["time"]; depth=selected["depth_atr"]; extreme=selected["extreme"]
    if depth>max_depth:return {"status":"excessive_excursion","confirmed":False,"time":breach_time,"depth_atr":depth,"extreme":extreme}
    if depth<min_depth:return {"status":"shallow_excursion","confirmed":False,"time":breach_time,"depth_atr":depth,"extreme":extreme}
    reclaim_time=selected.get("reclaim_time")
    if reclaim_time is None:return {"status":"waiting_reclaim","confirmed":False,"time":breach_time,"depth_atr":depth,"extreme":extreme}
    if reclaim_time<recent.index[0] or reclaim_time>recent.index[-1]:return {"status":"expired_reclaim","confirmed":False,"time":breach_time,"reclaim_time":reclaim_time,"depth_atr":depth,"extreme":extreme}
    return {"status":"confirmed","confirmed":True,"time":breach_time,"reclaim_time":reclaim_time,"depth_atr":depth,"extreme":extreme}


def zone_reclaim_eligible(zone,at,reclaim_bars=60,bar_freq="1min"):
    """Keep an invalidated zone eligible for its declared reclaim window."""
    at=pd.Timestamp(at)
    available=pd.Timestamp(zone.available_at)
    if at<available or (zone.expires_at is not None and at>=pd.Timestamp(zone.expires_at)):
        return False
    if zone.is_active(at):
        return True
    invalidation_times=[pd.Timestamp(value) for value in (zone.swept_at,zone.invalidated_at) if value is not None]
    if not invalidation_times:
        return False
    invalidated_at=min(invalidation_times)
    return invalidated_at<=at<=invalidated_at+max(0,int(reclaim_bars))*pd.Timedelta(bar_freq)


def pending_flow_reason(sweep):
    """Describe why an episode has not reached an actionable frozen gate."""
    status=str((sweep or {}).get("status") or "none")
    return {
        "none":"awaiting_sweep_breach",
        "waiting_reclaim":"provisional_awaiting_reclaim",
        "shallow_excursion":"sweep_depth_below_minimum",
        "excessive_excursion":"sweep_depth_above_maximum",
        "expired_reclaim":"sweep_reclaim_expired",
    }.get(status,"awaiting_sweep_breach")


class Predictor:
    def __init__(self,risk_fraction=.0025,atr_mult=1.5,min_rr=1.5,sweep_atr=(.05,2.0),flow_freq="1min",reclaim_bars=60,require_15m_align=True,half_life_minutes=30.0,retrace_entry_atr=None,retrace_pct=0.5,sweep_rearm_bars=3,sweep_rearm_atr=.5,flow_gate_mode="independent",legacy_orderflow_threshold=.40,market_flow_threshold=.40,raw_footprint_threshold=.40,footprint_price_bucket=25.0,footprint_full_credit_ratio=1.5,venue_freshness_seconds=150,cache_closed_frames=False,use_fixed_pct_exits=None,stop_pct=None,target_pct=None):
        self.risk_fraction,self.atr_mult,self.min_rr,self.sweep_atr,self.flow_freq=risk_fraction,atr_mult,min_rr,sweep_atr,flow_freq
        # Default off so isolated strategy tests keep ATR/structural geometry.
        # Live app passes True (0.5% SL / 1% TP).
        self.use_fixed_pct_exits=bool(use_fixed_pct_exits) if use_fixed_pct_exits is not None else False
        self.stop_pct=live_policy.FIXED_STOP_PCT if stop_pct is None else float(stop_pct)
        self.target_pct=live_policy.FIXED_TARGET_PCT if target_pct is None else float(target_pct)
        self.reclaim_bars=reclaim_bars; self.require_15m_align=require_15m_align; self.half_life_minutes=half_life_minutes
        self.sweep_rearm_bars=max(1,int(sweep_rearm_bars)); self.sweep_rearm_atr=max(0.0,float(sweep_rearm_atr))
        # Late-entry guard: when a confirmed sweep is deeper than
        # `retrace_entry_atr` ATR, enter on a `retrace_pct` pullback of the
        # sweep leg (pending limit) instead of at market. None = disabled
        # (market entry, original behavior).
        self.retrace_entry_atr=retrace_entry_atr; self.retrace_pct=float(retrace_pct)
        requested_gate=str(flow_gate_mode or "independent").lower()
        self.flow_gate_mode=requested_gate if requested_gate in ("independent","calibrated") else "shadow"
        self.legacy_orderflow_threshold=float(legacy_orderflow_threshold)
        self.market_flow_threshold=float(market_flow_threshold); self.raw_footprint_threshold=float(raw_footprint_threshold)
        self.footprint_price_bucket=float(footprint_price_bucket); self.footprint_full_credit_ratio=float(footprint_full_credit_ratio)
        self.venue_freshness_seconds=max(1.0,float(venue_freshness_seconds))
        self._held_bias="neutral"; self.last_regimes={"4h":"neutral","1h":"neutral","15m":"neutral"}
        self.last_session_cvd=None
        self.cache_closed_frames=bool(cache_closed_frames)
        self._structure_cache={}; self._atr_cache={}; self._zone_cache={}; self._frozen_flow_cache={}

    @staticmethod
    def _frame_signature(frame):
        if frame is None or frame.empty:return None
        last=frame.iloc[-1]
        return (len(frame),pd.Timestamp(frame.index[0]).value,pd.Timestamp(frame.index[-1]).value,tuple(float(last.get(name,np.nan)) for name in ("open","high","low","close","volume")))

    @staticmethod
    def _remember(cache,signature,value,max_entries=8):
        cache[signature]=value
        while len(cache)>max_entries:cache.pop(next(iter(cache)))
        return value

    def _last(self,frame):
        signature=self._frame_signature(frame) if self.cache_closed_frames else None
        if self.cache_closed_frames and signature in self._structure_cache:return self._structure_cache[signature]
        events=structure_events(frame)
        value=(events.iloc[-1].bias,events) if not events.empty else ("neutral",events)
        return self._remember(self._structure_cache,signature,value) if self.cache_closed_frames else value

    def _setup_atr(self,frame):
        if not self.cache_closed_frames:return atr(frame)
        signature=self._frame_signature(frame)
        if signature not in self._atr_cache:self._remember(self._atr_cache,signature,atr(frame))
        return self._atr_cache[signature]

    def _projected_zones(self,frame):
        if not self.cache_closed_frames:return build_projected_zones(frame)
        signature=self._frame_signature(frame)
        if signature not in self._zone_cache:self._remember(self._zone_cache,signature,build_projected_zones(frame))
        return self._zone_cache[signature]

    def _regime_bias(self,frames):
        signals=[]; event_sets=[]
        for name in ("4h","1h"):
            frame=frames.get(name)
            if frame is None or len(frame)<40:self.last_regimes[name]="unready"; return "neutral"
            signal,events=self._last(frame); signals.append(signal); event_sets.append(events); self.last_regimes[name]=signal
        setup=frames.get("15m"); setup_bias=self._last(setup)[0] if setup is not None else None
        if setup is not None:self.last_regimes["15m"]=setup_bias
        candidate=signals[0] if signals[0]==signals[1] else "neutral"
        if candidate=="neutral":return "neutral"
        if self.require_15m_align and setup is not None:
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
        kwargs.setdefault("flow_gate_mode",self.flow_gate_mode)
        kwargs.setdefault("market_flow_threshold",self.market_flow_threshold)
        kwargs.setdefault("raw_footprint_threshold",self.raw_footprint_threshold)
        kwargs.setdefault("session_cvd",self.last_session_cvd)
        return PredictorOutput(now,bias,regime_4h=self.last_regimes["4h"],regime_1h=self.last_regimes["1h"],setup_15m=self.last_regimes["15m"],**kwargs)

    def predict(self,ohlc,trades,equity=100_000,frames=None,flow_bars=None,flow_source=None,flow_aggregates=None,session_cvd=None):
        self.last_session_cvd=session_cvd
        if len(ohlc)<80 or trades.empty:return PredictorOutput(ohlc.index[-1],"neutral",no_trade_reason="insufficient_history",flow_gate_mode=self.flow_gate_mode,market_flow_threshold=self.market_flow_threshold,raw_footprint_threshold=self.raw_footprint_threshold,session_cvd=session_cvd)
        o=ohlc
        if not isinstance(o.index,pd.DatetimeIndex) or o.index.tz is None:
            o=ohlc.copy();o.index=pd.to_datetime(o.index,utc=True)
        t=trades
        if not isinstance(t["time"].dtype,pd.DatetimeTZDtype):
            t=trades.copy();t["time"]=pd.to_datetime(t.time,utc=True)
        frames=frames or {}; setup=frames.get("15m",o)
        bias=self._regime_bias(frames) if frames else self._last(o)[0]
        now=o.index[-1]; price=float(o.close.iloc[-1]); setup_atr=self._setup_atr(setup); a=float(setup_atr.iloc[-1]) if len(setup_atr) else np.nan
        if bias=="neutral" or not np.isfinite(a):return self._output(now,bias,no_trade_reason="timeframe_conflict" if self.last_regimes["4h"]!=self.last_regimes["1h"] else "neutral_or_unready_structure")
        zones=self._projected_zones(setup); directional=[z for z in zones if zone_reclaim_eligible(z,now,self.reclaim_bars,self.flow_freq) and ((bias=="bullish" and z.side=="below") or (bias=="bearish" and z.side=="above"))]
        if not directional:return self._output(now,bias,no_trade_reason="no_projected_zone")
        evaluated=[(z,detect_sweep(o,z,bias,a,*self.sweep_atr,self.reclaim_bars,self.sweep_rearm_bars,self.sweep_rearm_atr)) for z in directional]
        flow_evaluated=[]; observations=[]
        for z,sweep in evaluated:
            sweep_time=sweep.get("time")
            if sweep_time is None:
                flow_evaluated.append((z,sweep,None,None)); continue
            flow_end=pd.Timestamp(sweep.get("reclaim_time")) if sweep.get("confirmed") and sweep.get("reclaim_time") is not None else now
            usable=t.loc[t.time<flow_end]
            frozen_key=(bias,pd.Timestamp(sweep_time).isoformat(),flow_end.isoformat(),self.flow_gate_mode,self.legacy_orderflow_threshold,self.market_flow_threshold,self.raw_footprint_threshold,self.footprint_price_bucket,self.footprint_full_credit_ratio) if sweep.get("confirmed") else None
            cached_flow=self._frozen_flow_cache.get(frozen_key) if frozen_key is not None else None
            if cached_flow is not None:
                flow_confirm,flow=cached_flow
            else:
                flow_confirm,flow=footprint_confirmation(
                    usable,flow_bars,bias,sweep_time,flow_end,
                    min_score=self.legacy_orderflow_threshold,
                    footprint_bars=flow_aggregates,
                    price_bucket=self.footprint_price_bucket,
                    full_credit_ratio=self.footprint_full_credit_ratio,
                    market_threshold=self.market_flow_threshold,
                    raw_threshold=self.raw_footprint_threshold,
                    gate_mode=self.flow_gate_mode,
                    venue_freshness_seconds=self.venue_freshness_seconds,
                )
                if frozen_key is not None:self._remember(self._frozen_flow_cache,frozen_key,(flow_confirm,flow),max_entries=256)
            state="frozen" if sweep.get("confirmed") else "provisional"
            flow_evaluated.append((z,sweep,flow_confirm,flow))
            observations.append({
                "zone":z.zone_id,"sweep_time":pd.Timestamp(sweep_time).isoformat(),
                "reclaim_time":pd.Timestamp(sweep.get("reclaim_time")).isoformat() if sweep.get("reclaim_time") is not None else None,
                "sweep_status":sweep.get("status"),"flow_state":state,
                "market_flow_score":flow.get("market_flow_score"),"raw_footprint_score":flow.get("raw_footprint_score"),
                "orderflow_score":flow.get("score"),"contributing_exchanges":flow.get("contributing_exchanges") or [],
                "fresh_exchanges":flow.get("fresh_exchanges") or [],
            })
        confirmed=[pair for pair in flow_evaluated if pair[1].get("confirmed")]
        if not confirmed:
            waiting=[p for p in flow_evaluated if p[1].get("status")=="waiting_reclaim"]
            selected=max(waiting or flow_evaluated,key=lambda p:(p[0].score,-abs(price-p[0].midpoint)))
            z,sweep,_,flow=selected
            diagnostics={"flow_state":"waiting_for_breach"}
            if flow is not None:
                diagnostics=dict(orderflow_score=flow.get("score"),orderflow_threshold=flow.get("threshold"),orderflow_bars=flow.get("bars"),orderflow_exchanges=tuple(flow.get("contributing_exchanges") or ()),orderflow_fresh_exchanges=tuple(flow.get("fresh_exchanges") or ()),exchange_agreement=flow.get("agreement",0.0),market_flow_score=flow.get("market_flow_score"),market_flow_threshold=flow.get("market_flow_threshold"),market_flow_confirmed=flow.get("market_flow_confirmed",False),raw_footprint_score=flow.get("raw_footprint_score"),raw_footprint_ratio=flow.get("raw_footprint_ratio"),raw_footprint_threshold=flow.get("raw_footprint_threshold"),raw_footprint_confirmed=flow.get("raw_footprint_confirmed",False),raw_footprint_eligible=flow.get("raw_footprint_eligible",False),flow_state="provisional")
            return self._output(now,bias,zone=z.zone_id,zone_kind=z.kind,sweep_status=sweep.get("status","approaching"),sweep_depth_atr=sweep.get("depth_atr"),sweep_time=str(sweep.get("time")) if sweep.get("time") is not None else None,sweep_evaluation_status="evaluated",orderflow_evaluation_status="provisional" if flow is not None else "not_evaluated",orderflow_reason=pending_flow_reason(sweep),orderflow_source=flow_source,sweep_observations=tuple(observations),**diagnostics,no_trade_reason="sweep_not_confirmed")
        z,sweep,confirm,flow=max(confirmed,key=lambda p:(p[0].score,-abs(price-p[0].midpoint)))
        flow_diagnostics=dict(sweep_evaluation_status="evaluated",orderflow_evaluation_status="evaluated",orderflow_reason=flow["reason"],orderflow_score=flow.get("score"),orderflow_threshold=flow.get("threshold"),orderflow_bars=flow.get("bars"),orderflow_source=flow_source,orderflow_exchanges=tuple(flow.get("contributing_exchanges") or ()),orderflow_fresh_exchanges=tuple(flow.get("fresh_exchanges") or ()),exchange_agreement=flow.get("agreement",0.0),market_flow_score=flow.get("market_flow_score"),market_flow_threshold=flow.get("market_flow_threshold"),market_flow_confirmed=flow.get("market_flow_confirmed",False),raw_footprint_score=flow.get("raw_footprint_score"),raw_footprint_ratio=flow.get("raw_footprint_ratio"),raw_footprint_threshold=flow.get("raw_footprint_threshold"),raw_footprint_confirmed=flow.get("raw_footprint_confirmed",False),raw_footprint_eligible=flow.get("raw_footprint_eligible",False),flow_state="frozen",sweep_observations=tuple(observations))
        if not confirm:return self._output(now,bias,zone=z.zone_id,zone_kind=z.kind,sweep_status="confirmed",sweep_depth_atr=sweep["depth_atr"],sweep_time=str(sweep.get("time")) if sweep.get("time") is not None else None,reclaim_time=str(sweep.get("reclaim_time")) if sweep.get("reclaim_time") is not None else None,**flow_diagnostics,no_trade_reason="orderflow_not_confirmed")
        
        # Exponential state decay calculation
        reclaim_time = pd.Timestamp(sweep.get("reclaim_time", now))
        delta_t_minutes = max(0.0, float((now - reclaim_time).total_seconds() / 60.0))
        tau = self.half_life_minutes / np.log(2)
        signal_decay = float(np.exp(-delta_t_minutes / tau))

        active_targets=[q for q in zones if q.is_active(now) and q.zone_id!=z.zone_id]
        entry=price; entry_type="market"
        if self.retrace_entry_atr is not None and sweep.get("depth_atr") is not None and sweep["depth_atr"]>=self.retrace_entry_atr:
            leg=abs(sweep["extreme"]-price)
            if leg>0:
                candidate_entry=(price-self.retrace_pct*leg) if bias=="bullish" else (price+self.retrace_pct*leg)
                if np.isfinite(candidate_entry) and candidate_entry>0:
                    entry=candidate_entry; entry_type="limit"
        if self.use_fixed_pct_exits:
            geo=live_policy.fixed_pct_exits(entry,bias,stop_pct=self.stop_pct,target_pct=self.target_pct)
            stop,target,risk,rr=geo["stop"],geo["target"],geo["risk"],geo["reward_risk"]
        elif bias=="bullish":
            stop=min(sweep["extreme"]-.5*a,price-self.atr_mult*a); risk=price-stop
            options=sorted(q.midpoint for q in active_targets if q.side=="above" and q.midpoint>price)
            target=next((t for t in options if risk>0 and (t-price)/risk>=self.min_rr), None)
            if target is None:
                target=max(price+self.min_rr*risk,price+2*a) if risk>0 else price+2*a
            rr=(target-price)/risk if risk>0 else 0.0
        else:
            stop=max(sweep["extreme"]+.5*a,price+self.atr_mult*a); risk=stop-price
            options=sorted((q.midpoint for q in active_targets if q.side=="below" and q.midpoint<price),reverse=True)
            target=next((t for t in options if risk>0 and (price-t)/risk>=self.min_rr), None)
            if target is None:
                target=min(price-self.min_rr*risk,price-2*a) if risk>0 else price-2*a
            rr=(price-target)/risk if risk>0 else 0.0
        if entry_type=="limit" and not self.use_fixed_pct_exits:
            candidate_risk=(entry-stop) if bias=="bullish" else (stop-entry)
            if candidate_risk>0 and np.isfinite(candidate_risk):
                risk=candidate_risk
                options=sorted(q.midpoint for q in active_targets if q.side==("above" if bias=="bullish" else "below"))
                if bias=="bullish":
                    target=next((t for t in options if t>entry and (t-entry)/risk>=self.min_rr),max(entry+self.min_rr*risk,entry+2*a))
                    rr=(target-entry)/risk
                else:
                    target=next((t for t in options if t<entry and (entry-t)/risk>=self.min_rr),min(entry-self.min_rr*risk,entry-2*a))
                    rr=(entry-target)/risk
            else:
                log.info("retrace_entry_skipped bias=%s depth_atr=%.2f market=%.2f candidate=%.2f invalid_risk=%.2f",bias,sweep["depth_atr"],price,entry,candidate_risk)
                entry=price; entry_type="market"

        skipped=[]
        for candidate in active_targets:
            midpoint=float(candidate.midpoint)
            opposing=(bias=="bullish" and candidate.side=="above" and entry<midpoint<target) or (bias=="bearish" and candidate.side=="below" and target<midpoint<entry)
            if opposing and risk>0:
                candidate_rr=((midpoint-entry)/risk) if bias=="bullish" else ((entry-midpoint)/risk)
                if candidate_rr<self.min_rr:
                    distance_atr=abs(midpoint-entry)/a if a>0 else None
                    skipped.append({"kind":candidate.kind,"mid":round(midpoint,2),"dist_atr":round(distance_atr,2) if distance_atr is not None else None})
        log.info("target_select bias=%s entry=%.2f stop=%.2f target=%.2f rr=%.2f skipped_zones=%s",bias,entry,stop,target,rr,skipped)
        # Heuristic only — display/log. Never size or hard-gate from this p.
        prob = self._probability_estimate(flow.get("score",0.0), rr, bias, decay=signal_decay)
        base=dict(setup_type="reversal",zone=z.zone_id,zone_kind=z.kind,sweep_status="confirmed",sweep_depth_atr=sweep["depth_atr"],sweep_time=str(sweep.get("time")) if sweep.get("time") is not None else None,reclaim_time=str(sweep.get("reclaim_time")) if sweep.get("reclaim_time") is not None else None,orderflow_confirmation=True,orderflow_reason=flow["reason"],orderflow_evaluation_status="evaluated",sweep_evaluation_status="evaluated",orderflow_score=flow.get("score"),orderflow_threshold=flow.get("threshold"),orderflow_bars=flow.get("bars"),orderflow_source=flow_source,orderflow_exchanges=tuple(flow.get("contributing_exchanges") or ()),orderflow_fresh_exchanges=tuple(flow.get("fresh_exchanges") or ()),exchange_agreement=flow.get("agreement"),market_flow_score=flow.get("market_flow_score"),market_flow_threshold=flow.get("market_flow_threshold"),market_flow_confirmed=flow.get("market_flow_confirmed",False),raw_footprint_score=flow.get("raw_footprint_score"),raw_footprint_ratio=flow.get("raw_footprint_ratio"),raw_footprint_threshold=flow.get("raw_footprint_threshold"),raw_footprint_confirmed=flow.get("raw_footprint_confirmed",False),raw_footprint_eligible=flow.get("raw_footprint_eligible",False),flow_state="frozen",sweep_observations=tuple(observations),entry=entry,entry_type=entry_type,stop=stop,target=target,reward_risk=rr,probability_tp_before_sl=prob,setup_atr=a)
        if rr<self.min_rr or not np.isfinite(risk) or risk<=0:
            log.info("insufficient_reward_risk bias=%s entry=%.2f stop=%.2f target=%.2f rr=%.2f min_rr=%.2f skipped_zones=%s",bias,entry,stop,target,rr,self.min_rr,skipped)
            return self._output(now,bias,**base,no_trade_reason="insufficient_reward_risk")
        # Size from fixed risk fraction only (never from heuristic probability).
        return self._output(now,bias,**base,position_size=equity*self.risk_fraction/risk)
