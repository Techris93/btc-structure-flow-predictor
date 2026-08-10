import json

import pandas as pd
import pytest

from btc_predictor.flow_calibration import CalibrationConfig, _event_outcome, walk_forward_periods
from btc_predictor.flow_gate import load_flow_gate
from btc_predictor.flow_state import FlowStateStore
from btc_predictor.footprint import footprint_confirmation
from btc_predictor.models import Zone
from btc_predictor.strategy import Predictor


def _flow_bars(count=30):
    idx=pd.date_range("2026-01-01 00:01",periods=count,freq="min",tz="UTC")
    return pd.DataFrame({"open":100.0,"high":101.0,"low":99.0,"close":[100+i*.1 for i in range(count)],"volume":10.0,"trades":10,"taker_buy_volume":9.0},index=idx)


def test_single_venue_can_never_confirm_raw_footprint():
    bars=_flow_bars(); end=bars.index[-1]
    trades=pd.DataFrame([
        {"time":end-pd.Timedelta(seconds=30),"price":103,"qty":10,"side":"buy","exchange":"binance"}
    ])
    confirmed,details=footprint_confirmation(trades,bars,"bullish",end,end,gate_mode="calibrated",market_threshold=0.0,raw_threshold=0.0)
    assert details["market_flow_confirmed"] is True
    assert details["raw_footprint_eligible"] is False
    assert details["raw_footprint_confirmed"] is False
    assert confirmed is False


def test_shadow_mode_preserves_legacy_composite_gate():
    bars=_flow_bars(); end=bars.index[-1]
    trades=pd.DataFrame([
        {"time":end-pd.Timedelta(seconds=40),"price":103,"qty":2,"side":"buy","exchange":"binance"},
        {"time":end-pd.Timedelta(seconds=20),"price":103,"qty":2,"side":"buy","exchange":"bybit"},
    ])
    confirmed,details=footprint_confirmation(trades,bars,"bullish",end,end,min_score=.40,gate_mode="shadow")
    assert confirmed is (details["score"]>=.40)
    assert details["flow_gate_mode"]=="shadow"


def test_independent_gate_requires_both_scores_and_two_venues():
    bars=_flow_bars(); end=bars.index[-1]
    trades=pd.DataFrame([
        {"time":end-pd.Timedelta(seconds=40),"price":103,"qty":2,"side":"buy","exchange":"binance"},
        {"time":end-pd.Timedelta(seconds=20),"price":103,"qty":2,"side":"buy","exchange":"bybit"},
    ])
    confirmed,details=footprint_confirmation(
        trades,bars,"bullish",end,end,gate_mode="independent",
        market_threshold=0.0,raw_threshold=1.1,
    )
    assert details["market_flow_confirmed"] is True
    assert details["raw_footprint_confirmed"] is False
    assert confirmed is False


def test_independent_gate_rejects_a_present_but_stale_venue():
    bars=_flow_bars(); end=bars.index[-1]; sweep=end-pd.Timedelta(minutes=5)
    trades=pd.DataFrame([
        {"time":end-pd.Timedelta(seconds=30),"price":103,"qty":2,"side":"buy","exchange":"binance"},
        {"time":end-pd.Timedelta(minutes=3),"price":103,"qty":2,"side":"buy","exchange":"bybit"},
    ])
    confirmed,details=footprint_confirmation(
        trades,bars,"bullish",sweep,end,gate_mode="independent",
        market_threshold=0.0,raw_threshold=0.0,venue_freshness_seconds=150,
    )
    assert details["agreement_status"]=="cross_exchange"
    assert details["fresh_exchanges"]==["binance"]
    assert details["raw_footprint_eligible"] is False
    assert confirmed is False


def test_session_cvd_persists_resets_and_ignores_late_trade(tmp_path):
    path=tmp_path/"flow.json"; store=FlowStateStore(path)
    trades=[]
    for minute in range(3):
        for exchange in ("binance","bybit"):
            trades.append({"time":pd.Timestamp("2026-01-01 00:00:10Z")+pd.Timedelta(minutes=minute),"price":100,"qty":1,"side":"buy","exchange":exchange})
    frame=pd.DataFrame(trades); decision=pd.Timestamp("2026-01-01 00:03:00Z")
    store.update(frame,decision); before=store.session_cvd(decision)
    assert before["complete"] is True and before["combined"]==600.0
    reopened=FlowStateStore(path); assert reopened.session_cvd(decision)==before
    late=pd.concat([frame,pd.DataFrame([{"time":pd.Timestamp("2026-01-01 00:00:20Z"),"price":100,"qty":100,"side":"sell","exchange":"binance"}])],ignore_index=True)
    reopened.update(late,decision); assert reopened.session_cvd(decision)==before
    london=pd.DataFrame([
        {"time":pd.Timestamp("2026-01-01 08:00:10Z"),"price":100,"qty":1,"side":"sell","exchange":exchange}
        for exchange in ("binance","bybit")
    ])
    reopened.update(london,pd.Timestamp("2026-01-01 08:01:00Z")); after=reopened.session_cvd(pd.Timestamp("2026-01-01 08:01:00Z"))
    assert after["session"]=="london" and after["combined"]==-200.0


def test_frozen_sweep_state_cannot_repaint_and_multiple_sweeps_coexist(tmp_path):
    store=FlowStateStore(tmp_path/"flow.json"); now=pd.Timestamp("2026-01-01 00:05:00Z")
    first={"zone":"a","sweep_time":"2026-01-01T00:01:00Z","flow_state":"provisional","orderflow_score":.3}
    second={"zone":"b","sweep_time":"2026-01-01T00:02:00Z","flow_state":"provisional","orderflow_score":.4}
    store.record_sweeps([first,second],now)
    store.record_sweeps([{**first,"flow_state":"frozen","orderflow_score":.5}],now+pd.Timedelta(minutes=1))
    store.record_sweeps([{**first,"flow_state":"frozen","orderflow_score":.9}],now+pd.Timedelta(minutes=2))
    states=store.sweep_states()
    assert len(states)==2
    assert states["a|2026-01-01T00:01:00+00:00"]["orderflow_score"]==.5


def test_provisional_flow_cannot_create_entry(monkeypatch):
    idx=pd.date_range("2026-01-01",periods=100,freq="min",tz="UTC")
    ohlc=pd.DataFrame({"open":100.0,"high":101.0,"low":99.0,"close":100.0,"volume":10.0},index=idx)
    trades=pd.DataFrame({"time":idx,"price":100.0,"qty":1.0,"side":"buy"})
    frames={"15m":ohlc.iloc[::15],"1h":ohlc.iloc[-40:],"4h":ohlc.iloc[-40:]}
    zone=Zone("waiting","swing","below",95,96,2,idx[1],idx[2])
    monkeypatch.setattr(Predictor,"_regime_bias",lambda self,frames:"bullish")
    monkeypatch.setattr("btc_predictor.strategy.build_projected_zones",lambda frame:[zone])
    monkeypatch.setattr("btc_predictor.strategy.atr",lambda frame:pd.Series(5.0,index=frame.index))
    monkeypatch.setattr("btc_predictor.strategy.detect_sweep",lambda *args,**kwargs:{"confirmed":False,"status":"waiting_reclaim","time":idx[-1],"depth_atr":.2,"extreme":94.0})
    monkeypatch.setattr("btc_predictor.strategy.footprint_confirmation",lambda *args,**kwargs:(True,{"reason":"confirmed","score":.9,"threshold":.4,"market_flow_score":.9,"market_flow_threshold":.4,"market_flow_confirmed":True,"raw_footprint_score":.9,"raw_footprint_threshold":.4,"raw_footprint_confirmed":True,"raw_footprint_eligible":True}))
    result=Predictor().predict(ohlc,trades,frames=frames)
    assert result.flow_state=="provisional" and result.orderflow_evaluation_status=="provisional"
    assert result.entry is None and result.orderflow_confirmation is False


def test_calibrated_gate_requires_passed_artifact(tmp_path):
    path=tmp_path/"artifact.json"; path.write_text(json.dumps({"promotion_passed":False,"selected_config":{"market_threshold":.5,"raw_threshold":.5,"price_bucket":50,"full_credit_ratio":2}}))
    config,_=load_flow_gate(path,"calibrated"); assert config["gate_mode"]=="shadow"
    path.write_text(json.dumps({"promotion_passed":True,"run_hash":"abc","selected_config":{"market_threshold":.5,"raw_threshold":.55,"price_bucket":50,"full_credit_ratio":2}}))
    config,_=load_flow_gate(path,"calibrated")
    assert config["gate_mode"]=="calibrated" and config["raw_threshold"]==.55 and config["artifact_run_hash"]=="abc"


def test_independent_gate_needs_no_calibration_artifact(tmp_path):
    config,artifact=load_flow_gate(tmp_path/"missing.json","independent")
    assert artifact is None
    assert config["gate_mode"]=="independent"
    assert config["market_threshold"]==.40 and config["raw_threshold"]==.40


def test_flow_evidence_window_excludes_pre_breach_bars(monkeypatch):
    idx=pd.date_range("2026-01-01 00:01",periods=22,freq="min",tz="UTC")
    features=pd.DataFrame({
        "delta_z":[-5.0]*20+[0.0,0.0],
        "sell_absorption":[True]*20+[False,False],
        "buy_absorption":False,
        "bullish_delta_reversal":False,
        "bearish_delta_reversal":False,
        "delta":[-100.0]*20+[0.0,0.0],
        "price_response":0.0,
        "low_price_impact_score":0.0,
        "intensity_z":0.0,
    },index=idx)
    monkeypatch.setattr("btc_predictor.footprint.flow_features_from_bars",lambda *args,**kwargs:features)
    sweep_time=idx[-2]; decision_time=idx[-1]
    trades=pd.DataFrame([
        {"time":sweep_time-pd.Timedelta(seconds=50),"price":100,"qty":1,"side":"sell","exchange":"binance"},
        {"time":sweep_time-pd.Timedelta(seconds=40),"price":100,"qty":1,"side":"sell","exchange":"bybit"},
        {"time":decision_time-pd.Timedelta(seconds=50),"price":101,"qty":1,"side":"buy","exchange":"binance"},
        {"time":decision_time-pd.Timedelta(seconds=40),"price":101,"qty":1,"side":"buy","exchange":"bybit"},
    ])
    _,details=footprint_confirmation(
        trades,_flow_bars(),"bullish",sweep_time,decision_time,
        gate_mode="independent",market_threshold=0.0,raw_threshold=0.0,
    )
    assert details["market_flow_window_bars"]==2
    assert details["raw_sweep_trades"]==4
    assert details["flow_window_start"]==(sweep_time-pd.Timedelta(minutes=1)).isoformat()


def test_predictor_recalculates_provisional_bars_but_never_repaints_frozen(monkeypatch):
    idx=pd.date_range("2026-01-01",periods=101,freq="min",tz="UTC")
    ohlc=pd.DataFrame({"open":100.0,"high":101.0,"low":99.0,"close":100.0,"volume":10.0},index=idx)
    trades=pd.DataFrame({"time":idx,"price":100.0,"qty":1.0,"side":"buy","exchange":"binance"})
    zone=Zone("episode","swing","below",95,96,2,idx[1],idx[2])
    monkeypatch.setattr(Predictor,"_regime_bias",lambda self,frames:"bullish")
    monkeypatch.setattr("btc_predictor.strategy.build_projected_zones",lambda frame:[zone])
    monkeypatch.setattr("btc_predictor.strategy.atr",lambda frame:pd.Series(5.0,index=frame.index))
    calls=[]
    def fake_flow(*args,**kwargs):
        calls.append(pd.Timestamp(args[4]))
        score=.2+.1*len(calls)
        return False,{"reason":"score_below_threshold","score":score,"threshold":.4,
            "market_flow_score":score,"market_flow_threshold":.4,"market_flow_confirmed":False,
            "raw_footprint_score":score,"raw_footprint_threshold":.4,"raw_footprint_confirmed":False,
            "raw_footprint_eligible":False,"contributing_exchanges":[]}
    monkeypatch.setattr("btc_predictor.strategy.footprint_confirmation",fake_flow)
    frames={"15m":ohlc.iloc[::15],"1h":ohlc.iloc[-40:],"4h":ohlc.iloc[-40:]}
    predictor=Predictor()
    monkeypatch.setattr("btc_predictor.strategy.detect_sweep",lambda *args,**kwargs:{
        "confirmed":False,"status":"waiting_reclaim","time":idx[-4],"depth_atr":.2,"extreme":94.0,
    })
    first=predictor.predict(ohlc.iloc[:-1],trades.iloc[:-1],frames=frames)
    second=predictor.predict(ohlc,trades,frames=frames)
    assert first.entry is None and second.entry is None and len(calls)==2
    monkeypatch.setattr("btc_predictor.strategy.detect_sweep",lambda *args,**kwargs:{
        "confirmed":True,"status":"confirmed","time":idx[-4],"reclaim_time":idx[-3],"depth_atr":.2,"extreme":94.0,
    })
    frozen1=predictor.predict(ohlc.iloc[:-1],trades.iloc[:-1],frames=frames)
    frozen2=predictor.predict(ohlc,trades,frames=frames)
    assert frozen1.flow_state=="frozen" and frozen2.flow_state=="frozen"
    assert frozen1.orderflow_score==frozen2.orderflow_score
    assert len(calls)==3


def test_walk_forward_and_holdout_are_non_overlapping():
    start=pd.Timestamp("2026-01-01",tz="UTC"); folds=walk_forward_periods(start); holdout=start+pd.Timedelta(days=40)
    assert [(item["validation_start"]-start).days for item in folds]==[20,25,30,35]
    assert all(item["train_end"]==item["validation_start"] and item["validation_end"]<=holdout for item in folds)


def test_event_outcome_charges_fees_and_slippage():
    idx=pd.date_range("2026-01-01 00:01",periods=3,freq="min",tz="UTC")
    bars=pd.DataFrame({"open":100.0,"high":100.0,"low":100.0,"close":100.0},index=idx)
    event={"decision_time":idx[0],"bias":"bullish","stop":90.0,"target":110.0}
    outcome=_event_outcome(event,bars,idx[-1]+pd.Timedelta(minutes=1))
    assert outcome["pnl_per_unit"]<0
    assert outcome["entry"]>100.0 and outcome["exit"]<100.0
