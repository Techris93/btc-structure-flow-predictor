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
