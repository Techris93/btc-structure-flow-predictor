import pandas as pd

from btc_predictor.footprint import cross_exchange_agreement, orderflow_features
from btc_predictor.models import Zone
from btc_predictor.strategy import detect_sweep
from btc_predictor.trade_store import TradeStore


def test_sweep_is_atr_bounded_and_can_reclaim_over_multiple_bars():
    idx=pd.date_range("2025-01-01",periods=5,freq="min",tz="UTC")
    x=pd.DataFrame({"open":100.,"high":[101]*5,"low":[99,99,94,95,99],"close":[100,100,95,98,101],"volume":1.},index=idx)
    z=Zone("z","swing","below",96,97,1,idx[0],idx[0])
    result=detect_sweep(x,z,"bullish",10,.05,2,3)
    assert result["confirmed"] and result["reclaim_time"]==idx[-2]
    assert result["depth_atr"]==.2
    assert not detect_sweep(x,z,"bullish",1,.05,1,3)["confirmed"]


def test_orderflow_reversals_are_symmetric_and_exchange_agreement_is_required():
    idx=pd.date_range("2025-01-01",periods=3,freq="min",tz="UTC")
    trades=[]
    for exchange in ("binance","bybit"):
        trades += [{"time":idx[0]-pd.Timedelta(seconds=10),"price":100,"qty":2,"side":"buy","exchange":exchange},
                   {"time":idx[1]-pd.Timedelta(seconds=10),"price":99,"qty":3,"side":"sell","exchange":exchange}]
    t=pd.DataFrame(trades); f=orderflow_features(t,window=2)
    assert f.bearish_delta_reversal.iloc[-1]
    agrees,deltas=cross_exchange_agreement(t,idx[0]-pd.Timedelta(minutes=1),idx[1],"bearish")
    assert agrees and set(deltas)=={"binance","bybit"}


def test_trade_store_deduplicates_and_persists(tmp_path):
    path=tmp_path/"trades.sqlite3"; store=TradeStore(path); now=pd.Timestamp("2025-01-01",tz="UTC")
    row=pd.DataFrame([{"time":now,"price":100,"qty":1,"side":"buy","exchange":"binance","trade_id":"1"}])
    assert store.append(row)==1 and store.append(row)==0
    reopened=TradeStore(path); out=reopened.query(now-pd.Timedelta(seconds=1),now+pd.Timedelta(seconds=1))
    assert len(out)==1 and out.iloc[0].exchange=="binance"
