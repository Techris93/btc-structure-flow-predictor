import pandas as pd

from btc_predictor.models import Zone
from btc_predictor.strategy import Predictor


def test_strategy_uses_setup_atr_checks_all_zones_and_leaves_probability_uncalibrated(monkeypatch):
    idx=pd.date_range("2025-01-01",periods=100,freq="min",tz="UTC")
    o=pd.DataFrame({"open":100.,"high":101.,"low":99.,"close":100.,"volume":10.},index=idx)
    o.iloc[-1,o.columns.get_loc("low")]=89; o.iloc[-1,o.columns.get_loc("close")]=92
    setup=o.iloc[::15].copy()
    frames={"15m":setup,"1h":o.iloc[-40:].copy(),"4h":o.iloc[-40:].copy()}
    trades=pd.DataFrame({"time":idx,"price":100.,"qty":1.,"side":"buy"})
    near=Zone("near","swing","below",95,96,2,idx[1],idx[2])
    swept=Zone("swept","swing","below",90,91,3,idx[1],idx[2])
    target=Zone("target","swing","above",110,111,1,idx[1],idx[2])
    monkeypatch.setattr(Predictor,"_regime_bias",lambda self,frames:"bullish")
    monkeypatch.setattr("btc_predictor.strategy.build_projected_zones",lambda frame:[near,swept,target])
    monkeypatch.setattr("btc_predictor.strategy.atr",lambda frame:pd.Series(5.,index=frame.index))
    monkeypatch.setattr("btc_predictor.strategy.footprint_confirmation",lambda *a,**k:(True,{"reason":"confirmed","agreement":True}))
    result=Predictor(min_rr=.1).predict(o,trades,frames=frames)
    assert result.zone == "swept"
    assert result.stop <= result.entry - 1.5*5
    assert result.probability_tp_before_sl is None
