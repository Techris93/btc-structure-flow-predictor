import pandas as pd

from btc_predictor.models import Zone
from btc_predictor.structure import confirmed_pivots, structure_events
from btc_predictor.strategy import Predictor
from btc_predictor.timeframes import completed_timeframes, index_candles_by_close
from btc_predictor.zones import build_projected_zones


def structured_bars(count=180):
    idx=pd.date_range("2025-01-01 00:01",periods=count,freq="min",tz="UTC")
    close=pd.Series([100 + ((i%20)-10)*.2 for i in range(count)],index=idx)
    return pd.DataFrame({"open":close.shift(1).fillna(close),"high":close+1,"low":close-1,"close":close,"volume":10.},index=idx)


def test_each_swing_break_is_emitted_once():
    x=structured_bars(); x.iloc[-15:,x.columns.get_loc("close")]=120; x.iloc[-15:,x.columns.get_loc("high")]=121; x.iloc[-15:,x.columns.get_loc("low")]=119
    events=structure_events(x)
    assert not events.empty
    assert events.swing_id.value_counts().max() == 1
    assert confirmed_pivots(x).swing_id.is_unique


def test_zone_lifecycle_is_causal_and_ids_do_not_merge():
    x=structured_bars(400); zones=build_projected_zones(x,lookback=400,expiry_bars=30)
    assert zones and len({z.zone_id for z in zones}) == len(zones)
    assert any(z.expires_at is not None for z in zones)
    assert all(pd.Timestamp(z.available_at) >= pd.Timestamp(z.created_at) for z in zones)
    assert all(z.swept_at is None or pd.Timestamp(z.swept_at) > pd.Timestamp(z.available_at) for z in zones)
    earlier=build_projected_zones(x.iloc[:-20],lookback=400)
    later={z.zone_id:z for z in build_projected_zones(x,lookback=400)}
    for z in earlier:
        if z.zone_id in later:
            assert (z.low,z.high,z.created_at,z.available_at)==(later[z.zone_id].low,later[z.zone_id].high,later[z.zone_id].created_at,later[z.zone_id].available_at)


def test_close_indexing_and_completed_mtf_boundaries():
    opens=pd.date_range("2025-01-01",periods=61,freq="min",tz="UTC")
    x=pd.DataFrame({"open":1.,"high":2.,"low":0.,"close":1.,"volume":1.},index=opens)
    closed=index_candles_by_close(x,"1min")
    assert closed.index[0] == opens[0] + pd.Timedelta(minutes=1)
    frames=completed_timeframes(closed,closed.index[-1])
    assert all(frame.empty or frame.index.max() <= closed.index[-1] for frame in frames.values())
    assert frames["15m"].index.minute.isin([0,15,30,45]).all()


def test_timeframe_disagreement_is_neutral_and_held_only_internal(monkeypatch):
    def fake(frame):
        bias=frame.attrs["bias"]
        return pd.DataFrame([{"bias":bias,"event":"BOS","level":1,"swing_id":bias}],index=[frame.index[-1]])
    monkeypatch.setattr("btc_predictor.strategy.structure_events",fake)
    idx=pd.date_range("2025-01-01",periods=40,freq="h",tz="UTC")
    bull=pd.DataFrame({"close":range(40)},index=idx); bull.attrs["bias"]="bullish"
    bear=bull.copy(); bear.attrs["bias"]="bearish"
    p=Predictor(); p._held_bias="bullish"
    assert p._regime_bias({"4h":bull,"1h":bear}) == "neutral"
    assert p._held_bias == "bullish"


def test_opposing_regime_requires_choch(monkeypatch):
    event="BOS"
    def fake(frame):
        return pd.DataFrame([{"bias":"bearish","event":event,"level":1,"swing_id":"x"}],index=[frame.index[-1]])
    monkeypatch.setattr("btc_predictor.strategy.structure_events",fake)
    idx=pd.date_range("2025-01-01",periods=40,freq="h",tz="UTC"); frame=pd.DataFrame({"close":range(40)},index=idx)
    p=Predictor(); p._held_bias="bullish"
    assert p._regime_bias({"4h":frame,"1h":frame}) == "neutral"
    event="CHoCH"
    assert p._regime_bias({"4h":frame,"1h":frame}) == "bearish"
