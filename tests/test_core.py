import pandas as pd
from btc_predictor.indicators import atr
from btc_predictor.structure import confirmed_pivots
from btc_predictor.footprint import build_footprint, orderflow_features
from btc_predictor.synthetic import make_synthetic

def test_atr_and_confirmed_pivots_are_defined():
    o, _ = make_synthetic(2)
    assert atr(o).notna().sum() > 0
    p = confirmed_pivots(o)
    assert (p.available_at >= p.pivot_time).all()

def test_footprint_delta_and_cvd():
    o, t = make_synthetic(1)
    f = build_footprint(t)
    assert "delta" in f and "total" in f
    features = orderflow_features(t)
    assert "cvd" in features and features.index.is_monotonic_increasing

def test_walk_forward_has_no_future_training_overlap():
    from btc_predictor.backtest import walk_forward_splits
    idx = pd.date_range("2025-01-01", periods=1000, freq="min", tz="UTC")
    for train, test in walk_forward_splits(idx, 500, 100):
        assert train[-1] < test[0]
