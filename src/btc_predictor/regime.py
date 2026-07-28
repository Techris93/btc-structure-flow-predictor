"""Causal regime classification from the setup frame."""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 96  # 24h of 15m bars


def efficiency_ratio(close: pd.Series, window: int = DEFAULT_WINDOW) -> pd.Series:
    """Kaufman efficiency ratio: |net move| / path length over the window."""
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    return (net / path).where(path > 0)


def classify(setup: pd.DataFrame, range_er: float = 0.12, trend_er: float = 0.24,
             window: int = DEFAULT_WINDOW) -> dict | None:
    """Classify the regime at the last bar of the setup frame.

    Thresholds 0.12/0.24 are the train-window terciles (Jun 18 - Jul 14),
    rounded and frozen before any out-of-sample evaluation. Returns None when
    the frame is too short; callers treat None as "no constraint".
    """
    if setup is None or len(setup) <= window + 1:
        return None
    close = setup.close.astype(float)
    er = efficiency_ratio(close, window)
    value = float(er.iloc[-1]) if len(er) else np.nan
    if not np.isfinite(value):
        return None
    delta = float(close.iloc[-1] - close.iloc[-1 - window])
    drift = 0 if delta == 0 else (1 if delta > 0 else -1)
    regime = "range" if value < range_er else ("trend" if value >= trend_er else "transition")
    return {"er": value, "regime": regime, "drift": drift}
