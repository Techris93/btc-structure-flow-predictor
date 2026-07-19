from __future__ import annotations
import pandas as pd
import numpy as np
from .indicators import rolling_zscore

def build_footprint(trades: pd.DataFrame, bar_freq="1min", price_bucket=25.0) -> pd.DataFrame:
    x=trades.copy(); x["time"]=pd.to_datetime(x["time"], utc=True); x["notional"]=x["price"]*x["qty"]; x["bar"]=x.time.dt.floor(bar_freq); x["price_level"]=(x.price/price_bucket).round()*price_bucket
    f=x.groupby(["bar","price_level","side"])["notional"].sum().unstack("side", fill_value=0)
    for c in ("buy","sell"):
        if c not in f: f[c]=0.0
    f["delta"]=f.buy-f.sell; f["total"]=f.buy+f.sell
    return f.sort_index()

def orderflow_features(trades: pd.DataFrame, freq="1min", window=100) -> pd.DataFrame:
    x=trades.copy(); x["time"]=pd.to_datetime(x["time"], utc=True); x["notional"]=x["price"]*x["qty"]
    x["signed"] = np.where(x.side.str.lower().eq("buy"), x.notional, -x.notional)
    g=x.set_index("time").resample(freq).agg(price=("price","last"), buy=("signed",lambda s:s[s>0].sum()), sell=("signed",lambda s:-s[s<0].sum()), volume=("notional","sum"), trades=("price","size")).fillna(0)
    g["delta"]=g.buy-g.sell; g["cvd"]=g.delta.cumsum(); g["delta_z"]=rolling_zscore(g.delta,window); g["intensity_z"]=rolling_zscore(g.volume,window)
    g["price_response"] = g["price"].diff() / g["volume"].replace(0,np.nan)
    g["delta_reversal"]=(g.delta.shift(1)<0)&(g.delta>0)
    g["sell_absorption"]=(g.delta_z < -1.0)&(g.price.diff().abs() < g.price.rolling(10).std().fillna(0)*.25)
    g["buy_absorption"]=(g.delta_z > 1.0)&(g.price.diff().abs() < g.price.rolling(10).std().fillna(0)*.25)
    return g
