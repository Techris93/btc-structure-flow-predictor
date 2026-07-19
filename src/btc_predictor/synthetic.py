from __future__ import annotations
import numpy as np, pandas as pd

def make_synthetic(days=20, seed=7):
    rng=np.random.default_rng(seed); n=days*24*60; idx=pd.date_range("2025-01-01",periods=n,freq="min",tz="UTC"); close=100000+np.cumsum(rng.normal(0,30,n)); close[1000:1200]+=np.linspace(0,4000,200); close[1200:1400]+=np.linspace(4000,0,200); o=pd.DataFrame(index=idx); o["open"]=close+rng.normal(0,5,n); o["high"]=np.maximum(o.open,close)+rng.uniform(1,20,n); o["low"]=np.minimum(o.open,close)-rng.uniform(1,20,n); o["close"]=close; o["volume"]=rng.lognormal(10,0.5,n)
    m=days*24*10; ti=pd.date_range(idx[0],idx[-1],periods=m,tz="UTC"); p=np.interp(ti.view("i8"),idx.view("i8"),close)+rng.normal(0,8,m); q=rng.lognormal(-3,.5,m); t=pd.DataFrame({"time":ti,"price":p,"qty":q,"side":np.where(rng.random(m)>.5,"buy","sell"),"exchange":rng.choice(["binance","bybit"],m)})
    return o,t
