import subprocess, json, os, sys, time
import pandas as pd
import numpy as np

def fetch_binance(interval="1h", limit=200):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval={interval}&limit={limit}"
    for _ in range(4):
        try:
            out = subprocess.check_output(["curl", "--http1.1", "--retry", "2", "-s", "--max-time", "15", url])
            data = json.loads(out.decode())
            if not isinstance(data, list):
                continue
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
            ])
            for col in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
                df[col] = df[col].astype(float)
            df["taker_buy_volume"] = df["taker_buy_base"]
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df.set_index("close_time", inplace=True)
            return df
        except Exception:
            time.sleep(0.5)
    return None

def fetch_bybit(interval="60", limit=200):
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval={interval}&limit={limit}"
    for _ in range(4):
        try:
            out = subprocess.check_output(["curl", "--http1.1", "--retry", "2", "-s", "--max-time", "15", url])
            res = json.loads(out.decode())
            rows = list(reversed(res["result"]["list"]))
            df = pd.DataFrame(rows, columns=["start_time", "open", "high", "low", "close", "volume", "turnover"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df["taker_buy_volume"] = df["volume"] * 0.5
            mins = int(interval) if interval.isdigit() else 60
            df["close_time"] = pd.to_datetime(pd.to_numeric(df["start_time"]) + mins*60*1000, unit="ms", utc=True)
            df.set_index("close_time", inplace=True)
            return df
        except Exception:
            time.sleep(0.5)
    return None

def main():
    print("Fetching past week BTC candles...")
    df_1h = fetch_binance("1h", 200)
    if df_1h is None:
        df_1h = fetch_bybit("60", 200)
    df_4h = fetch_binance("4h", 80)
    if df_4h is None:
        df_4h = fetch_bybit("240", 80)
    df_15m = fetch_binance("15m", 800)
    if df_15m is None:
        df_15m = fetch_bybit("15", 800)
    print(f"1H Bars: {len(df_1h)} ({df_1h.index[0]} to {df_1h.index[-1]})")
    print(f"4H Bars: {len(df_4h)} ({df_4h.index[0]} to {df_4h.index[-1]})")
    print(f"15M Bars: {len(df_15m)} ({df_15m.index[0]} to {df_15m.index[-1]})")
    df_1h["day"] = df_1h.index.strftime("%Y-%m-%d")
    daily = df_1h.groupby("day").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    daily["change_pct"] = ((daily["close"] - daily["open"]) / daily["open"] * 100).round(2)
    daily["range_usd"] = (daily["high"] - daily["low"]).round(1)
    print("\n=======================================================")
    print("        BTC/USDT 1-WEEK DAILY CANDLESTICK OVERVIEW     ")
    print("=======================================================")
    print(daily[["open", "high", "low", "close", "range_usd", "change_pct", "volume"]].to_string())
    print("\n=======================================================")
    print("         BTC/USDT 4-HOUR CANDLESTICKS (PAST WEEK)      ")
    print("=======================================================")
    recent_4h = df_4h.tail(42)
    for ts, row in recent_4h.iterrows():
        o, h, l, c, v = row["open"], row["high"], row["low"], row["close"], row["volume"]
        bar_type = "BULL [▲]" if c >= o else "BEAR [▼]"
        body_pct = (abs(c - o) / o) * 100
        print(f"{ts.strftime('%Y-%m-%d %H:%M UTC')} | {bar_type} | Open: {o:8.1f} | High: {h:8.1f} | Low: {l:8.1f} | Close: {c:8.1f} | Body: {body_pct:4.2f}% | Vol: {v:7.1f}")
    sys.path.insert(0, os.path.abspath("src"))
    from btc_predictor.strategy import Predictor
    from btc_predictor.research import proxy_trades
    print("\n=======================================================")
    print("     PREDICTOR DECISION REPLAY OVER THE PAST WEEK      ")
    print("=======================================================")
    predictor = Predictor(flow_gate_mode="independent", cache_closed_frames=True)
    trades = proxy_trades(df_15m)
    eval_records = []
    start_idx = max(80, len(df_15m) - 672)
    for i in range(start_idx, len(df_15m), 4):
        now_ts = df_15m.index[i]
        sub_4h = df_4h.loc[df_4h.index <= now_ts]
        sub_1h = df_1h.loc[df_1h.index <= now_ts]
        sub_15m = df_15m.loc[df_15m.index <= now_ts]
        frames = {"4h": sub_4h, "1h": sub_1h, "15m": sub_15m}
        known_trades = trades.loc[trades["time"] <= now_ts].tail(1000)
        out = predictor.predict(sub_15m, known_trades, equity=100000.0, frames=frames, flow_bars=sub_15m, flow_source="websocket")
        eval_records.append({
            "time": now_ts,
            "close": sub_15m.iloc[-1]["close"],
            "bias": out.bias,
            "regime_4h": out.regime_4h,
            "regime_1h": out.regime_1h,
            "setup_15m": out.setup_15m,
            "zone": out.zone,
            "zone_kind": out.zone_kind,
            "sweep_status": out.sweep_status,
            "sweep_depth_atr": out.sweep_depth_atr,
            "orderflow_confirmation": out.orderflow_confirmation,
            "no_trade_reason": out.no_trade_reason,
            "market_flow_score": out.market_flow_score,
            "raw_footprint_score": out.raw_footprint_score,
            "entry": out.entry,
            "stop": out.stop,
            "target": out.target,
        })
    eval_df = pd.DataFrame(eval_records)
    print(f"\nTotal 15m evaluation steps in past 7 days: {len(eval_df)}")
    print("\n--- BIAS BREAKDOWN ---")
    print(eval_df["bias"].value_counts().to_string())
    print("\n--- NO TRADE REASON BREAKDOWN ---")
    print(eval_df["no_trade_reason"].value_counts().to_string())
    print("\n--- 4H / 1H / 15M REGIME ALIGNMENT SUMMARY ---")
    print(eval_df.groupby(["regime_4h", "regime_1h", "setup_15m", "bias"]).size().reset_index(name="count").to_string())
    print("\n--- SWEEP STATUS BREAKDOWN ---")
    print(eval_df["sweep_status"].value_counts().to_string())
    print("\n=======================================================")
    print("     DETAILED BREAKDOWN OF ALL 40 CONFIRMED SWEEPS     ")
    print("=======================================================")
    confirmed_sweeps = eval_df[eval_df["sweep_status"] == "confirmed"]
    cols = ["time", "close", "bias", "zone_kind", "no_trade_reason", "market_flow_score", "raw_footprint_score", "orderflow_confirmation"]
    print(confirmed_sweeps[cols].to_string())

if __name__ == "__main__":
    main()
