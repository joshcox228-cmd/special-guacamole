"""Would many small intraday Bitcoin trades around support and resistance have made money?

Rules fixed before looking at results (standard textbook definitions):
  support    = lowest low of the previous N bars;  resistance = highest high of the previous N bars
  BOUNCE     buy when price dips to within 0.25% of support and the bar still closes above it
  BREAKOUT   buy when a bar closes above resistance for the first time
  exit       at a profit target, at a stop-loss, or after a time limit; one position at a time
Costs are the whole question for this style, so every result is shown at four fee levels.
"""
import time
import numpy as np, pandas as pd, requests
pd.set_option("display.width", 250)

def bars(symbol, interval, rng):
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}", params={"interval": interval, "range": rng},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    res = r.json()["chart"]["result"][0]; q = res["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")}, index=pd.to_datetime(res["timestamp"], unit="s")).dropna()
    return df[~df.index.duplicated()]

def run(df, kind, lookback, target, stop, max_bars):
    h, l, c = df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    sup = pd.Series(l).shift(1).rolling(lookback).min().to_numpy(); res = pd.Series(h).shift(1).rolling(lookback).max().to_numpy()
    trades, i, n = [], lookback + 1, len(c)
    while i < n - 1:
        if kind == "bounce":
            go = l[i] <= sup[i] * 1.0025 and c[i] > sup[i]
        else:
            go = c[i] > res[i] and c[i - 1] <= res[i - 1]
        if not go:
            i += 1; continue
        entry = c[i]; tp, sl = entry * (1 + target), entry * (1 - stop); out = None
        for j in range(i + 1, min(i + 1 + max_bars, n)):
            if l[j] <= sl: out = (j, -stop); break            # if both are touched in one bar, assume the stop came first
            if h[j] >= tp: out = (j, target); break
        if out is None:
            j = min(i + max_bars, n - 1); out = (j, c[j] / entry - 1)
        trades.append(out[1]); i = out[0] + 1
    return np.array(trades)

def table(df, label, lookback, settings, max_bars):
    weeks = (df.index[-1] - df.index[0]).days / 7; hold = df.close.iloc[-1] / df.close.iloc[0] - 1
    print(f"\n=== {label}: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}; Bitcoin simply held over this period: {hold:+.0%} ===")
    rows = {}
    for kind in ("bounce", "breakout"):
        for target, stop in settings:
            t = run(df, kind, lookback, target, stop, max_bars)
            if not len(t): continue
            row = {"trades": len(t), "per week": len(t) / weeks, "win rate": (t > 0).mean(), "avg trade before fees %": 100 * t.mean()}
            for fee in (0.0, 0.002, 0.005, 0.010):
                row[f"total, fees {fee:.1%}"] = np.prod(1 + t - fee) - 1
            rows[f"{kind}: target {target:.2%}, stop {stop:.2%}"] = row
    print(pd.DataFrame(rows).T.round(3).to_string())

hourly = bars("BTC-USD", "1h", "730d")
table(hourly, "Hourly bars, about two years (support/resistance = previous 24 hours, 12-hour time limit)", 24,
      [(0.005, 0.005), (0.01, 0.005), (0.01, 0.01), (0.02, 0.01)], 12)
table(hourly, "Hourly bars, wider levels (previous 7 days, 48-hour time limit)", 168, [(0.01, 0.01), (0.02, 0.01), (0.03, 0.015)], 48)
five = bars("BTC-USD", "5m", "60d")
table(five, "5-minute bars, about two months (previous 4 hours, 2-hour time limit)", 48, [(0.0025, 0.0025), (0.005, 0.0025), (0.005, 0.005)], 24)
