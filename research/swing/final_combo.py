"""Put the pieces together on one consistent basis (2003-2026) and compare whole portfolios."""
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from marketalgo import backtest as bt, features as ft, model as ml
from marketalgo.config import Config
from marketalgo.data import load_prices
from engine import STOCK_DIR
pd.set_option("display.width", 250)

px = pd.read_parquet(STOCK_DIR / "leverage_extra.parquet")
ret = px.pct_change(fill_method=None)
bill = px["^IRX"].ffill() / 100 / 252
FUND, MARGIN = 0.018, 0.04            # yearly cost above T-bills: measured for SSO; assumed for a margin loan
START, SPLIT = pd.Timestamp("2003-03-01"), pd.Timestamp("2015-01-01")

# ---- the fund-rotation rule already in use, with and without margin borrowing
base = Config(core_weight=0.0)
prices = load_prices(base.tickers, base.start)
wide = ft.build_wide_features(prices, base)
def rule(leverage):
    runs = []
    for o in range(0, 10, 2):
        cfg = replace(base, leverage=leverage, offset=o)
        d = prices.index[ml.rebalance_positions(len(prices), cfg)]
        w = bt.weights_from_scores(wide["mom_12_1"].loc[d], (wide["dist_sma200"] > 0).loc[d], cfg)
        runs.append(bt.simulate(prices, w, cfg)["ret"])
    return pd.concat(runs, axis=1).mean(axis=1)
rule1, rule2 = rule(1.0), rule(2.0)

# ---- S&P 500 exposure managed by volatility
spy = ret["SPY"].dropna()
def managed(target=0.20, lookback=20, cap=2.0, band=0.10, lag=1, every=1, spread=FUND, dip_boost=0.0):
    sigma = spy.rolling(lookback).std() * np.sqrt(252)
    want = (target / sigma).clip(upper=cap).fillna(0.0).to_numpy(); held = np.zeros(len(spy)); cur = 0.0
    for i in range(len(spy)):
        if i % every == 0 and abs(want[i] - cur) > band: cur = want[i]
        held[i] = cur
    e = pd.Series(held, index=spy.index)
    if dip_boost:                                   # add exposure for the few days of a dip-buying signal
        p = px["SPY"].reindex(spy.index)
        d = p.diff(); up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
        rsi2 = 100 - 100 / (1 + up / dn)
        enter = ((rsi2 < 10) & (p > p.rolling(200).mean())).to_numpy(); leave = (p > p.rolling(5).mean()).to_numpy()
        on = np.zeros(len(p), bool); h = False
        for i in range(len(p)):
            if h and leave[i]: h = False
            elif not h and enter[i]: h = True
            on[i] = h
        e = (e + dip_boost * on).clip(upper=cap)
    e = e.shift(lag).fillna(0.0)
    rf = bill.reindex(spy.index).ffill().fillna(0.0)
    out = e * spy - (e - 1).clip(lower=0) * (rf + spread / 252) + (1 - e).clip(lower=0) * rf - e.diff().abs().fillna(0) * 2 / 1e4
    return out, e

def stats(r):
    r = r[r.index >= START].dropna(); rows = {}
    for name, x in (("2003-2014", r[r.index < SPLIT]), ("2015-2026", r[r.index >= SPLIT]), ("All", r)):
        rows[name] = (1 + x).prod() ** (252 / len(x)) - 1
        eq = (1 + x).cumprod(); rows[f"drop {name}"] = (eq / eq.cummax() - 1).min()
    m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1; y = (1 + r).groupby(r.index.year).prod() - 1
    rf = bill.reindex(r.index).ffill().fillna(0)
    rows.update({"Sharpe": (r - rf).mean() * 252 / (r.std() * np.sqrt(252)), "Worst month": m.min(), "Worst year": y.min(), "Losing years": int((y < 0).sum())})
    return rows

core, e = managed()
mix = lambda a, b: (0.5 * a + 0.5 * b).dropna()
table = {
    "SPY, plain": stats(spy),
    "P0 today: half SPY + half fund rule at 2x (margin loan)": stats(mix(spy, rule2)),
    "P1 all in volatility-managed S&P (SPY+SSO)": stats(core),
    "P2 half managed S&P + half fund rule, no margin loan": stats(mix(core, rule1)),
    "P3 half managed S&P + half fund rule at 2x (margin loan)": stats(mix(core, rule2)),
    "   fund rule alone, no borrowing": stats(rule1),
    "   fund rule alone, 2x on margin": stats(rule2),
}
print(pd.DataFrame(table).T.round(3).to_string())
ee = e[e.index >= START]
print(f"\nmanaged S&P exposure: average {ee.mean():.2f}x, at 2x on {(ee >= 1.99).mean():.0%} of days, below 1x on {(ee < 1).mean():.0%}, changes per year {((ee.diff().abs() > 0).sum() / (len(ee) / 252)):.0f}")
print("exposure by year-end:", {y: round(float(v), 2) for y, v in ee.groupby(ee.index.year).last().items() if y >= 2018})

print("\n=== How fussy is the managed-S&P piece about how it is run? (P1, 2003-2026) ===")
rows = {}
for label, kw in (("check daily, trade same day's close", {}), ("check daily, trade next day", {"lag": 2}), ("check every 2 days", {"every": 2, "lag": 2}),
                  ("check every 3 days", {"every": 3, "lag": 2}), ("check weekly", {"every": 5, "lag": 2}), ("check every 2 weeks", {"every": 10, "lag": 2}),
                  ("wider no-trade band 0.2", {"band": 0.2, "lag": 2}), ("target 15%", {"target": 0.15, "lag": 2}), ("target 25%", {"target": 0.25, "lag": 2}),
                  ("with dip-buying boost +0.5x", {"dip_boost": 0.5, "lag": 2}), ("borrowing by margin loan instead of SSO", {"spread": MARGIN, "lag": 2})):
    out, ex = managed(**kw); s = stats(out); ex = ex[ex.index >= START]
    rows[label] = {k: s[k] for k in ("2003-2014", "2015-2026", "All", "drop All", "Sharpe", "Worst month")} | {"changes/yr": (ex.diff().abs() > 0).sum() / (len(ex) / 252)}
print(pd.DataFrame(rows).T.round(3).to_string())

sig_now = spy.rolling(20).std().iloc[-1] * np.sqrt(252)
print(f"\ntoday: S&P 20-day swing {sig_now:.1%} a year -> target exposure {min(2.0, 0.20 / sig_now):.2f}x; held now by the rule {float(e.iloc[-1]):.2f}x (as of {spy.index[-1].date()})")
