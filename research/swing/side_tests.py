"""Three ideas that do not need stock picking: overnight holding, volatility-managed leverage, sector rotation."""
import numpy as np, pandas as pd
from engine import load, START, SPLIT, RECENT, TRADING_DAYS, nw_tstat
pd.set_option("display.width", 250)
d = load()
AO, A = d.adj_open, d.adj
rf = A["SHY"].pct_change(fill_method=None).fillna(0.0)          # stand-in for the cash rate
SPREAD = 0.04

def st(r, extra=None):
    r = r[r.index >= START]
    out = {}
    for name, x in (("2003-2014", r[r.index < SPLIT]), ("2015-2026", r[r.index >= SPLIT]), ("2021-2026", r[r.index >= RECENT]), ("All", r)):
        eq = (1 + x).cumprod(); yrs = len(x) / TRADING_DAYS
        out[name] = eq.iloc[-1] ** (1 / yrs) - 1
    eq = (1 + r).cumprod()
    out["Sharpe"] = r.mean() * 252 / (r.std() * np.sqrt(252)); out["Vol"] = r.std() * np.sqrt(252)
    out["Worst drop"] = (eq / eq.cummax() - 1).min()
    m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1; out["Worst month"] = m.min()
    if extra: out.update(extra)
    return out

# ---------------------------------------------------------------- (a) overnight vs daytime
print("=== (a) When do the gains happen? close-to-open (overnight) vs open-to-close (daytime) ===")
rows = {}
for tk in ("SPY", "QQQ"):
    night = (AO[tk] / A[tk].shift(1) - 1).fillna(0.0); day = (A[tk] / AO[tk] - 1).fillna(0.0)
    full = A[tk].pct_change(fill_method=None).fillna(0.0)
    rows[f"{tk} held always"] = st(full)
    rows[f"{tk} overnight only, no costs"] = st(night)
    rows[f"{tk} daytime only, no costs"] = st(day)
    for c in (0.5, 1.0):
        rows[f"{tk} overnight only, {c} bp per trade, cash earns interest by day"] = st(night - 2 * c / 1e4)
    # borrowed money used only overnight: 1x by day, 2x at night (interest is charged on overnight loans)
    rows[f"{tk} 1x by day + 2x at night"] = st(full + night - (rf + SPREAD / 252) - 2 * 0.5 / 1e4)
    rows[f"{tk} 2x always"] = st(2 * full - (rf + SPREAD / 252))
print(pd.DataFrame(rows).T.round(3).to_string())
n = (AO["SPY"] / A["SPY"].shift(1) - 1).dropna(); n = n[n.index >= START]
print("SPY overnight mean per day %.4f%%, t=%.2f; since 2015 %.4f%%, t=%.2f" % (100 * n.mean(), nw_tstat(n.values), 100 * n[n.index >= SPLIT].mean(), nw_tstat(n[n.index >= SPLIT].values)))

# ------------------------------------------------------- (b) volatility-managed leverage
print("\n=== (b) Borrow more when the market is calm, less when it is wild (S&P 500, cap 2x) ===")
def vol_managed(tk, target, cap=2.0, band=0.10, lookback=20, lag=1, cost_bps=2.0):
    r = A[tk].pct_change(fill_method=None).fillna(0.0)
    sigma = r.rolling(lookback).std() * np.sqrt(252)
    want = (target / sigma).clip(upper=cap).fillna(0.0).to_numpy()
    held = np.zeros(len(r)); cur = 0.0
    for i in range(len(r)):
        if abs(want[i] - cur) > band: cur = want[i]
        held[i] = cur
    e = pd.Series(held, index=r.index).shift(lag).fillna(0.0)       # decided at a close, in place `lag` days later
    turn = e.diff().abs().fillna(0.0)
    ret = e * r - (e - 1).clip(lower=0) * (rf + SPREAD / 252) + (1 - e).clip(lower=0) * rf - turn * cost_bps / 1e4
    return ret, e
def constant(tk, lev):
    r = A[tk].pct_change(fill_method=None).fillna(0.0)
    return lev * r - max(lev - 1, 0) * (rf + SPREAD / 252) + max(1 - lev, 0) * rf
rows = {}
for target in (0.10, 0.15, 0.20, 0.25, 0.30):
    ret, e = vol_managed("SPY", target); e = e[e.index >= START]
    rows[f"managed, target swing {target:.0%}"] = st(ret, {"Avg borrowed x": e.mean(), "Trades/yr": (e.diff().abs() > 0).sum() / (len(e) / 252)})
    rows[f"   constant {e.mean():.2f}x for comparison"] = st(constant("SPY", e.mean()), {"Avg borrowed x": e.mean()})
rows["constant 1x (plain SPY)"] = st(constant("SPY", 1.0)); rows["constant 2x"] = st(constant("SPY", 2.0))
print(pd.DataFrame(rows).T.round(3).to_string())
print("\nsame idea, slower yardstick (60-day) and on QQQ, target 20%:")
rows = {}
for tk, lb in (("SPY", 60), ("QQQ", 20), ("QQQ", 60)):
    ret, e = vol_managed(tk, 0.20, lookback=lb); e = e[e.index >= START]
    rows[f"{tk} managed {lb}d"] = st(ret, {"Avg borrowed x": e.mean()}); rows[f"   {tk} constant {e.mean():.2f}x"] = st(constant(tk, e.mean()))
print(pd.DataFrame(rows).T.round(3).to_string())

# --------------------------------------------------------------- (c) sector rotation
print("\n=== (c) Hold the 3 sector funds that rose most over the past month, reviewed weekly ===")
SECT = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
night = (AO[SECT] / A[SECT].shift(1) - 1).fillna(0.0); day = (A[SECT] / AO[SECT] - 1).fillna(0.0)
def rotation(lookback, n=3, every=5, offset=0, worst=False):
    score = (A[SECT] / A[SECT].shift(lookback) - 1)
    rk = score.rank(axis=1, ascending=worst)
    w = ((rk <= n).astype(float) / n).iloc[offset::every].reindex(A.index).ffill().fillna(0.0)
    w1, w2 = w.shift(1).fillna(0.0), w.shift(2).fillna(0.0)                 # decided at close, traded next open
    return (1 + (w2 * night).sum(axis=1)) * (1 + (w1 * day).sum(axis=1)) - 1 - (w1 - w2).abs().sum(axis=1) * 5 / 1e4
rows = {"SPY": st(A["SPY"].pct_change(fill_method=None).fillna(0.0)),
        "All sectors, equal weight": st(A[SECT].pct_change(fill_method=None).mean(axis=1).fillna(0.0))}
for lb in (21, 63, 126):
    rows[f"top 3 by {lb}-day return"] = pd.DataFrame([st(rotation(lb, offset=o)) for o in range(5)]).mean().to_dict()
rows["bottom 3 by 21-day return (placebo)"] = pd.DataFrame([st(rotation(21, offset=o, worst=True)) for o in range(5)]).mean().to_dict()
print(pd.DataFrame(rows).T.round(3).to_string())
