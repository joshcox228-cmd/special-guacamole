"""Leverage done cheaper and smarter: 2x funds versus a margin loan, and volatility-managed exposure."""
import sys, time
import numpy as np, pandas as pd
from engine import STOCK_DIR, TRADING_DAYS, nw_tstat
from download import fetch
pd.set_option("display.width", 250)

cache = STOCK_DIR / "leverage_extra.parquet"
if cache.exists():
    px = pd.read_parquet(cache)
else:
    want = {"SPY": "1993-01-01", "QQQ": "1999-01-01", "SSO": "2006-01-01", "QLD": "2006-01-01", "UPRO": "2009-01-01", "TQQQ": "2010-01-01",
            "SHY": "2002-01-01", "^IRX": "1960-01-01", "^GSPC": "1950-01-01", "^VIX": "1990-01-01"}
    got = {}
    for k, start in want.items():
        df = fetch(k, start); got[k] = df["adj"] if df is not None else None; print(k, None if df is None else (len(df), df.index[0].date()))
    px = pd.concat({k: v for k, v in got.items() if v is not None}, axis=1).sort_index(); px.to_parquet(cache)
ret = px.pct_change(fill_method=None)
bill = (px["^IRX"].ffill() / 100 / 252)                      # 13-week T-bill yield, per day

def cagr(r): r = r.dropna(); return (1 + r).prod() ** (252 / len(r)) - 1
def dd(r): eq = (1 + r.dropna()).cumprod(); return (eq / eq.cummax() - 1).min()
def sharpe(r): r = r.dropna(); return (r.mean() - bill.reindex(r.index).fillna(0).mean()) * 252 / (r.std() * np.sqrt(252))

# ------------------------------------------------ 1. what does a 2x fund really charge for its borrowing?
print("=== 1. Cost of leverage inside a leveraged fund, measured from its actual returns ===")
for fund, base, k in (("SSO", "SPY", 2), ("QLD", "QQQ", 2), ("UPRO", "SPY", 3), ("TQQQ", "QQQ", 3)):
    j = pd.concat([ret[fund], ret[base], bill], axis=1, keys=["f", "b", "rf"]).dropna()
    gap = (j["f"] - k * j["b"])                               # daily shortfall versus k times the index
    implied = -gap.mean() * 252 / (k - 1)                     # yearly cost per borrowed dollar
    print(f"{fund}: since {j.index[0].date()}  fund {cagr(j['f']):.1%}/yr vs index {cagr(j['b']):.1%}/yr | "
          f"cost per borrowed dollar {implied:.2%}/yr, T-bill averaged {j['rf'].mean() * 252:.2%} -> {implied - j['rf'].mean() * 252:+.2%} above T-bills | "
          f"day-to-day match with {k}x index: corr {np.corrcoef(j['f'], k * j['b'])[0, 1]:.4f}")

# ------------------------------------------------ 2. volatility-managed exposure, many variants
def managed(r, sigma, target, cap=2.0, band=0.10, lag=1, spread=0.015, cost_bps=2.0, rf=bill):
    want = (target / sigma).clip(upper=cap).fillna(0.0).to_numpy(); held = np.zeros(len(r)); cur = 0.0
    for i in range(len(r)):
        if abs(want[i] - cur) > band: cur = want[i]
        held[i] = cur
    e = pd.Series(held, index=r.index).shift(lag).fillna(0.0)
    rfd = rf.reindex(r.index).ffill().fillna(0.0)
    out = e * r - (e - 1).clip(lower=0) * (rfd + spread / 252) + (1 - e).clip(lower=0) * rfd - e.diff().abs().fillna(0) * cost_bps / 1e4
    return out, e
def constant(r, lev, spread=0.015, rf=bill):
    rfd = rf.reindex(r.index).ffill().fillna(0.0)
    return lev * r - max(lev - 1, 0) * (rfd + spread / 252) + max(1 - lev, 0) * rfd

r = ret["SPY"].dropna()
sig = {f"{n}-day realised": r.rolling(n).std() * np.sqrt(252) for n in (10, 20, 30, 40, 60)}
sig["exponential (RiskMetrics 0.94)"] = np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean() * 252)
sig["VIX index x0.8"] = (px["^VIX"].reindex(r.index).ffill() / 100 * 0.8)
windows = {"2000-2014": ("2000-03-01", "2014-12-31"), "2015-2026": ("2015-01-01", None), "2000-2026": ("2000-03-01", None)}
print("\n=== 2. Volatility-managed S&P 500 exposure (cap 2x, target swing 20%), borrowing at the leveraged-fund cost ===")
rows = {}
for name, s in sig.items():
    out, e = managed(r, s, 0.20)
    row = {}
    for w, (a, b) in windows.items():
        row[w] = cagr(out.loc[a:b]); row[f"drop {w}"] = dd(out.loc[a:b])
    row["avg exposure"] = e.loc["2000-03-01":].mean(); row["Sharpe"] = sharpe(out.loc["2000-03-01":]); row["changes/yr"] = (e.loc["2000-03-01":].diff().abs() > 0).sum() / (len(e.loc["2000-03-01":]) / 252)
    rows[name] = row
for lev in (1.0, 1.25, 1.5, 2.0):
    out = constant(r, lev); row = {}
    for w, (a, b) in windows.items():
        row[w] = cagr(out.loc[a:b]); row[f"drop {w}"] = dd(out.loc[a:b])
    row["avg exposure"] = lev; row["Sharpe"] = sharpe(out.loc["2000-03-01":]); rows[f"constant {lev}x"] = row
print(pd.DataFrame(rows).T.round(3).to_string())

print("\n=== 3. Same, by target (20-day realised), fund-cost borrowing vs margin-loan borrowing (+4%) ===")
rows = {}
for target in (0.12, 0.15, 0.20, 0.25, 0.30):
    for label, sp in (("fund", 0.015), ("margin", 0.04)):
        out, e = managed(r, sig["20-day realised"], target, spread=sp); row = {}
        for w, (a, b) in windows.items():
            row[w] = cagr(out.loc[a:b]); row[f"drop {w}"] = dd(out.loc[a:b])
        e2 = e.loc["2000-03-01":]; row["avg exposure"] = e2.mean(); row["share of days at 2x"] = (e2 >= 1.99).mean(); row["share below 1x"] = (e2 < 1).mean()
        rows[f"target {target:.0%} ({label})"] = row
print(pd.DataFrame(rows).T.round(3).to_string())

# ------------------------------------------------ 4. sixty-plus years, including 1973-74, 1987, 2000-02, 2008
print("\n=== 4. Long history: S&P 500 index since 1960 (dividends approximated), 20-day rule, target 20% ===")
g = ret["^GSPC"].dropna().loc["1960-01-01":]
g = g + np.where(g.index < "1990-01-01", 0.035, 0.019) / 252      # rough dividend yield
s20 = g.rolling(20).std() * np.sqrt(252)
m, e = managed(g, s20, 0.20); c1 = constant(g, 1.0); ce = constant(g, float(e.mean())); c2 = constant(g, 2.0)
rows = {}
for label, a, b in (("1960-1979", "1960", "1979"), ("1980-1999", "1980", "1999"), ("2000-2026", "2000", None), ("1960-2026", "1960", None)):
    rows[label] = {"managed": cagr(m.loc[a:b]), f"constant {e.mean():.2f}x": cagr(ce.loc[a:b]), "1x": cagr(c1.loc[a:b]), "2x": cagr(c2.loc[a:b]),
                   "drop managed": dd(m.loc[a:b]), "drop constant": dd(ce.loc[a:b]), "drop 1x": dd(c1.loc[a:b]), "drop 2x": dd(c2.loc[a:b])}
print(pd.DataFrame(rows).T.round(3).to_string())
oct87 = pd.Timestamp("1987-10-19"); print("exposure going into the Oct-1987 crash:", round(float(e.loc[oct87]), 2), " that day: managed", round(float(m.loc[oct87]), 3), "index", round(float(g.loc[oct87]), 3))
worst = m.nsmallest(5); print("managed strategy worst days:", {str(k.date()): round(v, 3) for k, v in worst.items()})

# ------------------------------------------------ 5. with the real funds: SPY + SSO, no margin loan at all
print("\n=== 5. Built from real funds (SPY + the 2x fund SSO + cash), 2006-2026 ===")
j = pd.concat([ret["SPY"], ret["SSO"], bill], axis=1, keys=["spy", "sso", "rf"]).dropna()
s = (j["spy"].rolling(20).std() * np.sqrt(252)).reindex(j.index)
sfull = (ret["SPY"].rolling(20).std() * np.sqrt(252)).reindex(j.index)
for target in (0.15, 0.20, 0.25):
    _, e = managed(ret["SPY"].dropna(), sig["20-day realised"], target); e = e.reindex(j.index).fillna(0)
    w_sso = (e - 1).clip(lower=0); w_spy = np.where(e > 1, 2 - e, e); w_cash = (1 - e).clip(lower=0)
    real = w_sso * j["sso"] + w_spy * j["spy"] + w_cash * j["rf"] - e.diff().abs().fillna(0) * 2 / 1e4
    model, _ = managed(ret["SPY"].dropna(), sig["20-day realised"], target); model = model.reindex(j.index)
    print(f"target {target:.0%}: real funds {cagr(real):.1%}/yr, worst drop {dd(real):.0%} | my model {cagr(model):.1%}/yr | plain SPY {cagr(j['spy']):.1%}/yr, drop {dd(j['spy']):.0%} | "
          f"all-SSO {cagr(j['sso']):.1%}/yr, drop {dd(j['sso']):.0%} | 2015+: real {cagr(real.loc['2015':]):.1%} vs SPY {cagr(j['spy'].loc['2015':]):.1%} vs SSO {cagr(j['sso'].loc['2015':]):.1%}")
