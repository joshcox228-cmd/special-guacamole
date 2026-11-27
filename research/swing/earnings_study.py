"""Post-earnings drift with real earnings dates. Defined before looking at any result:

  A report counts as a GOOD SURPRISE when earnings beat the analysts' estimate AND the stock
  beat the S&P 500 by 5% or more over the report day and the day after.
  Buy at the open two days after the report date; hold 20 trading days; 10 positions.

The two-day window is used because the data does not say whether a company reported before
the open or after the close; waiting for both days means nothing is bought on unseen news.
"""
import numpy as np, pandas as pd
from engine import load, simulate, by_period, trade_stats, nw_tstat, STOCK_DIR, START, SPLIT, RECENT
from signals import Signals
pd.set_option("display.width", 250)
d = load(); sig = Signals(d)
E = pd.read_parquet(STOCK_DIR / "fmp_earnings.parquet").dropna(subset=["epsActual", "epsEstimated"])
E = E[E.symbol.isin(sig.cols) & (E.date >= "2002-06-01") & (E.date <= d.dates[-3])]
dates = d.dates; pos = dates.searchsorted(E.date.to_numpy())           # first trading day on/after the report date
E = E.assign(i0=pos)[(pos >= 2) & (pos < len(dates) - 2)]
col = {c: k for k, c in enumerate(sig.cols)}
A = sig.adj.to_numpy(); spy = sig.spy.to_numpy()
j = E.symbol.map(col).to_numpy(); i0 = E.i0.to_numpy()
react = A[i0 + 1, j] / A[i0 - 1, j] - 1 - (spy[i0 + 1] / spy[i0 - 1] - 1)
liquid = sig.u500.to_numpy()[i0 - 1, j]
E = E.assign(react=react, liquid=liquid, beat=E.epsActual > E.epsEstimated, miss=E.epsActual < E.epsEstimated).dropna(subset=["react"])
E = E[E.liquid]
print(f"{len(E)} reports from liquid stocks; beat {E.beat.mean():.0%}, miss {E.miss.mean():.0%}; average two-day move vs S&P: beats {E.react[E.beat].mean():+.2%}, misses {E.react[E.miss].mean():+.2%}")

def flags(sub):
    f = np.zeros(A.shape, bool); f[sub.i0.to_numpy() + 1, sub.symbol.map(col).to_numpy()] = True
    return pd.DataFrame(f, index=dates, columns=sig.cols)
def score(sub):
    s = np.full(A.shape, np.nan); s[sub.i0.to_numpy() + 1, sub.symbol.map(col).to_numpy()] = -sub.react.to_numpy()
    return pd.DataFrame(s, index=dates, columns=sig.cols)

groups = {"GOOD SURPRISE: beat and stock +5% or more": E[E.beat & (E.react >= 0.05)],
          "beat, stock up 0 to 5%": E[E.beat & (E.react >= 0) & (E.react < 0.05)],
          "beat, but stock fell": E[E.beat & (E.react < 0)],
          "miss and stock -5% or worse": E[E.miss & (E.react <= -0.05)]}
AO = d.adj_open[sig.cols]; peers = sig.u500
print("\n=== Return over the following days, minus the average liquid stock, before costs (buying at the open 2 days after the report) ===")
out = []
for name, sub in groups.items():
    f = flags(sub)
    for k in (5, 10, 20, 40, 60):
        fwd = AO.shift(-(k + 1)) / AO.shift(-1) - 1
        abn = fwd.sub(fwd.where(peers).mean(axis=1), axis=0).where(f)
        row = {"group": name, "days": k, "reports": len(sub)}
        for label, sl in (("03-14", slice(START, SPLIT)), ("15-26", slice(SPLIT, None)), ("21-26", slice(RECENT, None)), ("All", slice(START, None))):
            b = abn.loc[sl]; row[f"{label} %"] = 100 * b.stack().mean(); row[f"{label} t"] = nw_tstat(b.mean(axis=1).dropna().to_numpy(), lags=5)
        out.append(row)
print(pd.DataFrame(out).set_index(["group", "days"]).round(2).to_string())

print("\n=== Traded: 10 positions, 5 bps per trade ===")
good = groups["GOOD SURPRISE: beat and stock +5% or more"]
rows = {}
for hold in (5, 10, 20, 40, 60):
    daily, tr = simulate(d, flags(good), score(good), None, slots=10, max_hold=hold)
    p = by_period(daily, sig.spy.pct_change(fill_method=None).fillna(0)); t = trade_stats(tr)
    rows[f"hold {hold} days"] = {"03-14": p["2003-2014"]["Return/yr"], "15-26": p["2015-2026"]["Return/yr"], "21-26": p["2021-2026"]["Return/yr"], "All": p["All"]["Return/yr"],
                               "Sharpe": p["All"]["Sharpe"], "drop": p["All"]["Worst drop"], "invested": p["All"]["Invested"], "avg trade %": 100 * t["Avg trade"],
                               "win rate": t["Win rate"], "trades/yr": t["Trades/yr"], "alpha t": p["All"]["Alpha t"]}
print(pd.DataFrame(rows).T.round(3).to_string())
