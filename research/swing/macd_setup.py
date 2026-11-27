"""Test of a user-specified setup over the last five years, defined before looking at any result:

  BUY when, at the close:   the MACD line (12/26) crosses above its signal line (9),
                            price is above its 50-day average,
                            price is below its 200-day average but within 5% of it ("approaching"),
                            and the 14-day RSI is under 70 (not overbought).
  Stock list: the 500 most traded US stocks at the time (delisted companies included).
  Fill at the next open, 10 positions of 10% each, 5 bps per trade.

No exit was specified, so the common ones are all shown.
"""
import numpy as np, pandas as pd
from engine import Data, load, simulate, trade_stats, nw_tstat, rsi, sma, TRADING_DAYS
from signals import Signals
pd.set_option("display.width", 250)
full = load()
FIELDS = ("open", "high", "low", "close", "adj", "volume")
d = Data(pool=full.pool, **{f: getattr(full, f).loc["2019-09-01":] for f in FIELDS})     # enough run-up for the averages
sig = Signals(d); adj = sig.adj
BEGIN = d.dates[-1] - pd.DateOffset(years=5)
print(f"test window: {BEGIN.date()} to {d.dates[-1].date()}")

macd = adj.ewm(span=12, adjust=False).mean() - adj.ewm(span=26, adjust=False).mean()
line = macd.ewm(span=9, adjust=False).mean()
cross_up = (macd > line) & (macd.shift(1) <= line.shift(1)) & adj.notna()
sma50, sma200, rsi14 = sma(adj, 50), sig.sma200, rsi(adj, 14)

def setup(band=0.05, rsi_max=70, side="below"):
    near = (adj < sma200) & (adj >= (1 - band) * sma200) if side == "below" else (adj / sma200 - 1).abs() <= band
    return sig.u500 & cross_up & (adj > sma50) & near & (rsi14 < rsi_max)

def perf(r):
    eq = (1 + r).cumprod(); yrs = len(r) / TRADING_DAYS
    return eq.iloc[-1] ** (1 / yrs) - 1, (eq / eq.cummax() - 1).min()

spy = sig.spy.pct_change(fill_method=None).fillna(0.0).loc[BEGIN:]
pool = sig.ret.where(sig.u500.shift(1, fill_value=False)).mean(axis=1).fillna(0.0).loc[BEGIN:]
(spy_ret, spy_dd), (pool_ret, pool_dd) = perf(spy), perf(pool)
year = lambda r: ((1 + r).groupby(r.index.year).prod() - 1)

entry = setup()
n = int(entry.loc[BEGIN:].sum().sum())
print(f"signals in the window: {n} (about {n / 5:.0f} a year, {n / 5 / 52:.1f} a week)")

print("\n=== After a signal: average return over the next days, and versus the average of all 500 stocks (before costs) ===")
AO = d.adj_open[sig.cols]; rows = []
for k in (1, 3, 5, 10, 20, 40):
    fwd = AO.shift(-(k + 1)) / AO.shift(-1) - 1                        # buy next open, sell k days later
    abn = fwd.sub(fwd.where(sig.u500).mean(axis=1), axis=0).where(entry).loc[BEGIN:]
    raw = fwd.where(entry).loc[BEGIN:].stack().dropna()
    rows.append({"days held": k, "signals": len(raw), "avg return %": 100 * raw.mean(), "share that rose": (raw > 0).mean(),
                 "vs the other stocks %": 100 * abn.stack().dropna().mean(), "t-stat": nw_tstat(abn.mean(axis=1).dropna().to_numpy(), lags=k)})
print(pd.DataFrame(rows).set_index("days held").round(2).to_string())

rank = (sma200 / adj - 1)                                              # closest to the 200-day average first
def row(daily, tr):
    r = daily["ret"]; ret, dd = perf(r); t = trade_stats(tr)
    return {"return/yr": ret, "worst drop": dd, "invested": daily["invested"].mean(), "win rate": t.get("Win rate", np.nan),
            "avg trade %": 100 * t.get("Avg trade", np.nan), "avg win %": 100 * t.get("Avg win", np.nan), "avg loss %": 100 * t.get("Avg loss", np.nan),
            "worst trade %": 100 * t.get("Worst", np.nan), "avg days": t.get("Avg days", np.nan), "trades/yr": len(tr) / 5}
exits = {
    "sell when MACD crosses back down": dict(exit_=macd < line, max_hold=None),
    "sell on a close below the 50-day average": dict(exit_=adj < sma50, max_hold=None),
    "sell on either of those": dict(exit_=(macd < line) | (adj < sma50), max_hold=None),
    "sell after 5 days": dict(exit_=None, max_hold=5), "sell after 10 days": dict(exit_=None, max_hold=10),
    "sell after 20 days": dict(exit_=None, max_hold=20), "sell after 40 days": dict(exit_=None, max_hold=40),
}
print("\n=== Traded as a portfolio for five years (10 positions, costs included, idle cash earns nothing) ===")
rows, yearly = {}, {}
for name, kw in exits.items():
    daily, tr = simulate(d, entry, rank, slots=10, start=BEGIN, **kw)
    rows[name] = row(daily, tr); yearly[name] = year(daily["ret"])
rows["S&P 500 (SPY), held"] = {"return/yr": spy_ret, "worst drop": spy_dd, "invested": 1.0}
rows["all 500 stocks, equal amounts, held"] = {"return/yr": pool_ret, "worst drop": pool_dd, "invested": 1.0}
print(pd.DataFrame(rows).T.round(3).to_string())
yearly["S&P 500 (SPY)"] = year(spy)
print("\nyear by year:"); print(pd.DataFrame(yearly).T.round(3).to_string())

print("\n=== Small changes to the rules (exit: MACD cross down or close below the 50-day) ===")
ex = dict(exit_=(macd < line) | (adj < sma50), max_hold=None)
rows = {}
for label, e in (("as specified (within 5% below the 200-day)", entry), ("within 3% below", setup(band=0.03)), ("within 10% below", setup(band=0.10)),
                 ("within 5% either side of the 200-day", setup(band=0.05, side="either")), ("RSI under 60", setup(rsi_max=60)),
                 ("no 200-day condition", sig.u500 & cross_up & (adj > sma50) & (rsi14 < 70)),
                 ("only when the S&P 500 is above its 200-day average", entry & sig.market_up.to_numpy()[:, None])):
    rows[label] = row(*simulate(d, e, rank, slots=10, start=BEGIN, **ex))
rnd = [row(*simulate(d, entry, rank, slots=10, start=BEGIN, seed_choice=np.random.default_rng(k), **ex)) for k in range(5)]
rows["as specified, random pick when signals exceed slots (avg of 5)"] = pd.DataFrame(rnd).mean().to_dict()
print(pd.DataFrame(rows).T.round(3).to_string())
