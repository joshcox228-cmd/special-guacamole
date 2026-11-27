"""Stress the 'momentum leaders' idea: is it one lucky configuration, or a family that holds up?"""
import pickle
import numpy as np, pandas as pd
from engine import load, simulate, by_period, trade_stats, liquid_universe, sma, START, SPLIT, STOCK_DIR
from signals import Signals
from study import pool_returns
pd.set_option("display.width", 250)
d = load(); sig = Signals(d)
pool500 = pool_returns(sig, sig.u500); spy = sig.spy.pct_change(fill_method=None).fillna(0.0)

def line(daily, trades, bench=pool500):
    p = by_period(daily, bench); t = trade_stats(trades)
    return {"03-14": p["2003-2014"]["Return/yr"], "15-26": p["2015-2026"]["Return/yr"], "21-26": p["2021-2026"]["Return/yr"], "All": p["All"]["Return/yr"],
            "Sharpe": p["All"]["Sharpe"], "Sh 15-26": p["2015-2026"]["Sharpe"], "Worst drop": p["All"]["Worst drop"],
            "Alpha 03-14": p["2003-2014"]["Alpha/yr"], "Alpha 15-26": p["2015-2026"]["Alpha/yr"], "Alpha t": p["All"]["Alpha t"],
            "Trades/yr": t.get("Trades/yr", 0), "Invested": p["All"]["Invested"]}

def run(spec, **over):
    kw = {k: v for k, v in spec.items() if k not in ("entry", "rank", "exit_")}; kw.update(over)
    return simulate(d, spec["entry"], spec["rank"], spec["exit_"], **kw)

rows = {}
base = sig.momentum_leaders()
daily0, trades0 = run(base); rows["base: top-500, 20 names, weekly"] = line(daily0, trades0)
for o in range(1, 5):
    rows[f"review day shifted {o}"] = line(*run(sig.momentum_leaders(offset=o)))
rows["10 bps cost"] = line(*run(base, cost_bps=10))
rows["trade at the close instead"] = line(*run(base, mode="same_close"))
rows["10 names instead of 20"] = line(*run(sig.momentum_leaders(hold_n=10, keep_n=50)))
rows["30 names"] = line(*run(sig.momentum_leaders(hold_n=30, keep_n=150)))
rows["top-200 list only"] = line(*run(sig.momentum_leaders(top=200, keep_n=40)))
rows["no market switch"] = line(*run({**base, "entry_gate": None}))

# only companies in today's S&P 500/400/600 (drop every hand-added name)
index_only = [c for c in sig.cols if d.pool.loc[c, "source"] != "extra"]
mask = pd.DataFrame(False, index=sig.adj.index, columns=sig.cols); mask[index_only] = True
spec = sig.momentum_leaders()
calm = sig.ret.abs().rolling(90).max() < 0.15
ok = sig.u500 & mask & (sig.adj > sig.sma100) & calm
rk = sig.trend_score.where(ok).rank(axis=1, ascending=False); review = sig.every(5).to_numpy()[:, None]
rows["without my hand-added names"] = line(*run(dict(entry=(rk <= 20) & review, rank=rk, exit_=~(rk <= 100) & review, max_hold=None, entry_gate=sig.market_up, slots=20)))

# other ways of measuring 'strong stock' (same structure, different yardstick)
def with_score(score, name):
    okk = sig.u500 & (sig.adj > sig.sma100) & calm
    r = score.where(okk).rank(axis=1, ascending=False)
    rows[name] = line(*run(dict(entry=(r <= 20) & review, rank=r, exit_=~(r <= 100) & review, max_hold=None, entry_gate=sig.market_up, slots=20)))
with_score(sig.adj / sig.adj.shift(126) - 1, "yardstick: 6-month return")
with_score(sig.adj.shift(21) / sig.adj.shift(252) - 1, "yardstick: 12-month return skipping last month")
with_score(sig.adj / sig.adj.shift(63) - 1, "yardstick: 3-month return")
rng = np.random.default_rng(0)
for k in range(3):
    with_score(pd.DataFrame(rng.random(sig.adj.shape), index=sig.adj.index, columns=sig.cols), f"yardstick: random numbers #{k + 1}")
print(pd.DataFrame(rows).T.round(3).to_string())

# year by year
yr = lambda r: (1 + r).groupby(r.index.year).prod() - 1
tab = pd.DataFrame({"Leaders": yr(daily0.ret), "SPY": yr(spy[spy.index >= START]), "Pool": yr(pool500[pool500.index >= START])}).round(3)
tab["Beat SPY"] = tab["Leaders"] > tab["SPY"]
print("\n" + tab.T.to_string()); print("years beating SPY:", int(tab["Beat SPY"].sum()), "of", len(tab))

# who made the money?
tr = trades0.copy(); tr["src"] = tr.ticker.map(d.pool["source"])
print("\nshare of trades / of summed trade returns by where the stock came from:")
print(pd.DataFrame({"trades": tr.groupby(tr.src.str.split("+").str[0]).size(), "sum of returns": tr.groupby(tr.src.str.split("+").str[0]).ret.sum()}).round(1).to_string())
top = tr.groupby("ticker").ret.agg(["sum", "count"]).sort_values("sum", ascending=False)
print("\nbiggest contributors:", top.head(15).round(2).to_dict()["sum"]); print("worst:", top.tail(8).round(2).to_dict()["sum"])
print("share of total from top 10 tickers:", round(top["sum"].head(10).sum() / top["sum"].sum(), 2), " distinct tickers traded:", len(top))
pickle.dump({"daily": daily0, "trades": trades0}, open(STOCK_DIR / "results" / "leaders_base.pkl", "wb"))
