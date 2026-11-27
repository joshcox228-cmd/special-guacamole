"""Concentrated stock momentum: average over start days, with and without the added blow-ups."""
import numpy as np, pandas as pd
from engine import load, simulate, by_period, trade_stats, liquid_universe, Data, START
from signals import Signals
pd.set_option("display.width", 250)
full = load()
def without(data, source):
    keep = [c for c in data.adj.columns if not (c in data.pool.index and data.pool.loc[c, "source"] == source)]
    return Data(pool=data.pool, **{f: getattr(data, f)[keep] for f in ("open", "high", "low", "close", "adj", "volume")})
rows = {}
for label, d in (("with blow-ups added", full), ("original list", without(full, "blowup"))):
    sig = Signals(d); spy = sig.spy.pct_change(fill_method=None).fillna(0.0)
    def avg(make, offsets, **kw):
        res = []
        for o in offsets:
            spec = make(o); k = {a: b for a, b in spec.items() if a not in ("entry", "rank", "exit_")}; k.update(kw)
            daily, tr = simulate(d, spec["entry"], spec["rank"], spec["exit_"], **k)
            p = by_period(daily, spy); y = (1 + daily.ret).groupby(daily.index.year).prod() - 1
            res.append({"03-14": p["2003-2014"]["Return/yr"], "15-26": p["2015-2026"]["Return/yr"], "21-26": p["2021-2026"]["Return/yr"], "All": p["All"]["Return/yr"],
                        "drop 03-14": p["2003-2014"]["Worst drop"], "drop 15-26": p["2015-2026"]["Worst drop"], "Sharpe": p["All"]["Sharpe"],
                        "worst year": y.min(), "trades/yr": trade_stats(tr)["Trades/yr"]})
        df = pd.DataFrame(res); out = df.mean().to_dict(); out["15-26 range"] = f"{df['15-26'].min():.3f}..{df['15-26'].max():.3f}"; return out
    rows[(label, "H top 10 of top-200, 12-1 momentum, 2-weekly")] = avg(lambda o: sig.stock_momentum(offset=o), range(0, 10, 2))
    rows[(label, "H with 20 names")] = avg(lambda o: sig.stock_momentum(offset=o, hold_n=20, keep_n=40), range(0, 10, 2))
    rows[(label, "H from top-500")] = avg(lambda o: sig.stock_momentum(top=500, offset=o), range(0, 10, 2))
    rows[(label, "H top 10, only when S&P is in an uptrend")] = avg(lambda o: {**sig.stock_momentum(offset=o), "entry_gate": sig.market_up}, range(0, 10, 2))
    rows[(label, "M 20 steadiest uptrends, weekly")] = avg(lambda o: sig.momentum_leaders(offset=o), range(5))
print(pd.DataFrame(rows).T.round(3).to_string())
print("SPY: 03-14 9.9%, 15-26 12.9%, 21-26 13.2%, worst drop -55% / -34%")
