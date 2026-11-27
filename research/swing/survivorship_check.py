"""The decisive check: rerun the stock ideas with delisted and dropped-from-index companies included."""
import numpy as np, pandas as pd
from engine import load, simulate, by_period, trade_stats
from signals import Signals
pd.set_option("display.width", 260)

def evaluate(d, label, haircut=0.0):
    sig = Signals(d); spy = sig.spy.pct_change(fill_method=None).fillna(0.0)
    def avg(make, offsets, **kw):
        res, ends = [], 0
        for o in offsets:
            spec = make(o); k = {a: b for a, b in spec.items() if a not in ("entry", "rank", "exit_")}; k.update(kw)
            daily, tr = simulate(d, spec["entry"], spec["rank"], spec["exit_"], delist_haircut=haircut, **k)
            p = by_period(daily, spy); t = trade_stats(tr)
            res.append({"03-14": p["2003-2014"]["Return/yr"], "15-26": p["2015-2026"]["Return/yr"], "21-26": p["2021-2026"]["Return/yr"], "All": p["All"]["Return/yr"],
                        "drop 03-14": p["2003-2014"]["Worst drop"], "drop 15-26": p["2015-2026"]["Worst drop"], "Sharpe": p["All"]["Sharpe"],
                        "worst trade": t["Worst"], "trades/yr": t["Trades/yr"]})
        return pd.DataFrame(res).mean().to_dict()
    rows = {
        (label, "A dip-buying in uptrending stocks"): avg(lambda o: sig.pullback(), [0]),
        (label, "E fresh 52-week highs"): avg(lambda o: sig.new_high(), [0]),
        (label, "M 20 steadiest uptrends, weekly"): avg(lambda o: sig.momentum_leaders(offset=o), range(5)),
        (label, "M with 10 names"): avg(lambda o: sig.momentum_leaders(offset=o, hold_n=10, keep_n=50), (0, 2, 4)),
        (label, "H 10 strongest of top-200, 2-weekly"): avg(lambda o: sig.stock_momentum(offset=o), range(0, 10, 2)),
        (label, "H 20 strongest"): avg(lambda o: sig.stock_momentum(offset=o, hold_n=20, keep_n=40), (0, 4, 8)),
    }
    n_in = int(sig.u500.sum(axis=1).mean())
    by_src = sig.u500.loc["2015":].sum().groupby(d.pool["source"].reindex(sig.cols).str.split("+").str[0].to_numpy()).sum()
    print(f"[{label}] stocks: {len(sig.cols)}; share of top-500 slots since 2015 by origin: {(by_src / by_src.sum()).round(3).to_dict()}", flush=True)
    return rows

import sys
from engine import STOCK_DIR
which = sys.argv[1]
if which == "survivors":
    rows = evaluate(load(delisted=False), "survivors only")
elif which == "full":
    rows = evaluate(load(delisted=True), "with delisted added")
else:
    rows = evaluate(load(delisted=True), "with delisted, failures cut a further 50%", haircut=0.5)
out = pd.DataFrame(rows).T.round(4)
out.to_csv(STOCK_DIR / "results" / f"survivorship_{which}.csv")
print(out.round(3).to_string())
