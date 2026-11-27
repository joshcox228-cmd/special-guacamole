"""Before calling short-term stock trading dead: try it on other stock lists, stricter settings, both fill times."""
import numpy as np, pandas as pd
from engine import load, simulate, by_period, trade_stats, liquid_universe, sma, START, SPLIT
from signals import Signals
from study import pool_returns
pd.set_option("display.width", 250)
d = load(); sig = Signals(d)
spy = sig.spy.pct_change(fill_method=None).fillna(0.0)
mid = liquid_universe(d, 800, skip_top=200)[sig.cols]            # ranks 201-800: smaller, jumpier stocks
lists = {"top 200": sig.u200, "top 500": sig.u500, "ranks 201-800": mid}
rows = {}
for lname, U in lists.items():
    pool = pool_returns(sig, U)
    for below in (5, 10):
        for mode in ("next_open", "same_close"):
            cost = 5 if lname == "top 200" else 10
            entry = U & sig.uptrend & (sig.rsi2 < below)
            daily, tr = simulate(d, entry, sig.rsi2, sig.adj > sig.sma5, slots=10, max_hold=10, mode=mode, cost_bps=cost)
            p = by_period(daily, pool); t = trade_stats(tr)
            rows[(lname, f"RSI<{below}", "next open" if mode == "next_open" else "at close")] = {
                "03-14": p["2003-2014"]["Return/yr"], "15-26": p["2015-2026"]["Return/yr"], "21-26": p["2021-2026"]["Return/yr"],
                "alpha 03-14": p["2003-2014"]["Alpha/yr"], "alpha 15-26": p["2015-2026"]["Alpha/yr"], "t 15-26": p["2015-2026"]["Alpha t"],
                "drop": p["All"]["Worst drop"], "avg trade %": 100 * t["Avg trade"], "trades/yr": t["Trades/yr"]}
print("=== Pullback (dip-buying) in stocks: every list, two strictness levels, two fill times ===")
print(pd.DataFrame(rows).T.round(3).to_string())
print("SPY for reference: 03-14 9.9%, 15-26 12.9%")

print("\n=== Gap-up continuation: variants, and year by year ===")
rows = {}
for (gap, day, vol, hold) in ((0.04, 0.05, 3.0, 10), (0.04, 0.05, 3.0, 5), (0.04, 0.05, 3.0, 20), (0.03, 0.04, 2.0, 10), (0.06, 0.08, 3.0, 10), (0.04, 0.05, 2.0, 10)):
    spec = sig.gap_up(gap=gap, day=day, vol=vol, hold=hold)
    daily, tr = simulate(d, spec["entry"], spec["rank"], None, slots=10, max_hold=hold, cost_bps=10)
    p = by_period(daily, pool_returns(sig, sig.u500)); t = trade_stats(tr)
    rows[f"gap {gap:.0%}, day {day:.0%}, volume {vol:g}x, hold {hold}d"] = {
        "03-14": p["2003-2014"]["Return/yr"], "15-26": p["2015-2026"]["Return/yr"], "21-26": p["2021-2026"]["Return/yr"],
        "alpha 15-26": p["2015-2026"]["Alpha/yr"], "t 15-26": p["2015-2026"]["Alpha t"], "drop": p["All"]["Worst drop"],
        "avg trade %": 100 * t["Avg trade"], "trades/yr": t["Trades/yr"], "invested": p["All"]["Invested"]}
print(pd.DataFrame(rows).T.round(3).to_string())
spec = sig.gap_up(); daily, tr = simulate(d, spec["entry"], spec["rank"], None, slots=10, max_hold=10, cost_bps=10)
by = tr.groupby(tr.entry.dt.year).ret.agg(["count", "mean"]); by["mean"] = (100 * by["mean"]).round(2)
print("average trade % by year:", by["mean"].to_dict())
