"""First pass: run every pre-chosen idea once, with standard settings, and report all of them.

Usage: python study.py
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine import RECENT, SPLIT, START, STOCK_DIR, by_period, load, nw_tstat, simulate, trade_stats
from signals import Signals

pd.set_option("display.width", 250)
OUT = STOCK_DIR / "results"
PERIODS = {"2003-2014": (START, SPLIT), "2015-2026": (SPLIT, None), "2021-2026": (RECENT, None), "All": (START, None)}


def pool_returns(sig: Signals, mask: pd.DataFrame) -> pd.Series:
    """Return from simply holding every stock on the liquid list, equally weighted."""
    return sig.ret.where(mask.shift(1, fill_value=False)).mean(axis=1).fillna(0.0)


def event_study(sig: Signals, entry: pd.DataFrame, peers: pd.DataFrame, horizons=(1, 2, 3, 5, 10),
                at_close: bool = False) -> pd.DataFrame:
    """Average return over the next k days after a signal, minus the average for the peer
    group on the same days. Before costs. Buying at the next open unless at_close."""
    px = sig.adj if at_close else sig.d.adj_open[sig.cols]
    lead = 0 if at_close else 1
    rows = []
    for k in horizons:
        fwd = px.shift(-(k + lead)) / px.shift(-lead) - 1
        abn = fwd.sub(fwd.where(peers).mean(axis=1), axis=0).where(entry)
        for name, (a, b) in PERIODS.items():
            sl = slice(a, b)
            block = abn.loc[sl]
            per_day = block.mean(axis=1).dropna()
            rows.append({"period": name, "days held": k, "signals": int(block.notna().sum().sum()),
                         "raw %": 100 * fwd.where(entry).loc[sl].stack().mean(),
                         "vs peers %": 100 * block.stack().mean(),
                         "t": nw_tstat(per_day.to_numpy(), lags=max(k, 1))})
    return pd.DataFrame(rows).pivot(index="days held", columns="period")[["signals", "raw %", "vs peers %", "t"]].round(2)


def run(sig: Signals, spec: dict, bench: pd.Series, **over):
    kw = dict(slots=10)
    kw.update({k: v for k, v in spec.items() if k not in ("entry", "rank", "exit_")})
    kw.update(over)
    daily, trades = simulate(sig.d, spec["entry"], spec["rank"], spec["exit_"], **kw)
    return daily, trades


def summarise(name: str, daily: pd.DataFrame, trades: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    p = by_period(daily, bench)
    t = trade_stats(trades)
    rows = {k: {"Return/yr": v["Return/yr"], "Sharpe": v["Sharpe"], "Worst drop": v["Worst drop"],
                "Invested": v["Invested"], "Alpha vs pool": v["Alpha/yr"], "Alpha t": v["Alpha t"]} for k, v in p.items()}
    df = pd.DataFrame(rows).T.round(3)
    df.insert(0, "idea", name)
    extra = (f"   trades/yr {t.get('Trades/yr', 0):.0f}, win rate {t.get('Win rate', np.nan):.0%}, avg trade {t.get('Avg trade', np.nan):+.2%}, "
             f"avg win {t.get('Avg win', np.nan):+.2%}, avg loss {t.get('Avg loss', np.nan):+.2%}, worst {t.get('Worst', np.nan):+.1%}, avg days {t.get('Avg days', np.nan):.1f}")
    return df, extra


def main() -> None:
    t0 = time.time()
    data = load()
    sig = Signals(data)
    print(f"{len(sig.cols)} stocks, {data.dates[0].date()} to {data.dates[-1].date()}  (loaded in {time.time() - t0:.0f}s)")
    pool200, pool500 = pool_returns(sig, sig.u200), pool_returns(sig, sig.u500)
    spy = sig.spy.pct_change(fill_method=None).fillna(0.0)

    print("\n=== 0. What simply holding the stock list earned (shows how much the list itself flatters results) ===")
    rows = {}
    for name, r in (("SPY buy & hold", spy), ("All top-200 liquid stocks, equal weight", pool200),
                    ("All top-500 liquid stocks, equal weight", pool500)):
        p = by_period(pd.DataFrame({"ret": r[r.index >= START], "invested": 1.0}))
        rows[name] = {k: v["Return/yr"] for k, v in p.items()} | {"Worst drop": p["All"]["Worst drop"], "Sharpe": p["All"]["Sharpe"]}
    print(pd.DataFrame(rows).T.round(3).to_string())

    ideas = {
        "A pullback in uptrend": (sig.pullback(), pool200, sig.u200 & sig.uptrend),
        "C weekly losers": (sig.weekly_losers(), pool200, sig.u200),
        "D gap-up continuation": (sig.gap_up(), pool500, sig.u500),
        "E fresh 52-week high": (sig.new_high(), pool500, sig.u500),
        "M momentum leaders (weekly)": (sig.momentum_leaders(), pool500, None),
        "H stock momentum (2-weekly)": (sig.stock_momentum(), pool200, None),
    }
    print("\n=== 1. Does the signal pick stocks that beat their peers over the next few days? (before costs) ===")
    for name, (spec, _, peers) in ideas.items():
        if peers is None:
            continue
        print(f"\n-- {name}: buy at next open --")
        print(event_study(sig, spec["entry"], peers).to_string())
    print("\n-- A pullback in uptrend: buy at the signal day's close instead --")
    print(event_study(sig, ideas["A pullback in uptrend"][0]["entry"], sig.u200 & sig.uptrend, at_close=True).to_string())

    print("\n=== 2. Traded as a portfolio, after 5 bps per trade, no borrowing, cash earns nothing ===")
    OUT.mkdir(exist_ok=True)
    saved = {}
    for name, (spec, bench, _) in ideas.items():
        over = {"slots": 20} if name.startswith("C") else {}
        daily, trades = run(sig, spec, bench, **over)
        df, extra = summarise(name, daily, trades, bench)
        print("\n" + df.to_string() + "\n" + extra)
        saved[name] = (daily, trades)
    saved["_bench"] = pd.DataFrame({"spy": spy, "pool200": pool200, "pool500": pool500})
    with open(OUT / "first_pass.pkl", "wb") as fh:
        pickle.dump(saved, fh)
    print(f"\n(total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
