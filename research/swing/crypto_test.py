"""Would a slice of Bitcoin have helped the plan? Uses the same volatility-managed rule, capped at 1x
(no borrowing), on Bitcoin's daily price. Short and unusual history: read with care."""
import io, contextlib, sys
from pathlib import Path
import numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    import final_combo as fc
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from marketalgo import managed as mg
from marketalgo.data import fetch_history
pd.set_option("display.width", 250)
px = fc.px; bill = px["^IRX"].ffill() / 100
days = px["SPY"].dropna().index
coins = {}
for sym in ("BTC-USD", "ETH-USD"):
    h = fetch_history(sym, "2014-01-01")["close"]
    h.index = h.index + pd.Timedelta(days=1)      # Yahoo stamps a coin's day at midnight UTC, which reads as the evening before in New York
    coins[sym] = h.reindex(h.index.union(days)).ffill().reindex(days).dropna()      # price at each stock-market close
    print(sym, "from", coins[sym].index[0].date(), "last", round(float(coins[sym].iloc[-1])), "| swings (all history)", f"{coins[sym].pct_change().std() * np.sqrt(252):.0%} a year")
cfg = mg.ManagedConfig()
spy_m = mg.backtest(px["SPY"].dropna(), bill, cfg)["ret"]; qqq_m = mg.backtest(px["QQQ"].dropna(), bill, cfg)["ret"]
btc_cfg = mg.ManagedConfig(index="BTC", fund2x="-", cap=1.0, dip_boost=0.0)
btc_run = mg.backtest(coins["BTC-USD"], bill, btc_cfg); btc_m, btc_e = btc_run["ret"], btc_run["exposure"]
eth_m = mg.backtest(coins["ETH-USD"], bill, btc_cfg)["ret"]
btc, eth = coins["BTC-USD"].pct_change(), coins["ETH-USD"].pct_change()
plan = (spy_m + qqq_m + fc.rule2).dropna() / 3

def stats(r, a, b=None):
    r = r.loc[a:b].dropna()
    if len(r) < 200: return (np.nan, np.nan)
    eq = (1 + r).cumprod(); return (eq.iloc[-1] ** (252 / len(r)) - 1, (eq / eq.cummax() - 1).min())
def row(r):
    out = {}
    for label, a, b in (("2015-2020", "2015-01-01", "2020-12-31"), ("2021-2026", "2021-01-01", None), ("since 2015", "2015-01-01", None), ("last 3 yrs", "2023-11-27", None)):
        g, dd = stats(r, a, b); out[label] = g; out[f"drop {label}"] = dd
    return out
mix = lambda parts: sum(w * r for w, r in parts).dropna()
table = {
    "Bitcoin, simply held": row(btc), "Ethereum, simply held": row(eth),
    "Bitcoin, managed (hold less when wild, max 1x)": row(btc_m),
    "S&P 500, held": row(px["SPY"].pct_change(fill_method=None)),
    "YOUR PLAN (three-way)": row(plan),
    "plan 95% + Bitcoin 5%, simply held": row(mix([(0.95, plan), (0.05, btc)])),
    "plan 90% + Bitcoin 10%, simply held": row(mix([(0.90, plan), (0.10, btc)])),
    "plan 90% + managed Bitcoin 10%": row(mix([(0.90, plan), (0.10, btc_m)])),
    "four equal parts, managed Bitcoin as the fourth": row(mix([(0.75, plan), (0.25, btc_m)])),
}
print(pd.DataFrame(table).T.round(3).to_string())
e = btc_e.loc["2015":]
print(f"\nmanaged Bitcoin: average holding {e.mean():.0%} of its slice (range {e.min():.0%} to {e.max():.0%}); now {btc_e.iloc[-1]:.0%}")
j = pd.concat([btc, px['SPY'].pct_change(fill_method=None), px['QQQ'].pct_change(fill_method=None)], axis=1, keys=["btc", "spy", "qqq"]).dropna()
print("how closely Bitcoin moves with the S&P 500 (0 = unrelated, 1 = identical): 2015-2019", round(float(j.loc['2015':'2019'].corr().loc['btc', 'spy']), 2),
      "| 2020-2026", round(float(j.loc['2020':].corr().loc['btc', 'spy']), 2), "| with the Nasdaq 2020-2026", round(float(j.loc['2020':].corr().loc['btc', 'qqq']), 2))
yr = lambda r: ((1 + r.loc['2015':].dropna()).groupby(r.loc['2015':].dropna().index.year).prod() - 1)
print(pd.DataFrame({"Bitcoin": yr(btc), "S&P 500": yr(px['SPY'].pct_change(fill_method=None)), "your plan": yr(plan)}).round(2).T.to_string())
worst = (coins["BTC-USD"] / coins["BTC-USD"].cummax() - 1)
print("Bitcoin's deepest fall from its peak, by year:", {int(y): round(float(v), 2) for y, v in worst.groupby(worst.index.year).min().items() if v < -0.5})
