"""Solana versus Bitcoin as the crypto slice: returns, drops, and what a 10% slice would have done to the plan."""
import io, contextlib, sys
from pathlib import Path
import numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    import final_combo as fc
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from marketalgo import managed as mg
from marketalgo.data import fetch_history
pd.set_option("display.width", 250)
px = fc.px; bill = px["^IRX"].ffill() / 100; days = px["SPY"].dropna().index
def coin(sym):
    h = fetch_history(sym, "2014-01-01")["close"]; h.index = h.index + pd.Timedelta(days=1)
    return h.reindex(h.index.union(days)).ffill().reindex(days).dropna()
P = {k: coin(k) for k in ("BTC-USD", "ETH-USD", "SOL-USD")}
R = {k: v.pct_change() for k, v in P.items()}
for k, v in P.items():
    print(k, "history from", v.index[0].date(), "| price now", round(float(v.iloc[-1]), 2), "| peak", round(float(v.max()), 2), "| now vs peak", f"{v.iloc[-1] / v.max() - 1:.0%}")
cfg = mg.ManagedConfig()
plan = (mg.backtest(px["SPY"].dropna(), bill, cfg)["ret"] + mg.backtest(px["QQQ"].dropna(), bill, cfg)["ret"] + fc.rule2).dropna() / 3
spy = px["SPY"].pct_change(fill_method=None)
def st(r, a, b=None):
    r = r.loc[a:b].dropna()
    if len(r) < 150: return "n/a"
    eq = (1 + r).cumprod(); return f"{eq.iloc[-1] ** (252 / len(r)) - 1:+.0%} a yr, drop {(eq / eq.cummax() - 1).min():.0%}"
wins = (("2021 only", "2021-01-01", "2021-12-31"), ("2022 only", "2022-01-01", "2022-12-31"), ("since 2022", "2022-01-01", None), ("last 3 years", "2023-11-27", None), ("last 12 months", "2025-11-27", None))
rows = {"Bitcoin": R["BTC-USD"], "Ethereum": R["ETH-USD"], "Solana": R["SOL-USD"], "S&P 500": spy, "your plan, no crypto": plan,
        "plan with 10% Bitcoin": 0.9 * plan + 0.1 * R["BTC-USD"], "plan with 10% Solana": 0.9 * plan + 0.1 * R["SOL-USD"],
        "plan with 5% Bitcoin + 5% Solana": 0.9 * plan + 0.05 * R["BTC-USD"] + 0.05 * R["SOL-USD"]}
print(pd.DataFrame({name: {w: st(r.dropna(), a, b) for w, a, b in wins} for name, r in rows.items()}).T.to_string())
j = pd.concat(R, axis=1).loc["2021":].dropna()
print("\nswings a year since 2021:", {k: f"{v.std() * np.sqrt(252):.0%}" for k, v in j.items()}, "| S&P 500:", f"{spy.loc['2021':].std() * np.sqrt(252):.0%}")
print("moves together with Bitcoin (1 = identical): Solana", round(float(j.corr().loc["SOL-USD", "BTC-USD"]), 2), "| Ethereum", round(float(j.corr().loc["ETH-USD", "BTC-USD"]), 2))
yr = lambda r: ((1 + r.loc["2021":].dropna()).groupby(r.loc["2021":].dropna().index.year).prod() - 1)
print(pd.DataFrame({"Bitcoin": yr(R["BTC-USD"]), "Solana": yr(R["SOL-USD"]), "S&P 500": yr(spy)}).round(2).T.to_string())
for t in ("BSOL", "GSOL", "SOLZ", "SSK", "IBIT"):
    try:
        h = fetch_history(t, "2024-01-01"); print(t, "fund trades since", h.index[0].date(), "last", round(float(h["close"].iloc[-1]), 2))
    except Exception as e:
        print(t, "not found")
