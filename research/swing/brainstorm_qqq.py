"""Brainstorm check: the same volatility-managed approach applied to the Nasdaq-100 (QQQ), alone and
alongside the managed S&P 500, including the 2000-02 crash."""
import io, contextlib, sys
from pathlib import Path
import numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    import final_combo as fc
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from marketalgo import managed as mg
pd.set_option("display.width", 250)
px = fc.px; bill = px["^IRX"].ffill() / 100
cfg = mg.ManagedConfig()
spy_m = mg.backtest(px["SPY"].dropna(), bill, cfg)["ret"]
qqq_m = mg.backtest(px["QQQ"].dropna(), bill, cfg)["ret"]
qqq_e = mg.backtest(px["QQQ"].dropna(), bill, cfg)["exposure"]
spy, qqq = px["SPY"].pct_change(fill_method=None).dropna(), px["QQQ"].pct_change(fill_method=None).dropna()

def line(r, a, b=None):
    r = r.loc[a:b].dropna(); eq = (1 + r).cumprod()
    return (eq.iloc[-1] ** (252 / len(r)) - 1, (eq / eq.cummax() - 1).min())
def row(r):
    out = {}
    for label, a, b in (("2000-2002", "2000-03-01", "2002-12-31"), ("2003-2014", "2003-03-01", "2014-12-31"), ("2015-2026", "2015-01-01", None),
                        ("since 2003", "2003-03-01", None), ("since 2000", "2000-03-01", None)):
        try:
            g, dd = line(r, a, b); out[label] = g
            if label.startswith("since"): out[f"drop {label}"] = dd
        except Exception:
            out[label] = np.nan
    return out
mix = lambda *parts: sum(w * r for w, r in parts).dropna()
rule2 = fc.rule2
table = {
    "S&P 500, held": row(spy), "Nasdaq-100 (QQQ), held": row(qqq),
    "managed S&P": row(spy_m), "managed Nasdaq": row(qqq_m),
    "CURRENT: half managed S&P + half fund rule": row(mix((0.5, spy_m), (0.5, rule2))),
    "IDEA: half managed S&P + half managed Nasdaq": row(mix((0.5, spy_m), (0.5, qqq_m))),
    "IDEA: thirds (managed S&P, managed Nasdaq, fund rule)": row(mix((1/3, spy_m), (1/3, qqq_m), (1/3, rule2))),
    "IDEA: half managed Nasdaq + half fund rule": row(mix((0.5, qqq_m), (0.5, rule2))),
}
print(pd.DataFrame(table).T.round(3).to_string())
print("\nmanaged Nasdaq: average exposure", round(float(qqq_e.loc['2000-03-01':].mean()), 2), "x; correlation of daily returns, managed S&P vs managed Nasdaq:",
      round(float(pd.concat([spy_m, qqq_m], axis=1).loc['2003':].corr().iloc[0, 1]), 2), "| managed S&P vs fund rule:",
      round(float(pd.concat([spy_m, rule2], axis=1).loc['2003':].corr().iloc[0, 1]), 2))
yr = lambda r: ((1 + r.loc['2000-03-01':]).groupby(r.loc['2000-03-01':].index.year).prod() - 1)
print(pd.DataFrame({"managed Nasdaq": yr(qqq_m), "managed S&P": yr(spy_m), "QQQ held": yr(qqq)}).round(2).T.to_string())
