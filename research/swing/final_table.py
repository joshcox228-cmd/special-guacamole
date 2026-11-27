"""The final side-by-side: today's portfolio against the candidates, all on the same 2003-2026 basis."""
import sys
import numpy as np, pandas as pd
import final_combo as fc
from engine import load, simulate
from signals import Signals
pd.set_option("display.width", 250)

d = load(); sig = Signals(d)
runs = []
for o in range(5):
    spec = sig.momentum_leaders(offset=o)
    kw = {k: v for k, v in spec.items() if k not in ("entry", "rank", "exit_")}
    runs.append(simulate(d, spec["entry"], spec["rank"], spec["exit_"], **kw)[0]["ret"])
leaders = pd.concat(runs, axis=1).mean(axis=1)

core, e = fc.managed(lag=2)                       # check in the evening, trade the next day
boost, eb = fc.managed(lag=2, dip_boost=0.5)
mix = lambda a, b: (0.5 * a + 0.5 * b).dropna()
rows = {
    "S&P 500, plain": fc.stats(fc.spy),
    "TODAY: half S&P + half fund rule at 2x on margin": fc.stats(mix(fc.spy, fc.rule2)),
    "A  managed S&P (SPY + SSO)": fc.stats(core),
    "A+ managed S&P with dip-buying boost": fc.stats(boost),
    "B  half managed S&P + half fund rule (no margin)": fc.stats(mix(core, fc.rule1)),
    "C  half managed S&P + half stock leaders": fc.stats(mix(core, leaders)),
    "   stock leaders alone (data flatters this)": fc.stats(leaders),
}
print(pd.DataFrame(rows).T.round(3).to_string())
eb2 = eb[eb.index >= fc.START]
print(f"\nA+: average exposure {eb2.mean():.2f}x; changes per year {(eb2.diff().abs() > 0).sum() / (len(eb2) / 252):.0f}; at 2x on {(eb2 >= 1.99).mean():.0%} of days; below 1x on {(eb2 < 1).mean():.0%}")
yr = lambda r: ((1 + r[r.index >= fc.START]).groupby(r[r.index >= fc.START].index.year).prod() - 1)
t = pd.DataFrame({"A+": yr(boost), "Today": yr(mix(fc.spy, fc.rule2)), "S&P": yr(fc.spy)}).round(3)
print(t.T.to_string()); print("years A+ beat the S&P:", int((t["A+"] > t["S&P"]).sum()), "of", len(t), "| beat today's mix:", int((t["A+"] > t["Today"]).sum()))
# from 2000, to include the dot-com bear market (the fund rule cannot be tested that far back)
r0 = lambda r: r.loc["2000-03-01":]
cg = lambda r: (1 + r0(r)).prod() ** (252 / len(r0(r))) - 1
ddn = lambda r: ((1 + r0(r)).cumprod() / (1 + r0(r)).cumprod().cummax() - 1).min()
print(f"\nfrom 2000 (includes 2000-02 and 2008): S&P {cg(fc.spy):.1%} drop {ddn(fc.spy):.0%} | A {cg(core):.1%} drop {ddn(core):.0%} | A+ {cg(boost):.1%} drop {ddn(boost):.0%}")
sigma = fc.spy.rolling(20).std() * np.sqrt(252)
p = fc.px["SPY"].dropna()
print(f"\nNOW ({fc.spy.index[-1].date()}): 20-day swing {sigma.iloc[-1]:.1%}; target exposure {min(2, 0.2 / sigma.iloc[-1]):.2f}x; SPY {p.iloc[-1]:.2f} vs 200-day avg {p.rolling(200).mean().iloc[-1]:.2f}; last 6 exposure readings {[round(float(x), 2) for x in (0.2 / sigma).clip(upper=2).iloc[-6:]]}")
print("SSO last price:", round(float(fc.px['SSO'].dropna().iloc[-1]), 2), "| SHY/cash; SPY last", round(float(p.iloc[-1]), 2))
