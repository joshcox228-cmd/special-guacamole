"""Final comparison after the survivorship and earnings checks (2003-2026, one consistent basis)."""
import io, contextlib
import numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    import final_combo as fc
pd.set_option("display.width", 250)
core, _ = fc.managed(lag=2); boost, eb = fc.managed(lag=2, dip_boost=0.5)
mix = lambda a, b: (0.5 * a + 0.5 * b).dropna()
rows = {
    "S&P 500, plain": fc.stats(fc.spy),
    "TODAY: half plain S&P + half fund rule (2x, margin)": fc.stats(mix(fc.spy, fc.rule2)),
    "UPGRADE: half managed S&P with dip boost + half fund rule (2x, margin)": fc.stats(mix(boost, fc.rule2)),
    "   same, fund rule without borrowing": fc.stats(mix(boost, fc.rule1)),
    "ALL-IN: managed S&P with dip boost": fc.stats(boost),
    "   managed S&P, no dip boost": fc.stats(core),
}
t = pd.DataFrame(rows).T; print(t.round(3).to_string())
yr = lambda r: ((1 + r[r.index >= fc.START]).groupby(r[r.index >= fc.START].index.year).prod() - 1)
y = pd.DataFrame({"Upgrade": yr(mix(boost, fc.rule2)), "Today": yr(mix(fc.spy, fc.rule2)), "All-in": yr(boost), "S&P": yr(fc.spy)}).round(3)
print("\n" + y.T.to_string())
print("years Upgrade beat Today:", int((y.Upgrade > y.Today).sum()), "of", len(y), "| beat S&P:", int((y.Upgrade > y["S&P"]).sum()), "| All-in beat S&P:", int((y["All-in"] > y["S&P"]).sum()))
# rolling 3-year comparison: how often is the upgrade ahead of today's mix?
up, td = mix(boost, fc.rule2), mix(fc.spy, fc.rule2)
up, td = up[up.index >= fc.START], td[td.index >= fc.START]
roll = lambda r: (1 + r).rolling(756).apply(np.prod, raw=True) - 1
print("share of 3-year windows where Upgrade beats Today:", round(float((roll(up) > roll(td)).dropna().mean()), 2), "| All-in beats Today:", round(float((roll(boost[boost.index >= fc.START]) > roll(td)).dropna().mean()), 2))
e = eb[eb.index >= fc.START]; d = e.diff().abs()
print(f"exposure changes per year: {(d > 0).sum() / (len(e) / 252):.0f}; of which dip-boost switches: about {((d - 0.5).abs() < 1e-9).sum() / (len(e) / 252):.0f}; median days between changes: {np.median(np.diff(np.flatnonzero((d > 0).to_numpy()))):.0f}")
