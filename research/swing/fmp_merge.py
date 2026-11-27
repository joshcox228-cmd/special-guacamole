"""Decide which FMP-recovered companies to add, dropping any that duplicate a stock already in the data
(for example an old ticker of a company that was simply renamed), and report how complete the result is."""
import numpy as np, pandas as pd
from engine import STOCK_DIR

yahoo = pd.read_parquet(STOCK_DIR / "adj.parquet").astype("float64")
extra = pd.read_parquet(STOCK_DIR / "fmp_adj.parquet").astype("float64").reindex(yahoo.index)
meta = pd.read_csv(STOCK_DIR / "fmp_added.csv", index_col=0, parse_dates=["ref"])
ry, rx = yahoo.pct_change(fill_method=None), extra.pct_change(fill_method=None)

def correlations(a: pd.DataFrame, b: pd.DataFrame, min_overlap: int = 250) -> np.ndarray:
    """Correlation of every column of `a` with every column of `b`, each pair measured over the
    days both have data. One matrix product per term instead of a loop over pairs."""
    A, B = a.to_numpy(dtype="float32"), b.to_numpy(dtype="float32")
    MA, MB = np.isfinite(A), np.isfinite(B)
    A, B = np.where(MA, A, 0), np.where(MB, B, 0)
    MAf, MBf = MA.astype("float32"), MB.astype("float32")
    n = MAf.T @ MBf
    sx, sy = A.T @ MBf, MAf.T @ B
    sxx, syy, sxy = (A * A).T @ MBf, MAf.T @ (B * B), A.T @ B
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (n * sxy - sx * sy) / np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
    return np.where(n >= min_overlap, np.nan_to_num(corr), 0.0)


c = correlations(rx.clip(-0.5, 0.5), ry.clip(-0.5, 0.5))
best = c.argmax(axis=1)
dup = {s: ry.columns[best[i]] for i, s in enumerate(rx.columns) if c[i, best[i]] > 0.97}
keep = [s for s in extra.columns if s not in dup]
# duplicates among the recovered names themselves: keep the longer history
rk = rx[keep]
cc = correlations(rk.clip(-0.5, 0.5), rk.clip(-0.5, 0.5)); np.fill_diagonal(cc, 0.0)
length = rk.notna().sum().to_numpy(); drop = set()
for i, j in zip(*np.where(np.triu(cc) > 0.97)):
    if keep[i] in drop or keep[j] in drop: continue
    drop.add(keep[i] if length[i] < length[j] else keep[j])
keep = [s for s in keep if s not in drop]
meta.loc[keep].to_csv(STOCK_DIR / "fmp_accepted.csv")
print(f"recovered {len(extra.columns)}; same company as a stock already held: {len(dup)}; duplicates of each other: {len(drop)}; added: {len(keep)}")
print("examples of renamed-ticker matches:", dict(list(dup.items())[:12]))
print(meta.loc[keep].why.value_counts().to_string())

# ---- how complete is the picture now? every company that left the S&P 500 since 2004
sp = pd.read_csv(STOCK_DIR / "fmp_sp500_history.csv", parse_dates=["date"]).dropna(subset=["removedTicker"])
sp = sp[sp.date >= "2004-01-01"]
have = pd.concat([yahoo, extra[keep]], axis=1)
def covered(row, table):
    t = row.removedTicker.strip()
    if t in dup: t = dup[t]
    if t not in table.columns: return False
    s = table[t].loc[:row.date].dropna()
    return len(s) > 0 and (row.date - s.index[-1]).days <= 45
sp["before"] = [covered(r, yahoo) for r in sp.itertuples()]
sp["after"] = [covered(r, have) for r in sp.itertuples()]
era = pd.cut(sp.date.dt.year, [2003, 2009, 2014, 2019, 2026], labels=["2004-09", "2010-14", "2015-19", "2020-26"])
print("\nCompanies that left the S&P 500: share with usable price history")
print(pd.DataFrame({"left the index": sp.groupby(era, observed=True).size(), "Yahoo only": sp.groupby(era, observed=True).before.mean().round(2),
                    "with FMP added": sp.groupby(era, observed=True).after.mean().round(2)}).to_string())
missing = sp[~sp.after]
print("still missing, most recent:", list(missing.sort_values('date').removedTicker.tail(25)))
