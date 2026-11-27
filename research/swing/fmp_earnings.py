"""Download past earnings reports (date, actual and expected earnings per share) for every stock
that was ever among the 500 most traded.   Usage: FMP_KEY_FILE=... python fmp_earnings.py"""
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import fmp
from engine import STOCK_DIR, load, liquid_universe

d = load()
u = liquid_universe(d, 500)
symbols = [c for c in u.columns if u[c].any()]
print(len(symbols), "symbols", flush=True)
def one(sym):
    x = fmp.get("earnings", symbol=sym, limit=250)
    return [dict(r, symbol=sym) for r in x] if isinstance(x, list) else []
with ThreadPoolExecutor(max_workers=5) as ex:
    rows = [r for chunk in ex.map(one, symbols) for r in chunk]
df = pd.DataFrame(rows)[["symbol", "date", "epsActual", "epsEstimated", "revenueActual", "revenueEstimated"]]
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).drop_duplicates(["symbol", "date"])
df.to_parquet(STOCK_DIR / "fmp_earnings.parquet")
have = df.dropna(subset=["epsActual", "epsEstimated"])
print(len(df), "reports;", len(have), "with both actual and expected;", have.symbol.nunique(), "symbols covered")
print(have.groupby(have.date.dt.year).size().to_string())
