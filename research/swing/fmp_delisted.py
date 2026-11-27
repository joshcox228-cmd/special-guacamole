"""Add the companies Yahoo no longer has: stocks that were delisted, or dropped out of the S&P 500.

Candidates come from two FMP lists: its delisted-companies list (good from 2016) and every
ticker ever removed from or added to the S&P 500 (reaches further back). A candidate is kept
only if its price history really covers the period before it left, which screens out
tickers that were later reused by a different company.

Usage: FMP_KEY_FILE=... python fmp_delisted.py
"""

from __future__ import annotations

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import fmp
from engine import STOCK_DIR

NOT_A_COMPANY = re.compile(
    r"Acquisition|Warrant|\bUnits?\b|\bRights?\b|ETF|ETN|\bFund\b|ProShares|iShares|Direxion|Invesco|SPDR|VanEck|WisdomTree|"
    r"Global X|First Trust|iPath|MicroSectors|Preferred|\bNotes\b|Depositary|Treasury|\bBond\b|\bIndex\b|%|Merger Corp|"
    r"Capital Corp\. [IVX]+|Holdings Corp\. [IVX]+|SPAC", re.I)


def candidates(live: set[str]) -> pd.DataFrame:
    dl = pd.read_csv(STOCK_DIR / "fmp_delisted_list.csv", parse_dates=["ipoDate", "delistedDate"])
    dl = dl[dl.exchange.isin(["NASDAQ", "NYSE", "AMEX"]) & dl.symbol.str.fullmatch(r"[A-Z]{1,5}")]
    dl = dl[~dl.companyName.fillna("").str.contains(NOT_A_COMPANY)]
    dl = dl[~((dl.symbol.str.len() == 5) & dl.symbol.str[-1].isin(list("WUR")))]
    dl = dl[(dl.delistedDate - dl.ipoDate).dt.days >= 400]
    a = pd.DataFrame({"symbol": dl.symbol, "ref": dl.delistedDate, "why": "delisted", "name": dl.companyName})
    sp = pd.read_csv(STOCK_DIR / "fmp_sp500_history.csv", parse_dates=["date"])
    sp = sp[sp.date >= "2003-01-01"]
    rem = sp.dropna(subset=["removedTicker"])
    b = pd.DataFrame({"symbol": rem.removedTicker.str.strip(), "ref": rem.date, "why": "left S&P 500", "name": rem.removedSecurity})
    c = pd.DataFrame({"symbol": sp.symbol.str.strip(), "ref": sp.date, "why": "joined S&P 500", "name": sp.addedSecurity})
    out = pd.concat([b, a, c]).dropna(subset=["symbol"])
    out = out[out.symbol.str.fullmatch(r"[A-Z]{1,5}") & ~out.symbol.isin(live)]
    return out.drop_duplicates("symbol").reset_index(drop=True)


def history(symbol: str) -> pd.DataFrame | None:
    rows, to = [], None
    for _ in range(3):                                # the API returns at most 5000 bars per call
        params = {"symbol": symbol, "from": "2000-01-01"}
        if to:
            params["to"] = to
        x = fmp.get("historical-price-eod/dividend-adjusted", **params)
        if not isinstance(x, list) or not x:
            break
        rows += x
        if len(x) < 5000:
            break
        to = (pd.Timestamp(min(r["date"] for r in x)) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates("date")
    df.index = pd.to_datetime(df.pop("date"))
    df = df.sort_index().rename(columns={"adjOpen": "open", "adjHigh": "high", "adjLow": "low", "adjClose": "close"})
    df["adj"] = df["close"]
    return df[["open", "high", "low", "close", "adj", "volume"]].astype("float64")


def usable(df: pd.DataFrame | None, ref: pd.Timestamp, why: str) -> bool:
    """Does the history genuinely cover the time before the company left?"""
    if df is None or len(df) < 252:
        return False
    if why == "joined S&P 500":                       # still listed: just needs a real history
        return df.index[-1] >= pd.Timestamp("2026-01-01") or len(df) >= 500
    before = df[df.index <= ref + pd.Timedelta(days=10)]
    if len(before) < 252:
        return False
    return (ref - before.index[-1]).days <= 45        # was still trading right up to the exit


CACHE = STOCK_DIR / "fmp_cache"


def cached_history(symbol: str) -> pd.DataFrame | None:
    """Download once; later runs read the saved copy."""
    path = CACHE / f"{symbol}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return df if len(df) else None
    df = history(symbol)
    (df if df is not None else pd.DataFrame(columns=["open", "high", "low", "close", "adj", "volume"])).to_parquet(path)
    return df


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    live = set(pd.read_parquet(STOCK_DIR / "adj.parquet").columns)
    cand = candidates(live)
    print(f"{len(cand)} candidates: {cand.why.value_counts().to_dict()}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(cached_history, cand.symbol))
    print(f"downloaded in {time.time() - t0:.0f}s", flush=True)
    keep, meta = {}, []
    for (_, row), df in zip(cand.iterrows(), results):
        if df is None or not len(df):
            continue
        df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
        if usable(df, row.ref, row.why):
            keep[row.symbol] = df
            meta.append({"symbol": row.symbol, "why": row.why, "ref": row.ref.date(), "name": row["name"],
                         "first": df.index[0].date(), "last": df.index[-1].date(), "bars": len(df)})
    print(f"kept {len(keep)} of {len(cand)}", flush=True)
    meta = pd.DataFrame(meta)
    meta.to_csv(STOCK_DIR / "fmp_added.csv", index=False)
    print(meta.why.value_counts().to_string())
    for f in ("open", "high", "low", "close", "adj", "volume"):
        pd.concat({k: v[f] for k, v in keep.items()}, axis=1).sort_index().astype("float32").to_parquet(STOCK_DIR / f"fmp_{f}.parquet")


if __name__ == "__main__":
    main()
