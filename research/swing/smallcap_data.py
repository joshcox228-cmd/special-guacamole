"""Build the small-company dataset (2014 onward) from FMP: every US-listed company trading today,
plus the companies delisted since 2016, with prices, insider purchases and earnings reports.

Usage: FMP_API_KEY=... python smallcap_data.py        (each stage is cached; rerun to resume)
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import fmp
from engine import STOCK_DIR
from fmp_delisted import NOT_A_COMPANY

SC = STOCK_DIR / "smallcap"
CACHE = SC / "price_cache"
FIELDS = ("open", "high", "low", "close", "adj", "volume")
START = "2014-01-01"
BENCH = ["SPY", "IWM", "QQQ", "SHY"]


def pmap(fn, items, workers=10, label=""):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(fn, items))
    print(f"  {label}: {len(items)} requests in {time.time() - t0:.0f}s", flush=True)
    return out


def active_list() -> pd.DataFrame:
    path = SC / "active.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for ex in ("NASDAQ", "NYSE", "AMEX"):
        got = fmp.get("company-screener", exchange=ex, isEtf="false", isFund="false", isActivelyTrading="true", limit=10000)
        rows += got if isinstance(got, list) else []
    df = pd.DataFrame(rows).drop_duplicates("symbol")
    df = df[df.symbol.str.fullmatch(r"[A-Z]{1,5}") & ~df.companyName.fillna("").str.contains(NOT_A_COMPANY)]
    df = df[~((df.symbol.str.len() == 5) & df.symbol.str[-1].isin(list("WUR")))]
    df.to_csv(path, index=False)
    return df


def price(symbol: str) -> pd.DataFrame | None:
    path = CACHE / f"{symbol}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return df if len(df) else None
    x = fmp.get("historical-price-eod/dividend-adjusted", symbol=symbol, **{"from": START})
    if isinstance(x, list) and x:
        df = pd.DataFrame(x).drop_duplicates("date")
        df.index = pd.to_datetime(df.pop("date"))
        df = df.sort_index().rename(columns={"adjOpen": "open", "adjHigh": "high", "adjLow": "low", "adjClose": "close"})
        df["adj"] = df["close"]
        df = df[list(FIELDS)].astype("float64")
    else:
        df = pd.DataFrame(columns=list(FIELDS))
    df.to_parquet(path)
    return df if len(df) else None


def build_prices() -> None:
    if (SC / "adj.parquet").exists():
        return
    act = active_list()
    print(f"active companies: {len(act)}", flush=True)
    symbols = list(dict.fromkeys(BENCH + list(act.symbol)))
    frames = dict(zip(symbols, pmap(price, symbols, label="active prices")))
    frames = {k: v[(v[["open", "high", "low", "close"]] > 0).all(axis=1)] for k, v in frames.items() if v is not None}
    frames = {k: v for k, v in frames.items() if len(v) >= 60}
    # companies that no longer trade: already downloaded by fmp_delisted.py
    meta = pd.read_csv(STOCK_DIR / "fmp_added.csv", index_col=0)
    gone = {}
    for sym in meta.index[meta.why == "delisted"]:
        p = STOCK_DIR / "fmp_cache" / f"{sym}.parquet"
        if sym in frames or not p.exists():
            continue
        df = pd.read_parquet(p)
        df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)].loc[START:]
        if len(df) >= 60:
            gone[sym] = df
    print(f"usable: {len(frames)} trading today, {len(gone)} delisted", flush=True)
    allf = {**frames, **gone}
    for f in FIELDS:
        pd.concat({k: v[f] for k, v in allf.items()}, axis=1, sort=True).astype("float32").to_parquet(SC / f"{f}.parquet")
    pool = pd.DataFrame({"name": "", "sector": "", "yahoo": "", "source": ["benchmark" if k in BENCH else "active" if k in frames else "delisted" for k in allf]},
                        index=list(allf))
    info = act.set_index("symbol")
    pool["name"] = info["companyName"].reindex(pool.index).fillna(meta["name"].reindex(pool.index)).fillna("")
    pool["sector"] = info["sector"].reindex(pool.index).fillna("")
    pool["cap"] = info["marketCap"].reindex(pool.index)
    pool.to_csv(SC / "pool.csv")


def liquid_symbols(top: int = 3000) -> list[str]:
    """Every company that was among the `top` most traded at some point since 2016."""
    path = SC / "liquid_symbols.txt"
    if path.exists():
        return path.read_text().split()
    close = pd.read_parquet(SC / "close.parquet").astype("float64"); vol = pd.read_parquet(SC / "volume.parquet").astype("float64")
    dollars = (close * vol).rolling(60, min_periods=40).median()
    dollars = dollars.drop(columns=[c for c in BENCH if c in dollars.columns])
    rank = dollars.rank(axis=1, ascending=False)
    ever = (rank.loc["2016-01-01":] <= top).any()
    syms = list(ever.index[ever])
    path.write_text("\n".join(syms))
    return syms


def insider(symbol: str) -> list[dict]:
    rows, page = [], 0
    while page < 5:
        x = fmp.get("insider-trading/search", symbol=symbol, transactionType="P-Purchase", page=page, limit=1000)
        if not isinstance(x, list) or not x:
            break
        rows += x
        if len(x) < 1000:
            break
        page += 1
    return rows


def earnings(symbol: str) -> list[dict]:
    x = fmp.get("earnings", symbol=symbol, limit=250)
    return [dict(r, symbol=symbol) for r in x] if isinstance(x, list) else []


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    build_prices()
    syms = liquid_symbols()
    print(f"companies ever in the 3000 most traded since 2016: {len(syms)}", flush=True)
    if not (SC / "insider.parquet").exists():
        rows = [r for chunk in pmap(insider, syms, label="insider purchases") for r in chunk]
        df = pd.DataFrame(rows)
        keep = ["symbol", "filingDate", "transactionDate", "reportingCik", "reportingName", "typeOfOwner", "acquisitionOrDisposition",
                "securitiesTransacted", "price", "securitiesOwned", "securityName", "formType"]
        df = df[keep]
        for c in ("filingDate", "transactionDate"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        df.to_parquet(SC / "insider.parquet")
        print(f"insider purchase rows: {len(df)}; filings by year: {df.groupby(df.filingDate.dt.year).size().loc[2012:].to_dict()}", flush=True)
    if not (SC / "earnings.parquet").exists():
        rows = [r for chunk in pmap(earnings, syms, label="earnings") for r in chunk]
        df = pd.DataFrame(rows)[["symbol", "date", "epsActual", "epsEstimated", "revenueActual", "revenueEstimated"]]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).drop_duplicates(["symbol", "date"])
        df.to_parquet(SC / "earnings.parquet")
        have = df.dropna(subset=["epsActual", "epsEstimated"])
        print(f"earnings reports with actual and estimate: {len(have)} for {have.symbol.nunique()} companies; by year: {have.groupby(have.date.dt.year).size().loc[2014:].to_dict()}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
