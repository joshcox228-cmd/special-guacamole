"""Daily price history from Yahoo Finance's public chart endpoint, cached on disk."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parent.parent / "data" / "prices.parquet"
HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
NY = ZoneInfo("America/New_York")
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_history(ticker: str, start: str, retries: int = 4) -> pd.DataFrame:
    """Daily history for one ticker with two columns:
    close = the price you actually see on a chart (split-adjusted only);
    adj   = also adjusted for dividends, which is what returns should be measured on."""
    params = {
        "period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
        "period2": int(time.time()) + 86400,
        "interval": "1d",
        "events": "div,splits",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        host = HOSTS[attempt % len(HOSTS)]
        try:
            resp = requests.get(
                f"https://{host}/v8/finance/chart/{ticker}",
                params=params, headers=HEADERS, timeout=30,
            )
            resp.raise_for_status()
            chart = resp.json()["chart"]
            if chart.get("error"):
                raise RuntimeError(chart["error"])
            result = chart["result"][0]
            dates = (
                pd.to_datetime(result["timestamp"], unit="s", utc=True)
                .tz_convert(NY).tz_localize(None).normalize()
            )
            frame = pd.DataFrame({
                "close": result["indicators"]["quote"][0]["close"],
                "adj": result["indicators"]["adjclose"][0]["adjclose"],
            }, index=dates, dtype="float64").dropna()
            return frame[~frame.index.duplicated(keep="last")]
        except Exception as exc:  # network hiccup or rate limit: back off and retry
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download {ticker}: {last_error}")


def fetch_ticker(ticker: str, start: str) -> pd.Series:
    """Dividend- and split-adjusted daily closes for one ticker."""
    return fetch_history(ticker, start)["adj"].rename(ticker)


def drop_unfinished_bar(prices: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Yahoo serves a live bar during market hours. Drop it: a half-finished day
    would leak a price nobody could actually have traded at the close."""
    now = now or datetime.now(NY)
    today = pd.Timestamp(now.date())
    if len(prices) and prices.index[-1] == today and (now.hour, now.minute) < (16, 15):
        return prices.iloc[:-1]
    return prices


def load_prices(tickers, start: str, refresh: bool = False) -> pd.DataFrame:
    """Wide table: one row per trading day, one column per ticker."""
    if CACHE.exists() and not refresh:
        cached = pd.read_parquet(CACHE)
        if set(tickers) <= set(cached.columns):
            return cached[list(tickers)]
    prices = pd.concat([fetch_ticker(t, start) for t in tickers], axis=1).sort_index()
    prices = drop_unfinished_bar(prices)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(CACHE)
    return prices
