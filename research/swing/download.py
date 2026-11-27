"""Download daily open/high/low/close/volume for every ticker in the pool from Yahoo Finance.

Usage: python download.py <dir containing pool.csv>     (writes one parquet file per field)

close  = split-adjusted price (what a chart shows)
adj    = also adjusted for dividends (use this to measure returns)
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from marketalgo.data import HEADERS, HOSTS, NY, drop_unfinished_bar  # noqa: E402

FIELDS = ("open", "high", "low", "close", "adj", "volume")


def fetch(symbol: str, start: str = "2000-01-01") -> pd.DataFrame | None:
    params = {"period1": int(pd.Timestamp(start, tz="UTC").timestamp()), "period2": int(time.time()) + 86400,
              "interval": "1d", "events": "div,splits"}
    for attempt in range(4):
        try:
            r = requests.get(f"https://{HOSTS[attempt % 2]}/v8/finance/chart/{symbol}", params=params,
                             headers=HEADERS, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            chart = r.json()["chart"]
            if chart.get("error"):
                return None
            res = chart["result"][0]
            if not res.get("timestamp"):
                return None
            idx = (pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(NY).tz_localize(None).normalize())
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                               "adj": res["indicators"]["adjclose"][0]["adjclose"], "volume": q["volume"]},
                              index=idx, dtype="float64")
            df = df.dropna(subset=["open", "close", "adj"])
            df = df[~df.index.duplicated(keep="last")]
            return drop_unfinished_bar(df)
        except Exception:
            time.sleep(2 ** attempt)
    return None


def main(out: Path) -> None:
    pool = pd.read_csv(out / "pool.csv", index_col=0)
    symbols = dict(zip(pool.index, pool["yahoo"]))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = dict(zip(symbols, ex.map(fetch, symbols.values())))
    got = {k: v for k, v in results.items() if v is not None and len(v) >= 60}
    print(f"downloaded {len(got)} of {len(symbols)} in {time.time() - t0:.0f}s; "
          f"missing: {sorted(set(symbols) - set(got))[:40]}")
    for field in FIELDS:
        wide = pd.concat({k: v[field] for k, v in got.items()}, axis=1).sort_index().astype("float32")
        wide.to_parquet(out / f"{field}.parquet")
    print("days:", len(wide), wide.index[0].date(), "to", wide.index[-1].date())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
