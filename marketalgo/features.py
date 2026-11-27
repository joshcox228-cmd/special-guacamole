"""Turn raw prices into model inputs (features) and the thing we predict (labels).

Every feature at date t uses prices up to and including t only. tests/ checks this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

TRADING_DAYS = 252
RANKED = ("ret_21", "ret_63", "ret_126", "mom_12_1", "vol_63", "dist_sma200", "risk_adj_mom")


def _rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def build_wide_features(prices: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Each value is a dates x tickers table for one feature."""
    px = prices[list(cfg.universe)]
    rets = px.pct_change(fill_method=None)
    f: dict[str, pd.DataFrame] = {}

    # Momentum: how much has it moved over the last 1 / 3 / 6 / 12 months?
    for days in (21, 63, 126, 252):
        f[f"ret_{days}"] = px / px.shift(days) - 1
    # Classic "12-1" momentum skips the latest month, which tends to reverse.
    f["mom_12_1"] = px.shift(21) / px.shift(252) - 1

    # Risk: how bumpy has the ride been?
    f["vol_21"] = rets.rolling(21).std() * np.sqrt(TRADING_DAYS)
    f["vol_63"] = rets.rolling(63).std() * np.sqrt(TRADING_DAYS)
    f["risk_adj_mom"] = f["ret_126"] / (rets.rolling(126).std() * np.sqrt(126))

    # Trend: where is price relative to its own averages and recent high?
    f["dist_sma50"] = px / px.rolling(50).mean() - 1
    f["dist_sma200"] = px / px.rolling(200).mean() - 1
    f["drawdown_252"] = px / px.rolling(252).max() - 1
    f["rsi_14"] = _rsi(px)

    # Relative standing: the same numbers as a 0-1 rank across the universe that day.
    for name in RANKED:
        f[f"{name}_rank"] = f[name].rank(axis=1, pct=True)

    # Market weather: identical for every ticker on a given day.
    bench = prices[cfg.benchmark]
    bench_ret = bench.pct_change(fill_method=None)
    weather = {
        "mkt_dist_sma200": bench / bench.rolling(200).mean() - 1,
        "mkt_vol_21": bench_ret.rolling(21).std() * np.sqrt(TRADING_DAYS),
        "mkt_ret_63": bench / bench.shift(63) - 1,
    }
    for name, series in weather.items():
        f[name] = pd.DataFrame({t: series for t in px.columns})
    return f


def to_long(wide: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack into one row per (date, ticker)."""
    long = pd.concat({k: v.stack() for k, v in wide.items()}, axis=1)
    long.index.names = ["date", "ticker"]
    return long


def forward_returns(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return actually earned if you see the signal at close t, buy at the close
    `exec_lag` days later, and hold for `horizon` days."""
    px = prices[list(cfg.universe)]
    entry = px.shift(-cfg.exec_lag)
    return px.shift(-(cfg.exec_lag + cfg.horizon)) / entry - 1


def build_labels(fwd: pd.DataFrame) -> pd.DataFrame:
    """Label = how the asset ranked against the others over the next month,
    from -0.5 (worst) to +0.5 (best). Ranks are far less noisy than raw returns."""
    return fwd.rank(axis=1, pct=True) - 0.5
