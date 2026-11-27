"""Volatility-managed S&P 500 exposure, with an optional dip-buying boost.

The idea: hold more of the index when the market is calm and less when it is wild, so the
account's own swings stay near a chosen level. Exposure above 1x is held through a 2x fund
(SSO) rather than a margin loan: it is cheaper and cannot trigger a margin call.

    exposure = target volatility / recent volatility      (capped, here at 2x)

Research and evidence: research/swing/README.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class ManagedConfig:
    index: str = "SPY"           # 1x S&P 500 fund
    fund2x: str = "SSO"          # 2x S&P 500 fund, used for exposure above 1x
    target_vol: float = 0.20     # aim for swings of about 20% a year
    lookback: int = 20           # measured over the last 20 trading days
    cap: float = 2.0             # never more than 2x
    band: float = 0.10           # ignore changes smaller than this, to avoid constant tinkering
    dip_boost: float = 0.5       # extra exposure while a dip-buying signal is on (0 turns it off)
    dip_rsi: float = 10.0        # dip = 2-day RSI below this ...
    trend_ma: int = 200          # ... while the index is above its 200-day average
    dip_exit_ma: int = 5         # boost ends on the first close above the 5-day average
    fund_cost: float = 0.018     # yearly cost above T-bills per borrowed dollar inside the 2x fund


def realized_vol(prices: pd.Series, lookback: int) -> pd.Series:
    """Annualised volatility of daily returns over the last `lookback` days."""
    return prices.pct_change(fill_method=None).rolling(lookback).std() * np.sqrt(TRADING_DAYS)


def rsi(prices: pd.Series, n: int = 2) -> pd.Series:
    d = prices.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def dip_signal(prices: pd.Series, cfg: ManagedConfig) -> pd.Series:
    """True from the close a dip is seen until the first close back above the short average."""
    enter = ((rsi(prices, 2) < cfg.dip_rsi) & (prices > prices.rolling(cfg.trend_ma).mean())).to_numpy()
    leave = (prices > prices.rolling(cfg.dip_exit_ma).mean()).to_numpy()
    on, held = np.zeros(len(prices), bool), False
    for i in range(len(prices)):
        if held and leave[i]:
            held = False
        elif not held and enter[i]:
            held = True
        on[i] = held
    return pd.Series(on, index=prices.index)


def exposure_path(prices: pd.Series, cfg: ManagedConfig) -> pd.DataFrame:
    """Day by day: measured volatility, the exposure it calls for, the exposure actually held
    after applying the no-trade band, whether the dip boost is on, and the total."""
    vol = realized_vol(prices, cfg.lookback)
    wanted = (cfg.target_vol / vol).clip(upper=cfg.cap)
    base, current = np.zeros(len(prices)), 0.0
    for i, w in enumerate(wanted.fillna(0.0).to_numpy()):
        if abs(w - current) > cfg.band:
            current = w
        base[i] = current
    dip = dip_signal(prices, cfg) if cfg.dip_boost else pd.Series(False, index=prices.index)
    total = np.minimum(base + cfg.dip_boost * dip.to_numpy(), cfg.cap)
    return pd.DataFrame({"vol": vol, "wanted": wanted, "base": base, "dip": dip, "exposure": total}, index=prices.index)


def split(exposure: float) -> dict[str, float]:
    """Turn an exposure number into shares of the account: 1x fund, 2x fund, cash."""
    exposure = float(min(max(exposure, 0.0), 2.0))
    if exposure <= 1.0:
        return {"index": exposure, "fund2x": 0.0, "cash": 1.0 - exposure}
    return {"index": 2.0 - exposure, "fund2x": exposure - 1.0, "cash": 0.0}


def backtest(prices: pd.Series, bill_rate: pd.Series, cfg: ManagedConfig, lag: int = 2,
             cost_bps: float = 2.0) -> pd.DataFrame:
    """Daily returns. lag=2 means: read the numbers after a close, trade at the next close.
    bill_rate is the yearly T-bill yield as a fraction (0.04 = 4%)."""
    r = prices.pct_change(fill_method=None).fillna(0.0)
    e = exposure_path(prices, cfg)["exposure"].shift(lag).fillna(0.0)
    rf = (bill_rate.reindex(prices.index).ffill().fillna(0.0)) / TRADING_DAYS
    borrowed, idle = (e - 1).clip(lower=0), (1 - e).clip(lower=0)
    ret = (e * r - borrowed * (rf + cfg.fund_cost / TRADING_DAYS) + idle * rf
           - e.diff().abs().fillna(0.0) * cost_bps / 1e4)
    return pd.DataFrame({"ret": ret, "exposure": e})
