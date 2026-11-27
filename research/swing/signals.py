"""The trading ideas under test, each written with the standard settings from its source.

Nothing here was tuned on this data. Every idea is defined before any result is seen,
and study.py reports all of them, including the ones that fail.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd

from engine import Data, liquid_universe, rsi, sma


class Signals:
    def __init__(self, data: Data):
        self.d = data
        self.cols = data.stocks()
        self.adj = data.adj[self.cols]
        self.open, self.high, self.low = data.open[self.cols], data.high[self.cols], data.low[self.cols]
        self.close, self.volume = data.close[self.cols], data.volume[self.cols]
        self.spy = data.adj["SPY"]

    # ------------------------------------------------------------ building blocks
    @cached_property
    def u200(self): return liquid_universe(self.d, 200)[self.cols]
    @cached_property
    def u500(self): return liquid_universe(self.d, 500)[self.cols]
    @cached_property
    def ret(self): return self.adj.pct_change(fill_method=None)
    @cached_property
    def sma5(self): return sma(self.adj, 5)
    @cached_property
    def sma100(self): return sma(self.adj, 100)
    @cached_property
    def sma200(self): return sma(self.adj, 200)
    @cached_property
    def rsi2(self): return rsi(self.adj, 2)
    @cached_property
    def uptrend(self): return self.adj > self.sma200
    @cached_property
    def vol_ratio(self):
        """Today's volume against the average of the previous 50 days."""
        return self.volume / self.volume.shift(1).rolling(50, min_periods=30).mean()
    @cached_property
    def market_up(self): return self.spy > self.spy.rolling(200).mean()

    def every(self, n: int, offset: int = 0) -> pd.Series:
        flag = pd.Series(False, index=self.adj.index)
        flag.iloc[offset::n] = True
        return flag

    def universe(self, top: int) -> pd.DataFrame:
        return {200: self.u200, 500: self.u500}.get(top) if top in (200, 500) else liquid_universe(self.d, top)[self.cols]

    # ------------------------------------------------------------------ the ideas
    def pullback(self, top=200, rsi_below=10, exit_ma=5, trend=True):
        """A. Buy a sharp short-term drop in a stock that is in a long-term uptrend;
        sell on the first close back above its 5-day average.
        Source: Connors & Alvarez, 'Short Term Trading Strategies That Work' (2008)."""
        entry = self.universe(top) & (self.rsi2 < rsi_below)
        if trend:
            entry &= self.uptrend
        return dict(entry=entry, rank=self.rsi2, exit_=self.adj > sma(self.adj, exit_ma), max_hold=10)

    def weekly_losers(self, top=200, offset=0, decile=0.10):
        """C. Each week buy the stocks that fell most over the past week relative to the
        market (classic short-term reversal; Lehmann 1990, Blitz et al. 2013)."""
        r5 = self.adj / self.adj.shift(5) - 1
        spy_ret = self.spy.pct_change(fill_method=None)
        cov = self.ret.mul(spy_ret, axis=0).rolling(120).mean() - self.ret.rolling(120).mean().mul(
            spy_ret.rolling(120).mean(), axis=0)
        beta = cov.div(spy_ret.rolling(120).var(ddof=0), axis=0)
        resid = (r5 - beta.mul(self.spy / self.spy.shift(5) - 1, axis=0)).where(self.universe(top))
        entry = (resid.rank(axis=1, pct=True) <= decile) & self.every(5, offset).to_numpy()[:, None]
        return dict(entry=entry, rank=resid, exit_=None, max_hold=5)

    def gap_up(self, top=500, gap=0.04, day=0.05, vol=3.0, hold=10):
        """D. Buy a stock the day after it gaps up on very heavy volume and holds the gain
        (the price footprint of a good earnings report; post-earnings drift, Bernard & Thomas 1989)."""
        prev = self.close.shift(1)
        clv = (self.close - self.low) / (self.high - self.low)
        entry = (self.universe(top) & (self.open / prev - 1 >= gap) & (self.close >= self.open)
                 & (self.close / prev - 1 >= day) & (self.vol_ratio >= vol) & (clv >= 0.5))
        return dict(entry=entry, rank=-self.vol_ratio, exit_=None, max_hold=hold)

    def new_high(self, top=500, vol=1.5, hold=10):
        """E. Buy a stock making its first 52-week closing high in a month, on above-average
        volume (52-week-high effect; George & Hwang 2004)."""
        at_high = self.adj >= self.adj.rolling(252, min_periods=252).max()
        recent = at_high.shift(1).rolling(20).max().fillna(0).astype(bool)
        entry = self.universe(top) & at_high & ~recent & (self.vol_ratio >= vol)
        return dict(entry=entry, rank=-self.vol_ratio, exit_=None, max_hold=hold)

    @cached_property
    def trend_score(self):
        """Annualised 90-day trend slope times its smoothness (R squared)."""
        n = 90
        y = np.log(self.adj.to_numpy())
        x = np.arange(n) - (n - 1) / 2
        sxx = (x ** 2).sum()
        win = np.lib.stride_tricks.sliding_window_view(y, n, axis=0)        # (T-n+1, N, n)
        slope = np.full_like(y, np.nan)
        slope[n - 1:] = win @ (x / sxx)
        syy = pd.DataFrame(y).rolling(n, min_periods=n).var(ddof=0).to_numpy() * n
        r2 = np.clip(slope ** 2 * sxx / syy, 0, 1)
        return pd.DataFrame((np.exp(slope * 252) - 1) * r2, index=self.adj.index, columns=self.cols)

    def momentum_leaders(self, top=500, offset=0, hold_n=20, keep_n=100):
        """M. Hold the 20 steadiest uptrends, reviewed weekly, and only buy when the market
        itself is in an uptrend. Source: Clenow, 'Stocks on the Move' (2015), simplified
        to equal position sizes."""
        calm = self.ret.abs().rolling(90).max() < 0.15
        ok = self.universe(top) & (self.adj > self.sma100) & calm
        rk = self.trend_score.where(ok).rank(axis=1, ascending=False)
        review = self.every(5, offset).to_numpy()[:, None]
        return dict(entry=(rk <= hold_n) & review, rank=rk, exit_=~(rk <= keep_n) & review, max_hold=None,
                    entry_gate=self.market_up, slots=hold_n)

    def stock_momentum(self, top=200, offset=0, hold_n=10, keep_n=20):
        """H. The rule already used for funds, applied to stocks: every two weeks hold the
        10 strongest stocks over the past year (skipping the last month) that are in uptrends."""
        mom = (self.adj.shift(21) / self.adj.shift(252) - 1).where(self.universe(top) & self.uptrend)
        rk = mom.rank(axis=1, ascending=False)
        review = self.every(10, offset).to_numpy()[:, None]
        return dict(entry=(rk <= hold_n) & review, rank=rk, exit_=~(rk <= keep_n) & review, max_hold=None,
                    slots=hold_n)
