"""Turn scores into a portfolio and replay it through history with costs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

TRADING_DAYS = 252


def target_weights(scores: pd.Series, trend_ok: pd.Series, cfg: Config) -> pd.Series:
    """Equal weight in the top-N scores; a pick in a downtrend is swapped for cash."""
    weights = pd.Series(0.0, index=list(cfg.tickers))
    picks = scores.dropna().nlargest(cfg.top_n).index
    slot = 1.0 / cfg.top_n
    for ticker in picks:
        if cfg.trend_filter and not bool(trend_ok.get(ticker, False)):
            weights[cfg.cash] += slot
        else:
            weights[ticker] += slot
    weights[cfg.cash] += 1.0 - weights.sum()  # fewer than N eligible assets
    return weights


def weights_from_scores(scores: pd.DataFrame, trend_ok: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return pd.DataFrame({d: target_weights(scores.loc[d], trend_ok.loc[d], cfg)
                         for d in scores.index}).T


def simulate(prices: pd.DataFrame, targets: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Daily replay. A target set on signal day t is traded at the close of
    t + exec_lag and starts earning the day after. Weights drift between trades."""
    cols = list(targets.columns)
    rets = prices[cols].pct_change(fill_method=None).fillna(0.0).to_numpy()
    dates = prices.index
    trade_at = {dates.get_loc(d) + cfg.exec_lag: targets.loc[d].to_numpy() for d in targets.index}
    first = min(trade_at)

    w = np.zeros(len(cols))
    daily, turnover = [], []
    for i in range(first, len(dates)):
        gross = float(w @ rets[i])
        if w.sum() > 0:
            w = w * (1 + rets[i]) / (1 + gross)  # holdings drift with prices
        traded = 0.0
        if i in trade_at:
            traded = float(np.abs(trade_at[i] - w).sum())
            w = trade_at[i].copy()
        daily.append((1 + gross) * (1 - traded * cfg.cost_bps / 1e4) - 1)
        turnover.append(traded)
    out = pd.DataFrame({"ret": daily, "turnover": turnover}, index=dates[first:])
    out = out.iloc[1:]  # the first row is just the opening purchase
    out["equity"] = (1 + out["ret"]).cumprod()
    return out


def buy_and_hold(prices: pd.DataFrame, ticker: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    ret = prices[ticker].pct_change(fill_method=None).reindex(index).fillna(0.0)
    return pd.DataFrame({"ret": ret, "turnover": 0.0, "equity": (1 + ret).cumprod()})


def metrics(result: pd.DataFrame) -> dict:
    ret = result["ret"]
    years = len(ret) / TRADING_DAYS
    equity = (1 + ret).cumprod()
    cagr = equity.iloc[-1] ** (1 / years) - 1
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    max_dd = (equity / equity.cummax() - 1).min()
    monthly = (1 + ret).groupby([ret.index.year, ret.index.month]).prod() - 1
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": ret.mean() * TRADING_DAYS / vol if vol else np.nan,
        "Max drawdown": max_dd,
        "Worst month": monthly.min(),
        "Winning months": (monthly > 0).mean(),
        "Turnover / yr": result["turnover"].sum() / years / 2,  # one-way
    }


def information_coefficient(scores: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Per signal date: rank correlation between what the model predicted and what
    happened next. 0 = coin flip, +1 = perfect foresight. ~0.03-0.05 is realistic."""
    ic = {}
    for d in scores.index:
        pair = pd.concat([scores.loc[d], fwd.loc[d]], axis=1).dropna()
        if len(pair) >= 5:
            ic[d] = pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank())
    return pd.Series(ic)


def random_pick_sharpes(prices: pd.DataFrame, eligible: pd.DataFrame, trend_ok: pd.DataFrame,
                        cfg: Config, n_sims: int = 200) -> np.ndarray:
    """Sharpe ratios of portfolios that pick at random under the same rules.
    If the model cannot beat most monkeys throwing darts, it has no skill."""
    rng = np.random.default_rng(cfg.seed)
    sharpes = []
    for _ in range(n_sims):
        noise = pd.DataFrame(rng.random(eligible.shape), index=eligible.index,
                             columns=eligible.columns).where(eligible)
        result = simulate(prices, weights_from_scores(noise, trend_ok, cfg), cfg)
        sharpes.append(metrics(result)["Sharpe"])
    return np.array(sharpes)
