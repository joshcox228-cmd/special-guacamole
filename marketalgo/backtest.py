"""Turn scores into a portfolio and replay it through history with costs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

TRADING_DAYS = 252


def target_weights(scores: pd.Series, trend_ok: pd.Series, cfg: Config) -> pd.Series:
    """Equal slots in the top-N scores; a pick in a downtrend gets nothing.

    Each slot is leverage / top_n of the account. Whatever is left sits in the cash
    fund. If the slots add up to more than the account, the cash weight goes negative:
    that is money borrowed on margin.

    With a core holding, all of the above is scaled down to the rule's share of the
    account and the core fund is added on top."""
    weights = pd.Series(0.0, index=list(cfg.tickers))
    slot = cfg.leverage / cfg.top_n
    for ticker in scores.dropna().nlargest(cfg.top_n).index:
        if not cfg.trend_filter or bool(trend_ok.get(ticker, False)):
            weights[ticker] += slot
    weights[cfg.cash] += 1.0 - weights.sum()
    if cfg.core_weight:  # the rule only runs its share; the core is held regardless
        weights *= 1.0 - cfg.core_weight
        weights[cfg.core] += cfg.core_weight
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
    cash = cols.index(cfg.cash)

    w = np.zeros(len(cols))
    daily, turnover = [], []
    for i in range(first, len(dates)):
        # A negative cash weight is a margin loan: it costs the cash fund's return
        # (about the T-bill rate) through the line below, plus the broker's spread.
        gross = float(w @ rets[i]) - max(-w[cash], 0.0) * cfg.margin_spread / TRADING_DAYS
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
