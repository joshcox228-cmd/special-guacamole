"""Combine the two halves of the portfolio into one set of holdings and orders.

Half the account follows the managed S&P 500 exposure (managed.py); the other half follows
the fund-rotation rule (backtest.py). This module turns both into whole-account targets
and, given current holdings, into the trades needed.
"""

from __future__ import annotations

import math

import pandas as pd

from .config import Config
from .managed import ManagedConfig, split


def account_weights(rule_weights: pd.Series, exposure: float, cfg: Config, mcfg: ManagedConfig) -> pd.Series:
    """Whole-account weights. `rule_weights` are the fund rule's weights as if it ran the
    whole account (they sum to 1; a negative cash weight is a margin loan). `exposure` is
    the managed S&P exposure, between 0 and 2. Cash from one half offsets borrowing in the other."""
    w = rule_weights * (1 - cfg.core_weight)
    for ticker in (mcfg.index, mcfg.fund2x, cfg.cash):
        if ticker not in w.index:
            w[ticker] = 0.0
    parts = split(exposure)
    w[mcfg.index] += cfg.core_weight * parts["index"]
    w[mcfg.fund2x] += cfg.core_weight * parts["fund2x"]
    w[cfg.cash] += cfg.core_weight * parts["cash"]
    return w


def orders(weights: pd.Series, prices: pd.Series, value: float, holdings: dict[str, float] | None = None,
           cash_ticker: str = "SHY", min_trade: float = 0.02, frozen: set[str] | None = None) -> pd.DataFrame:
    """Target share counts for an account worth `value` and the trades to get there.

    Whole shares, rounded down. Spare money is left as cash (the cash fund is not bought).
    When holdings are given, a difference worth less than `min_trade` of the account is
    left alone, so small drifts do not turn into pointless trades. Tickers in `frozen`
    are not traded at all today (for example fund-rule positions between reviews)."""
    holdings = dict(holdings or {})
    frozen = frozen or set()
    tickers = [t for t in weights.index if t != cash_ticker and (weights[t] > 0 or t in holdings)]
    tickers += [t for t in holdings if t not in tickers and t != cash_ticker]
    rows = []
    for t in tickers:
        w = float(weights.get(t, 0.0))
        price = float(prices[t])
        target = math.floor(value * w / price + 1e-9)
        held = float(holdings.get(t, 0.0))
        trade = target - held
        if t in frozen:
            trade = 0.0
        elif holdings and abs(trade * price) < min_trade * value and not (target == 0 and held > 0):
            trade = 0.0
        rows.append({"ticker": t, "weight": w, "price": price, "target": target, "held": held, "trade": trade,
                     "after": held + trade, "value": (held + trade) * price})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame(
        columns=["weight", "price", "target", "held", "trade", "after", "value"])


def review_due(today: pd.Timestamp, previous_day: pd.Timestamp, anchor: pd.Timestamp, every_days: int = 14) -> bool:
    """The fund rule is reviewed on a fixed two-week calendar: on the anchor date and every
    `every_days` after it, or on the first trading day after if the market was closed."""
    if today < anchor:
        return False
    latest = anchor + pd.Timedelta(days=every_days * ((today - anchor).days // every_days))
    return previous_day < latest <= today


def core_exposure_held(holdings: dict[str, float], prices: pd.Series, value: float, cfg: Config,
                       mcfg: ManagedConfig) -> float:
    """The S&P exposure the account actually carries, as a multiple of the core half."""
    dollars = holdings.get(mcfg.index, 0.0) * prices[mcfg.index] + 2 * holdings.get(mcfg.fund2x, 0.0) * prices[mcfg.fund2x]
    return float(dollars / (cfg.core_weight * value))
