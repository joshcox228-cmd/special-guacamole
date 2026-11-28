"""All the knobs for the strategy live here."""

from dataclasses import dataclass

# Broad, liquid ETFs that have existed for most of the test period. Using funds
# (not hand-picked stocks) avoids survivorship bias: we are not quietly choosing
# today's winners and then "predicting" that they did well.
UNIVERSE = (
    "SPY", "QQQ", "IWM", "MDY",                      # US equity size / style
    "EFA", "EEM",                                    # international equity
    "XLK", "XLF", "XLE", "XLV", "XLY",               # US sectors
    "XLP", "XLI", "XLB", "XLU",
    "VNQ",                                           # real estate
    "TLT", "IEF", "LQD", "HYG",                      # bonds
    "GLD", "DBC",                                    # gold, commodities
)


@dataclass(frozen=True)
class Config:
    universe: tuple = UNIVERSE
    cash: str = "SHY"            # where money sits when a pick fails the trend filter
    benchmark: str = "SPY"
    start: str = "2000-01-01"

    horizon: int = 21            # predict the next ~1 month (trading days)
    rebalance_every: int = 10    # check and trade every two weeks
    offset: int = 0              # shift the rebalance day; used for robustness checks
    top_n: int = 5               # number of positions held
    trend_filter: bool = True    # a pick below its 200-day average goes to cash instead
    leverage: float = 2.0        # each slot is leverage / top_n of the account; above 1.0
                                 # the account borrows once enough picks are in uptrends
    margin_spread: float = 0.04  # yearly margin interest above the T-bill rate
    # Sleeves managed by volatility: (1x fund, 2x fund, share of the account). See managed.py.
    # Whatever is left over is run by the fund rule.
    managed: tuple = (("SPY", "SSO", 0.30), ("QQQ", "QLD", 0.30))
    # Slices that are simply held: (fund to hold, price history used for backtests, share of the account).
    # Bought back to target only when they drift more than `fixed_band` (a quarter) away from it.
    fixed: tuple = (("IBIT", "BTC-USD", 0.10),)
    fixed_band: float = 0.25
    review_anchor: str = "2026-12-11"   # the fund rule is reviewed on this date and every 14 days after
    core: str = "SPY"            # used only for the plain "hold the S&P 500" comparisons
    core_weight: float = 0.0     # share of the account simply held in `core` (0 = not used)

    min_train_days: int = 756    # need ~3 years of history before the first prediction
    sample_every: int = 5        # train on weekly snapshots (daily ones overlap heavily)
    exec_lag: int = 1            # signal at today's close, trade at tomorrow's close
    cost_bps: float = 5.0        # cost per dollar traded, in basis points
    seed: int = 7

    @property
    def rule_weight(self) -> float:
        """Share of the account run by the fund rule."""
        return 1.0 - sum(share for _, _, share in self.managed) - sum(share for _, _, share in self.fixed)

    @property
    def tickers(self) -> tuple:
        extra = [t for t in dict.fromkeys((self.cash, self.benchmark, self.core)) if t not in self.universe]
        return tuple(self.universe) + tuple(extra)
