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
    rebalance_every: int = 21    # trade about once a month
    offset: int = 0              # shift the rebalance day; used for robustness checks
    top_n: int = 5               # number of positions held
    trend_filter: bool = True    # a pick below its 200-day average goes to cash instead

    min_train_days: int = 756    # need ~3 years of history before the first prediction
    sample_every: int = 5        # train on weekly snapshots (daily ones overlap heavily)
    exec_lag: int = 1            # signal at today's close, trade at tomorrow's close
    cost_bps: float = 5.0        # cost per dollar traded, in basis points
    seed: int = 7

    @property
    def tickers(self) -> tuple:
        extra = [t for t in (self.cash, self.benchmark) if t not in self.universe]
        return tuple(self.universe) + tuple(extra)
