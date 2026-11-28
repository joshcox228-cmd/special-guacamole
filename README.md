# Market strategy: managed S&P 500 and Nasdaq exposure, fund momentum, and a Bitcoin slice

The account is run in four parts.

**Managed S&P 500 and managed Nasdaq-100 (30% each).** Hold more of the index when the
market is calm and less when it is wild:

    exposure = 20% / (annualised volatility of the last 20 days), capped at 2x

Exposure up to 1x is held in the plain fund (SPY, QQQ); anything above 1x is held through
the matching 2x fund (SSO, QLD), so there is no margin loan. Changes smaller than 0.1x are
ignored. A **dip boost** adds 0.5x for a few days after a sharp drop while the index is
above its 200-day average.

**Fund rule (30%), reviewed every two weeks.**

1. Rank 22 liquid ETFs by their return from 12 months ago to 1 month ago.
2. Take the top 5. Each one that is above its 200-day average gets a slot (12% of the account).
3. A pick below its 200-day average gets nothing; spare money stays in cash.
4. With three or more picks this part borrows on margin (2x on its own share).

**Bitcoin (10%), simply held** through a spot Bitcoin fund (IBIT). It is topped up or
trimmed only when it drifts more than a quarter away from 10%. This slice is a speculative
choice by the account owner, not something the research found an edge in.

## Backtest

After trading costs, margin interest of 4% a year above T-bills on the fund rule, and the
measured cost of the 2x funds (about 1.8% above T-bills) on the managed parts.

The three tested parts, 2003-01 to 2026-11, averaged over several starting days:

| Portfolio | 2003-2014 | 2015-2026 | Whole period | Worst drawdown |
|---|---|---|---|---|
| Managed S&P + managed Nasdaq + fund rule, in thirds | 17.0% | 14.4% | 15.7% | -34% |
| Managed S&P + fund rule, halves (earlier version) | 16.6% | 11.6% | 14.1% | -31% |
| Managed S&P only | 13.8% | 15.2% | 14.5% | -46% |
| Managed Nasdaq only | 17.2% | 19.4% | 18.3% | -40% |
| SPY buy & hold | 9.9% | 12.9% | 11.4% | -55% |

With the Bitcoin slice, since 2021 only (the period after large institutions arrived):

| Portfolio | A year since 2021 | Worst drawdown |
|---|---|---|
| Current plan, 10% Bitcoin | 18.1% | -27% |
| Same plan without Bitcoin | 17.2% | -23% |
| Bitcoin, simply held | 13.3% | -77% |
| SPY buy & hold | 13.7% | -24% |

What this shows, and what it does not:

- Managing exposure improved on simply holding the index in every window tested,
  including the S&P 500 back to 1960 (about one point a year more, similar worst drop).
- **The Nasdaq part is a bet that large technology companies keep leading.** In 2000-2002
  the Nasdaq-100 lost 41% a year; the managed version lost 19% a year.
- **The Bitcoin slice has no cash flows behind it.** Since 2021 Bitcoin returned about what
  the S&P 500 did, with drawdowns three times as deep. `run.py backtest` reports a much
  higher long-run figure for the current plan because Bitcoin rose about 100-fold from
  2015 to 2020; that will not repeat and should be ignored.
- The fund rule's edge came from 2003-2014. Since 2015 it has trailed SPY. It is kept as
  a different kind of engine that behaved very differently in 2008.
- About fifteen stock-picking and short-term trading ideas were tested, in large and
  small companies, and none beat SPY once delisted companies were included. See
  `research/swing/README.md`.
- These are historical results chosen after testing many alternatives, so expect less.
  Taxes are not included, and frequent adjustments create short-term gains.

## Use

    pip install -r requirements.txt
    python run.py signals                                   # targets for every part
    python run.py signals --holdings "SPY:5,QQQ:4,XLE:37,XLB:44,IBIT:56" --cash 6300
                                                            # exact trades from what you hold
    python run.py signals --holdings "..." --cash N --rebalance   # line everything up with the targets
    python run.py backtest                                  # full history, chart and tables in outputs/
    python run.py backtest --ml                             # add the ML model for comparison
    python run.py portfolio                                 # check holdings against written sell rules
    python -m pytest

`--side` lists positions held outside the plan; they are left alone and paid for out of
the plan's spare cash. Prices come from Yahoo Finance and are cached in `data/`.

## Checking your own portfolio

Copy `portfolio.example.csv` to `portfolio.csv` and list your holdings with their stop
and trim prices. `portfolio.csv` is ignored by git, so real positions are never
committed. `python run.py portfolio` then reports, for the latest close:

- **Exit**: closed below its stop.
- **Trim**: closed at or above trim 1 or trim 2.
- **Watch**: within 3% of the stop.
- **Trend warning**: no rule has fired, but the holding is below its 200-day average
  with negative 12-1 month momentum, so the momentum strategy would not own it.
- **EMA stop**: where the stop would sit today under the 50/88/200-day EMA method
  (1.25% under the nearest EMA below price), to show when a written stop has gone stale.

It only reports. It never places trades, and thesis-break triggers are not checked.

## How it works

- `marketalgo/managed.py`: the managed sleeves (volatility-managed exposure and dip boost).
- `marketalgo/features.py`, `marketalgo/backtest.py`: the fund rule and its simulator.
- `marketalgo/plan.py`: combines the parts into whole-account targets and orders. A
  managed sleeve is adjusted on any day its exposure is off by more than 0.1x; fund
  positions only change on review days.
- `marketalgo/model.py`: the ML model that was tested first and lost to the fund rule.
- `research/swing/`: the stock and swing-trading research behind the current design.

Settings are in `marketalgo/config.py` and `marketalgo/managed.py`.

## Limits of the backtest

- 24 years is only a handful of bear markets. That is a small sample.
- Taxes are ignored. Monthly rotation creates short-term gains in a taxable account.
- Past behaviour of momentum may not continue.
- The ETF list was chosen today. It is broad, but not free of hindsight.

This is research code, not financial advice.
