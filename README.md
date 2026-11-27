# Market strategy: managed S&P 500 exposure plus fund momentum

The account is run in two halves.

**S&P 500 half, managed by volatility.** Hold more of the S&P 500 when the market is calm
and less when it is wild:

    exposure = 20% / (annualised volatility of the last 20 days), capped at 2x

Exposure up to 1x is held in SPY; anything above 1x is held through the 2x fund SSO, so
there is no margin loan. Changes smaller than 0.1x are ignored. A **dip boost** adds 0.5x
for a few days after a sharp drop while the S&P 500 is above its 200-day average.

**Fund-rule half, reviewed every two weeks.**

1. Rank 22 liquid ETFs by their return from 12 months ago to 1 month ago.
2. Take the top 5. Each one that is above its 200-day average gets 20% of the account.
3. A pick below its 200-day average gets nothing; spare money stays in cash.
4. With three or more picks this half borrows on margin (2x on its own share).

## Backtest, 2003-01 to 2026-11

After trading costs, margin interest of 4% a year above T-bills on the fund rule, and the
measured cost of the 2x fund (about 1.8% above T-bills) on the S&P half. Averaged over
several starting days. Return per year, by period:

| Portfolio | 2003-2014 | 2015-2026 | Whole period | Worst drawdown |
|---|---|---|---|---|
| Managed S&P + fund rule (this portfolio) | 16.6% | 11.6% | 14.1% | -31% |
| Plain S&P + fund rule (previous version) | 14.9% | 10.6% | 12.8% | -38% |
| Managed S&P only | 13.8% | 15.2% | 14.5% | -46% |
| Fund rule only, 2x | 18.3% | 7.6% | 12.8% | -37% |
| SPY buy & hold | 9.9% | 12.9% | 11.4% | -55% |

What this shows, and what it does not:

- Managing the S&P half improved on simply holding SPY in every window tested, including
  the S&P 500 index back to 1960 (about one point a year more, similar worst drop).
- The fund rule's edge came from 2003-2014. Since 2015 it has trailed SPY. It is kept as
  a second engine that behaved very differently in 2008, not because it has beaten the
  market recently.
- A dozen stock-picking and short-term trading ideas were tested and none beat SPY since
  2015 once delisted companies were included. See `research/swing/README.md`.
- These are historical results chosen after testing many alternatives, so expect less.
  Taxes are not included, and frequent adjustments create short-term gains.

## Use

    pip install -r requirements.txt
    python run.py signals                                   # targets for both halves
    python run.py signals --holdings "SPY:8,XLE:61,XLB:73" --cash 3700 --side "QQQ:3"
                                                            # exact trades from what you hold
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

- `marketalgo/managed.py`: the S&P 500 half (volatility-managed exposure and dip boost).
- `marketalgo/features.py`, `marketalgo/backtest.py`: the fund rule and its simulator.
- `marketalgo/plan.py`: combines both halves into whole-account targets and orders. The
  S&P half is adjusted on any day its exposure is off by more than 0.1x; fund positions
  only change on review days.
- `marketalgo/model.py`: the ML model that was tested first and lost to the fund rule.
- `research/swing/`: the stock and swing-trading research behind the current design.

Settings are in `marketalgo/config.py` and `marketalgo/managed.py`.

## Limits of the backtest

- 24 years is only a handful of bear markets. That is a small sample.
- Taxes are ignored. Monthly rotation creates short-term gains in a taxable account.
- Past behaviour of momentum may not continue.
- The ETF list was chosen today. It is broad, but not free of hindsight.

This is research code, not financial advice.
