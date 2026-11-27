# Market strategy: ETF momentum with a trend filter

The main strategy is a simple rule: each month, rank 22 liquid ETFs by their return
from 12 months ago to 1 month ago, hold the top 5 at 20% each, and replace any pick
trading below its 200-day average with cash (SHY).

A machine-learning model was tested first and lost to this rule, so it is kept only
as an optional comparison (`--ml`).

## Backtest, 2003-01 to 2026-11, after 5 bps per trade

| Strategy | Return / yr | Sharpe | Worst drawdown |
|---|---|---|---|
| Momentum rule top-5 | 10.9% | 0.79 | -27% |
| ML top-5 (comparison) | 7.4% | 0.58 | -31% |
| Equal weight, all 22 ETFs | 9.3% | 0.70 | -44% |
| SPY buy & hold | 10.9% | 0.65 | -55% |

- The momentum rule matched SPY's return with about half the worst drawdown.
- It beat 86% of 200 random-pick portfolios run under the same rules. That is
  encouraging but short of the usual 95% bar for "not luck".
- Shifting the rebalance day by 7 and 14 trading days gave Sharpe 0.70 and 0.75,
  so the result depends somewhat on which day of the month you trade.
- The edge is in risk, not return: expect to lag SPY in strong bull markets.

## Use

    pip install -r requirements.txt
    python run.py signals             # what to hold now
    python run.py portfolio           # check your own holdings against your sell rules
    python run.py backtest            # full history, chart and tables in outputs/
    python run.py backtest --offset 7 # same, rebalancing on a different day
    python run.py backtest --ml       # add the ML model for comparison (about 3 minutes)
    python -m pytest

Prices come from Yahoo Finance (`query1.finance.yahoo.com`) and are cached in `data/`.

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

1. **Features** (`marketalgo/features.py`): momentum, volatility, trend and relative
   rank for each ETF. The momentum rule uses two of them: 12-1 month momentum and
   distance from the 200-day average. The ML model uses all of them.
2. **Portfolio** (`marketalgo/backtest.py`): equal weight in the top 5 scores, with the
   trend filter. Signals use today's close and trade at the next close.
3. **ML comparison** (`marketalgo/model.py`): a small LightGBM model, retrained at
   every monthly rebalance using only outcomes already known by then.

Settings are in `marketalgo/config.py`.

## Limits of the backtest

- 24 years of monthly decisions is about 287 data points. That is a small sample.
- Taxes are ignored. Monthly rotation creates short-term gains in a taxable account.
- Past behaviour of momentum may not continue.
- The ETF list was chosen today. It is broad, but not free of hindsight.

This is research code, not financial advice.
