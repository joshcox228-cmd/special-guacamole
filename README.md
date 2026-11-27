# Market strategy: S&P 500 core plus ETF momentum

Half the account is always held in SPY. The other half is run by a momentum rule,
checked every 10 trading days (two weeks):

1. Rank 22 liquid ETFs by their return from 12 months ago to 1 month ago.
2. Take the top 5. Each one that is above its 200-day average gets a slot.
3. A pick below its 200-day average gets nothing.
4. Spare money sits in SHY (short-term Treasuries).

The rule's half uses **2x leverage**: each slot is 20% of the whole account. With up
to two picks in uptrends nothing is borrowed; with all five the account borrows 50%
of its value. Settings (`core_weight`, `leverage`) are in `marketalgo/config.py`.

A machine-learning model was tested first and lost to the momentum rule, so it is
kept only as an optional comparison (`--ml`).

## Backtest, 2003-01 to 2026-11

After 5 bps per trade and margin interest of 4% a year above the T-bill rate,
averaged over five starting days. Return per year, by period:

| Strategy | 2003-2014 | 2015-2026 | Whole period | Worst drawdown |
|---|---|---|---|---|
| Half SPY + half rule at 2x (default) | 14.7% | 10.5% | 12.6% | -39% |
| Rule only, 2x | 18.2% | 7.4% | 12.6% | -41% |
| Rule only, no borrowing | 12.8% | 7.0% | 9.9% | -21% |
| SPY buy & hold | 10.0% | 12.9% | 11.4% | -55% |

What this shows:

- **The momentum rule's edge came from 2003-2014.** It avoided most of 2008 and rode
  the commodities boom. Since 2015, holding SPY has beaten it by about 5 points a year,
  because recent sell-offs were short and the rule missed the rebounds.
- **Since 2015, leverage on the rule added almost nothing** (7.4% against 7.0%) while
  doubling the worst drawdown.
- **The SPY core is a hedge against that**, giving up some of the first period for a
  better second. It was chosen after seeing these results, so it is a judgment call,
  not a proven improvement.
- Nothing tested here reliably beats SPY. Three refinements (blended momentum
  look-backs, volatility-sized slots, and both) did not help in both periods.
- A real margin rate above the assumed one makes leveraged rows worse. Taxes are not
  included, and trading this often creates short-term gains.

## Use

    pip install -r requirements.txt
    python run.py signals             # what to hold now
    python run.py signals --account 13000   # same, with share counts for that account size
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
2. **Portfolio** (`marketalgo/backtest.py`): equal slots in the top 5 scores, with the
   trend filter and leverage. Signals use today's close and trade at the next close.
3. **ML comparison** (`marketalgo/model.py`): a small LightGBM model, retrained at
   every monthly rebalance using only outcomes already known by then.

Settings are in `marketalgo/config.py`.

## Limits of the backtest

- 24 years is only a handful of bear markets. That is a small sample.
- Taxes are ignored. Monthly rotation creates short-term gains in a taxable account.
- Past behaviour of momentum may not continue.
- The ETF list was chosen today. It is broad, but not free of hindsight.

This is research code, not financial advice.
