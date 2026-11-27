# Market strategy: ETF momentum with a trend filter

The main strategy is a simple rule, checked every 10 trading days (two weeks):

1. Rank 22 liquid ETFs by their return from 12 months ago to 1 month ago.
2. Take the top 5. Each one that is above its 200-day average gets a slot.
3. A pick below its 200-day average gets nothing.
4. Spare money sits in SHY (short-term Treasuries).

The default settings use **2x leverage**: each slot is 40% of the account instead
of 20%. With one or two picks in uptrends nothing is borrowed; with three or more the
account borrows on margin, up to 100% of its value when all five qualify. Set
`leverage = 1.0` in `marketalgo/config.py` for no borrowing.

A machine-learning model was tested first and lost to this rule, so it is kept only
as an optional comparison (`--ml`).

## Backtest, 2003-01 to 2026-11

After 5 bps per trade and margin interest of 4% a year above the T-bill rate.
Averaged over five different starting days, rebalancing every 10 trading days:

| Leverage | Return / yr | Sharpe | Worst drawdown | Worst month | Worst year |
|---|---|---|---|---|---|
| 1x (no borrowing) | 9.9% | 0.75 | -21% | -10% | -6% |
| 1.5x | 11.6% | 0.63 | -31% | -15% | -12% |
| 2x (default) | 12.6% | 0.56 | -41% | -21% | -18% |
| SPY buy & hold | 10.9% | 0.65 | -55% | -17% | |

- Leverage raises return, but losses grow faster than gains: 2x adds under 3 points
  of yearly return over 1x and roughly doubles the worst drawdown.
- At 2x the account was borrowing about 90% of the time, and the worst two-week
  stretch lost 29%.
- The 2x rule beat 90% of 200 random-pick portfolios run under the same rules.
  Encouraging, but short of the usual 95% bar for "not luck".
- Rebalancing monthly instead earned slightly more (10.4% at 1x) with a deeper worst
  drawdown (-27%). Weekly earned slightly less (9.6%) with more than twice the trading.
- A real margin rate above the assumed one makes the leveraged rows worse. Taxes are
  not included, and trading this often creates short-term gains.

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
