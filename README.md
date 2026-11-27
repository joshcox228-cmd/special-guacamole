# Market strategy: ML ranking of ETFs, tested honestly

A machine-learning model that ranks 22 liquid ETFs by expected performance over the
next month, holds the top 5, and is replayed through 2003-2026 with trading costs.

## Result so far: the ML model does not beat a simple rule

Out-of-sample, after 5 bps per trade, 2003-01 to 2026-11:

| Strategy | Return / yr | Sharpe | Worst drawdown |
|---|---|---|---|
| ML top-5 | 7.8% | 0.61 | -31% |
| Momentum rule top-5 | 10.9% | 0.79 | -27% |
| Equal weight, all 22 ETFs | 9.3% | 0.70 | -44% |
| SPY buy & hold | 10.9% | 0.65 | -55% |

- The ML portfolio beat only 18% of 200 random-pick portfolios run under the same rules.
- Shifting the rebalance day by 7 and 14 trading days gave ML Sharpe 0.55 and 0.66,
  against 0.70 and 0.75 for the momentum rule. Same conclusion each time.
- The model's forecasts do correlate slightly with outcomes (rank IC about +0.04 to
  +0.06), but that was not enough to pick a better top 5 than plain momentum.

Do not trade the ML signal on this evidence. The momentum rule is the stronger candidate,
and most of its benefit is smaller drawdowns than SPY, not higher return.

## Use

    pip install -r requirements.txt
    python run.py backtest            # full history, chart and tables in outputs/
    python run.py backtest --offset 7 # same, rebalancing on a different day
    python run.py signals             # today's ranking and target weights
    python -m pytest

Prices come from Yahoo Finance (`query1.finance.yahoo.com`) and are cached in `data/`.

## How it works

1. **Features** (`marketalgo/features.py`): momentum over 1-12 months, volatility,
   distance from 50/200-day averages, drawdown, RSI, each asset's rank within the
   universe, and three market-wide readings from SPY.
2. **Label**: each asset's rank among the others over the following 21 trading days.
3. **Model** (`marketalgo/model.py`): a small, heavily regularised LightGBM model,
   retrained at every monthly rebalance using only outcomes already known by then.
4. **Portfolio** (`marketalgo/backtest.py`): equal weight in the top 5 scores. A pick
   trading below its 200-day average is replaced by cash (SHY). Signals use today's
   close and trade at the next close.

Settings are in `marketalgo/config.py`.

## Limits of the backtest

- 24 years of monthly decisions is about 287 data points. That is a small sample.
- Taxes are ignored. Monthly rotation creates short-term gains in a taxable account.
- Past behaviour of momentum may not continue.
- The ETF list was chosen today. It is broad, but not free of hindsight.

This is research code, not financial advice.
