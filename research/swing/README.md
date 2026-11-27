# Stock and swing-trading research (November 2026)

Question: can shorter-term trading, including individual stocks, beat the portfolio in
the main README?

## Short answer

- **Trading individual stocks: no.** Twelve ideas were tested, holding from a few days to a
  few months. None beat simply holding the S&P 500 since 2015 once companies that later
  disappeared are included in the data.
- **Leverage can be done cheaper and safer.** A 2x fund costs about 1.8% a year above
  T-bills per borrowed dollar (measured from SSO's actual returns), against roughly 4%
  for a margin loan. Cutting S&P 500 exposure when the market is volatile kept the
  long-run return of constant leverage with a far smaller worst loss.
- **Buying sharp dips in the S&P 500 itself** (not in single stocks) is a small edge that
  held up in both halves of the data.

## Data and method

- **Stocks that still trade** (Yahoo Finance, from 2000): today's S&P 500, 400 and 600
  members, about 300 heavily traded names outside those indexes, and 68 known
  boom-and-bust stocks.
- **Stocks that no longer trade** (FMP): 2,571 companies that were delisted or dropped
  from the S&P 500. Share of companies leaving the S&P 500 with usable price history:

  | Left the index in | Yahoo only | With FMP added |
  |---|---|---|
  | 2004-2009 | 14% | 26% |
  | 2010-2014 | 18% | 40% |
  | 2015-2019 | 25% | 56% |
  | 2020-2026 | 54% | 98% |

  So results for 2020 onward are close to free of survivorship bias; earlier years
  still flatter stock strategies.
- **Earnings reports** (FMP): 121,000 reports with actual and expected earnings.
- The tradable list on any day is the 200 or 500 most traded stocks over the previous
  60 days, so a stock only becomes eligible once it was actually heavily traded.
- Each idea was written down with standard settings from its published source before
  any result was seen. All are reported, including the failures.
- Trades fill at the next day's open after an evening signal, with 5 bps cost per trade.
  A held stock that stops trading is paid out at its last price. Results are split into
  2003-2014 and 2015-2026 and averaged over different start days.

## Results: stock ideas (return per year, no borrowing, delisted companies included)

| Idea | Typical hold | 2003-2014 | 2015-2026 | Worst drop | Verdict |
|---|---|---|---|---|---|
| Buy sharp dips in uptrending stocks | 4 days | 21% | 4% | -41% | Stopped working around 2015 |
| Buy last week's biggest losers | 5 days | 11% | 9% | -67% | Loses to holding the same stocks |
| Buy after a gap up on heavy volume | 10 days | 3% | 9% | -57% | Unreliable; one great year (2021) |
| Buy fresh 52-week highs | 10 days | 12% | 9% | -22% | Below the S&P 500 |
| Buy after a strong earnings beat | 20 days | 16% | 10% | -37% | Drift is about 0.3% over a month: too small |
| 20 steadiest uptrends, reviewed weekly | 2.5 months | 16% | 11% | -34% | Behind the S&P 500 since 2015 |
| 10 strongest stocks of the past year | 4 months | 16% | 20% | -64% | Like a 2x S&P fund, with deeper drops |
| Sector funds: top 3 by one-month gain | 1 week | 9% | 7% | -60% | Loses to holding all sectors |
| S&P 500 (SPY), for reference | | 10% | 13% | -55% | |
| 2x S&P 500 fund, since 2015 | | | 19% | -59% | |

Adding the delisted companies lowered the momentum results by 1.4 to 2.6 points a year.
That is the size of the mistake a survivors-only backtest makes.

## Results: whole portfolios (2003-2026)

| Portfolio | 2003-2014 | 2015-2026 | All | Worst drop | Worst year |
|---|---|---|---|---|---|
| S&P 500, plain | 9.9% | 12.9% | 11.4% | -55% | -37% |
| Half plain S&P + half fund rule at 2x on margin (main README) | 14.9% | 10.6% | 12.8% | -38% | -20% |
| Half managed S&P with dip boost + half fund rule at 2x on margin | 16.6% | 11.6% | 14.1% | -31% | -16% |
| All in managed S&P with dip boost | 13.8% | 15.2% | 14.5% | -46% | -28% |
| All in managed S&P, no dip boost | 13.1% | 14.0% | 13.6% | -45% | -28% |

Starting in 2000 instead, to include the 2000-02 bear market: S&P 500 8.3% a year,
managed S&P 9.0%, with dip boost 9.8%, all with a worst drop of 55-59%. On the S&P 500
index since 1960 the managed approach earned about one point a year more than the index
with a similar worst drop.

**Managed S&P** (`marketalgo/managed.py`): exposure = 20% divided by the S&P 500's
annualised 20-day volatility, capped at 2x, changed only when it is off by more than 0.1.
Above 1x the extra is held through the 2x fund SSO, so there is no margin loan.
**Dip boost**: add 0.5x while the S&P 500 is above its 200-day average and its 2-day RSI
is below 10, until the first close above its 5-day average.

Borrowing through a margin loan instead of SSO lowered the managed result from 13.6% to
12.2% a year. Checking daily, every few days or weekly made little difference (13.1% to 13.6%).

## Limits

- Stock results before 2020 are still flattered, because many companies that vanished
  then could not be recovered.
- Many ideas were tried. The best of a dozen will look better than it really is.
- The managed S&P result leans on US stocks continuing to rise over long periods. In
  2000-2014 it earned 5% a year with a 59% drop, about the same as the index.
- Taxes are ignored. Frequent adjustments create short-term gains.

## Reproduce

    cd research/swing
    # index lists from Wikipedia saved as data/wiki/sp500.html, sp400.html, sp600.html
    python -I build_universe.py ../../data/wiki ../../data/stocks
    python download.py ../../data/stocks          # Yahoo prices, about 5 minutes
    python add_blowups.py
    FMP_KEY_FILE=/path/to/key python fmp_delisted.py   # delisted companies (needs an FMP API key)
    python fmp_merge.py
    FMP_KEY_FILE=/path/to/key python fmp_earnings.py
    python study.py                      # first pass of every stock idea
    python robust_short.py               # short-term ideas on other lists and settings
    python robust_m.py; python robust_h.py
    python survivorship_check.py full    # also: survivors
    python earnings_study.py
    python side_tests.py                 # overnight effect, sector rotation
    python leverage_tests.py             # cost of leveraged funds, managed exposure since 1960
    python final_table2.py               # whole-portfolio comparison
    python -m pytest -q                  # simulator checks

The FMP API key is read from the environment and is never stored in this repository.
