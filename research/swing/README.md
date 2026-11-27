# Stock and swing-trading research (November 2026)

Question: can shorter-term trading, including individual stocks, beat the portfolio in
the main README?

## Short answer

- **Short-term stock trading (1 to 10 day holds): no.** Every idea either never worked
  or stopped working around 2015.
- **Leverage can be done cheaper and safer.** A 2x fund costs about 1.8% a year above
  T-bills per borrowed dollar (measured from SSO's actual returns), against roughly 4%
  for a margin loan. Scaling S&P 500 exposure down when the market is volatile kept the
  long-run return of constant leverage with a far smaller worst loss.
- **Buying sharp dips in the S&P 500 itself** (not in single stocks) is a small edge that
  held up in both halves of the data.
- **Holding the steadiest uptrending stocks for months** beat the S&P 500 over 24 years
  and matched it since 2015. The data cannot fully verify this one (see Limits).

## Data and method

- About 1,700 US stocks: today's S&P 500, 400 and 600 members, about 200 heavily traded
  names outside those indexes, and 68 known boom-and-bust stocks added on purpose.
  Daily prices from 2000 (Yahoo Finance).
- The tradable list on any day is the 200 or 500 most traded stocks over the previous
  60 days, so a stock only becomes eligible once it was actually heavily traded.
- Each idea was written down with standard settings from its published source before
  any result was seen. All are reported, including the failures.
- Trades fill at the next day's open after an evening signal, with 5 bps cost per trade.
  Results are split into 2003-2014 and 2015-2026 and averaged over different start days.

## Results: stock ideas (return per year, no borrowing)

| Idea | Typical hold | 2003-2014 | 2015-2026 | Worst drop | Verdict |
|---|---|---|---|---|---|
| Buy sharp dips in uptrending stocks | 4 days | 17% | 6% | -38% | Stopped working |
| Buy last week's biggest losers | 5 days | 11% | 9% | -67% | Loses to holding the same stocks |
| Buy after a gap up on heavy volume | 10 days | 3% | 9% | -57% | Unreliable; one great year (2021) |
| Buy fresh 52-week highs | 10 days | 9% | 9% | -34% | Small edge, below the S&P 500 |
| 20 steadiest uptrends, reviewed weekly | 2.5 months | 17% | 13% | -31% | Matched the S&P 500 since 2015 |
| 10 strongest stocks of the past year | 4 months | 18% | 22% | -65% | Too concentrated; halves with 20 names |
| S&P 500 (SPY), for reference | | 10% | 13% | -55% | |

## Results: whole portfolios (2003-2026)

| Portfolio | 2003-2014 | 2015-2026 | All | Worst drop |
|---|---|---|---|---|
| S&P 500, plain | 9.9% | 12.9% | 11.4% | -55% |
| Half S&P + half fund rule at 2x on margin (main README) | 14.9% | 10.6% | 12.8% | -38% |
| A. Volatility-managed S&P (SPY + SSO) | 13.1% | 14.0% | 13.6% | -45% |
| A+. The same with a dip-buying boost | 13.8% | 15.2% | 14.5% | -46% |
| B. Half managed S&P + half fund rule, no margin | 13.3% | 10.7% | 12.0% | -26% |
| C. Half managed S&P + half stock leaders | 15.5% | 14.0% | 14.8% | -30% |

Starting in 2000 instead, to include the 2000-02 bear market: S&P 500 8.3% a year,
A 9.0%, A+ 9.8%, all with a worst drop of 55-59%. On the S&P 500 index since 1960 the
managed approach earned about one point a year more than the index with a similar worst drop.

"Volatility-managed" means: exposure = 20% divided by the S&P 500's annualised 20-day
volatility, capped at 2x, changed only when it is off by more than 0.1. Above 1x the
extra is held through the 2x fund SSO, so there is no margin loan.

## Limits

- **No delisted companies.** Yahoo only has stocks that still trade. This flatters any
  stock backtest. Adding 68 known blow-ups lowered the momentum results by 0.3 to 2
  points a year; the true gap from companies that vanished entirely is unknown.
- **Many ideas were tried.** The best of a dozen will look better than it really is.
- Taxes are ignored. Frequent trading creates short-term gains.
- Fractional shares are assumed for small positions.

## Reproduce

    cd research/swing
    curl -A "research-script/1.0" -o ../../data/wiki/sp500.html "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"   # likewise sp400, sp600
    python -I build_universe.py ../../data/wiki ../../data/stocks
    python download.py ../../data/stocks        # about 5 minutes
    python add_blowups.py
    python study.py            # first pass of every stock idea
    python robust_short.py     # short-term ideas on other stock lists and settings
    python robust_m.py         # stress test of the stock-leaders idea
    python robust_h.py         # concentrated momentum, with and without blow-ups
    python side_tests.py       # overnight effect, sector rotation
    python leverage_tests.py   # cost of leveraged funds, volatility-managed exposure
    python final_table.py      # whole-portfolio comparison
    python -m pytest -q        # simulator checks
