"""Checks that the simulator fills at the right price on the right day and cannot see the future."""

import numpy as np
import pandas as pd
import pytest

from engine import Data, clean_data, simulate, rsi, sma, liquid_universe, nw_tstat


def make(prices_open: dict, prices_close: dict, volume=1e6) -> Data:
    dates = pd.bdate_range("2020-01-01", periods=len(next(iter(prices_close.values()))))
    o = pd.DataFrame(prices_open, index=dates, dtype=float)
    c = pd.DataFrame(prices_close, index=dates, dtype=float)
    v = pd.DataFrame(volume, index=dates, columns=c.columns, dtype=float)
    pool = pd.DataFrame({"name": "", "sector": "", "source": "sp500"}, index=c.columns)
    return Data(open=o, high=pd.concat([o, c]).groupby(level=0).max(), low=pd.concat([o, c]).groupby(level=0).min(),
                close=c, adj=c.copy(), volume=v, pool=pool)


def flags(data, **true_at):
    f = pd.DataFrame(False, index=data.dates, columns=data.adj.columns)
    for col, days in true_at.items():
        f.loc[data.dates[list(days)], col] = True
    return f


OPEN = {"A": [10, 10, 11, 12, 13, 14, 15, 16], "B": [50, 50, 50, 50, 50, 50, 50, 50]}
CLOSE = {"A": [10, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5], "B": [50, 50, 50, 50, 50, 50, 50, 50]}


def test_next_open_fills_one_day_after_the_signal():
    d = make(OPEN, CLOSE)
    entry, exit_ = flags(d, A=[1]), flags(d, A=[4])       # see buy signal at close of day 1, sell signal day 4
    daily, tr = simulate(d, entry, entry * 0.0, exit_, slots=1, cost_bps=0, start=d.dates[0])
    assert len(tr) == 1
    t = tr.iloc[0]
    assert t.entry == d.dates[2] and t.exit == d.dates[5]  # bought at open of day 2, sold at open of day 5
    assert t.ret == pytest.approx(14 / 11 - 1)
    assert (1 + daily.ret).prod() == pytest.approx(14 / 11)
    assert daily.ret.iloc[1] == 0                           # nothing owned on the signal day itself


def test_same_close_fills_on_the_signal_day():
    d = make(OPEN, CLOSE)
    daily, tr = simulate(d, flags(d, A=[1]), flags(d) * 0.0, flags(d, A=[4]), slots=1, cost_bps=0,
                         start=d.dates[0], mode="same_close")
    assert tr.iloc[0].ret == pytest.approx(13.5 / 10.5 - 1)


def test_costs_are_charged_on_both_sides():
    d = make(OPEN, CLOSE)
    _, tr = simulate(d, flags(d, A=[1]), flags(d) * 0.0, flags(d, A=[4]), slots=1, cost_bps=10, start=d.dates[0])
    assert tr.iloc[0].ret == pytest.approx(14 * 0.999 / (11 * 1.001) - 1)


def test_time_stop_and_slot_limit_and_ranking():
    d = make(OPEN, CLOSE)
    entry = flags(d, A=[1], B=[1])
    rank = pd.DataFrame({"A": 2.0, "B": 1.0}, index=d.dates)   # B ranks better
    daily, tr = simulate(d, entry, rank, None, slots=1, max_hold=3, cost_bps=0, start=d.dates[0])
    assert list(tr.ticker) == ["B"]                             # only one slot, best rank wins
    assert tr.iloc[0].days == 3                                 # bought open of day 2, sold open of day 5
    assert tr.iloc[0].entry == d.dates[2] and tr.iloc[0].exit == d.dates[5]
    _, tr2 = simulate(d, entry, rank, None, slots=1, max_hold=3, cost_bps=0, start=d.dates[0], mode="same_close")
    assert tr2.iloc[0].entry == d.dates[1] and tr2.iloc[0].exit == d.dates[4] and tr2.iloc[0].days == 3


def test_position_is_a_fixed_share_of_the_account():
    d = make(OPEN, CLOSE)
    daily, _ = simulate(d, flags(d, A=[1]), flags(d) * 0.0, None, slots=4, cost_bps=0, start=d.dates[0])
    # a quarter of the account in A from the open of day 2; day 2 return = 0.25 * (11.5/11 - 1)
    assert daily.ret.iloc[2] == pytest.approx(0.25 * (11.5 / 11 - 1))
    assert daily.invested.iloc[2] == pytest.approx(0.25 * 11.5 / 11 / (1 + 0.25 * (11.5 / 11 - 1)))


def test_future_prices_cannot_change_past_results():
    rng = np.random.default_rng(1)
    n = 400
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, 6)), axis=0))
    o = c * np.exp(rng.normal(0, 0.005, (n, 6)))
    names = list("ABCDEF")
    d1 = make({k: o[:, i] for i, k in enumerate(names)}, {k: c[:, i] for i, k in enumerate(names)})
    c2, o2 = c.copy(), o.copy()
    c2[300:] *= 3; o2[300:] *= 0.2                               # rewrite the future
    d2 = make({k: o2[:, i] for i, k in enumerate(names)}, {k: c2[:, i] for i, k in enumerate(names)})
    res = []
    for d in (d1, d2):
        entry = (rsi(d.adj, 2) < 10) & (d.adj > sma(d.adj, 50))
        daily, _ = simulate(d, entry, rsi(d.adj, 2), d.adj > sma(d.adj, 5), slots=3, start=d.dates[60])
        res.append(daily.ret.iloc[:238])                          # days 60..297: all fills at or before day 298
    pd.testing.assert_series_equal(res[0], res[1])


def test_universe_uses_only_past_volume():
    d = make({k: [10.0] * 400 for k in "ABC"}, {k: [10.0] * 400 for k in "ABC"})
    d.volume.loc[d.dates[350:], "C"] = 1e9                      # C becomes huge only at the end
    d.volume["A"] = 5e6
    u = liquid_universe(d, top=1, lookback=20, min_history=100)
    assert u.loc[d.dates[340], "A"] and not u.loc[d.dates[340], "C"]
    assert u.loc[d.dates[399], "C"]


def test_tstat_of_noise_is_small():
    assert abs(nw_tstat(np.random.default_rng(0).normal(0, 1, 5000))) < 3


def test_cleaning_neutralises_a_fake_jump_but_keeps_a_real_crash():
    close = {"SPY": [100.0] * 8, "A": [1, 1, 1, 50, 50, 50, 50, 50], "B": [100, 100, 100, 5, 5, 5, 5, 5]}
    d = make({k: list(v) for k, v in close.items()}, close)
    d.pool.loc["SPY", "source"] = "benchmark"
    c = clean_data(d)
    ra, rb = c.adj["A"].pct_change().dropna(), c.adj["B"].pct_change().dropna()
    assert ra.abs().max() == pytest.approx(0.0)              # the x50 splice earns nothing
    assert rb.min() == pytest.approx(-0.95)                  # the -95% collapse is still a loss
    assert (c.close["A"] * c.volume["A"]).iloc[0] == pytest.approx((d.close["A"] * d.volume["A"]).iloc[0])


def test_never_borrows_even_when_earlier_winners_have_grown():
    close = {"A": [10, 10, 10, 40, 40, 40, 40, 40], "B": [10.0] * 8, "C": [10.0] * 8}
    d = make({k: list(v) for k, v in close.items()}, close)
    entry = flags(d, A=[0], B=[0], C=[4])                       # A quadruples before C is bought
    daily, _ = simulate(d, entry, flags(d) * 0.0, None, slots=3, weight=0.5, cost_bps=0, start=d.dates[0])
    assert daily.invested.max() <= 1.0 + 1e-9
