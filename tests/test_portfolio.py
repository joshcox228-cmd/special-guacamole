"""The sell-rule checks must say exactly what the written rules say."""

import numpy as np
import pandas as pd
import pytest

from marketalgo import portfolio as pf


def test_close_below_stop_is_an_exit():
    assert pf.classify(99.0, 100, 150, 180)[0] == "EXIT"


def test_exit_outranks_everything_and_trim_two_outranks_trim_one():
    assert pf.classify(160.0, 100, 150, 180)[0] == "TRIM"
    assert "trim 2" in pf.classify(185.0, 100, 150, 180)[1]
    assert "trim 1" in pf.classify(160.0, 100, 150, 180)[1]


def test_watch_band_and_quiet():
    assert pf.classify(102.0, 100, np.nan, np.nan)[0] == "WATCH"
    assert pf.classify(110.0, 100, np.nan, np.nan)[0] == "QUIET"
    assert pf.classify(110.0, np.nan, np.nan, np.nan)[0] == "NO STOP"


def test_ema_stop_sits_under_nearest_support():
    stop, note = pf.ema_stop(100.0, {50: 98.0, 88: 95.0, 200: 105.0})
    assert stop == pytest.approx(98.0 * (1 - pf.STOP_CUSHION))
    assert "50-day" in note


def test_no_stop_is_suggested_below_all_emas():
    stop, note = pf.ema_stop(90.0, {50: 98.0, 88: 95.0, 200: 105.0})
    assert stop is None
    assert "reclaim 95.00" in note


def test_review_flags_a_falling_holding(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=400)
    falling = pd.Series(np.linspace(200, 100, 400), index=dates)
    rising = pd.Series(np.linspace(100, 200, 400), index=dates)
    history = {"DOWN": pd.DataFrame({"close": falling, "adj": falling}),
               "UP": pd.DataFrame({"close": rising, "adj": rising})}
    path = tmp_path / "portfolio.csv"
    path.write_text("# comment\nticker,book,shares,avg_cost,stop,trim1,trim2\n"
                    "down,core,10,150,120,,\nUP,satellite,10,150,150,,\n")
    rep = pf.review(pf.load_portfolio(path), history)
    assert rep.loc["DOWN", "status"] == "EXIT" and rep.loc["UP", "status"] == "QUIET"
    assert rep.loc["UP", "uptrend"] and not rep.loc["DOWN", "uptrend"]
    assert rep["weight"].sum() == pytest.approx(1.0)
    assert rep.loc["UP", "mom_rank"] == 1
