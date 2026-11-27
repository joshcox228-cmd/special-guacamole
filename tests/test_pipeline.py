"""Guards against the classic ways a backtest lies to you."""

import numpy as np
import pandas as pd
import pytest

from marketalgo import backtest as bt
from marketalgo import features as ft
from marketalgo import model as ml
from marketalgo.config import Config

CFG = Config(universe=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"), cash="CASH",
             benchmark="AAA", top_n=2, min_train_days=300, leverage=1.0,
             rebalance_every=21, core_weight=0.0)


@pytest.fixture(scope="module")
def prices():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=700)
    rets = rng.normal(0.0004, 0.01, size=(len(dates), len(CFG.tickers)))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates, columns=list(CFG.tickers))


def test_features_do_not_look_ahead(prices):
    """Features for a day must be identical whether or not the future exists yet."""
    cut = 500
    full = ft.to_long(ft.build_wide_features(prices, CFG))
    truncated = ft.to_long(ft.build_wide_features(prices.iloc[: cut + 1], CFG))
    day = prices.index[cut]
    pd.testing.assert_frame_equal(full.xs(day, level="date"), truncated.xs(day, level="date"))


def test_training_data_excludes_unknown_outcomes(prices):
    """Every label used in training must be fully realised before the prediction day."""
    dates = prices.index
    pos = 400
    train = ml.training_rows(dates, pos, CFG)
    last_label_end = dates.get_loc(train[-1]) + CFG.exec_lag + CFG.horizon
    assert last_label_end <= pos


def test_forward_return_matches_tradeable_prices(prices):
    fwd = ft.forward_returns(prices, CFG)
    i = 100
    expected = prices["BBB"].iloc[i + 1 + CFG.horizon] / prices["BBB"].iloc[i + 1] - 1
    assert fwd["BBB"].iloc[i] == pytest.approx(expected)


def test_target_weights_respect_trend_filter():
    scores = pd.Series({"AAA": 0.9, "BBB": 0.8, "CCC": 0.1})
    trend_ok = pd.Series({"AAA": True, "BBB": False, "CCC": True})
    w = bt.target_weights(scores, trend_ok, CFG)
    assert w["AAA"] == 0.5 and w["BBB"] == 0 and w["CASH"] == 0.5
    assert w.sum() == pytest.approx(1.0)


def test_leverage_scales_slots_and_borrows_only_when_needed():
    lev = Config(**{**CFG.__dict__, "leverage": 2.0})
    scores = pd.Series({"AAA": 0.9, "BBB": 0.8, "CCC": 0.1})
    one_up = bt.target_weights(scores, pd.Series({"AAA": True, "BBB": False}), lev)
    assert one_up["AAA"] == 1.0 and one_up["CASH"] == 0.0          # fully invested, no loan
    both_up = bt.target_weights(scores, pd.Series({"AAA": True, "BBB": True}), lev)
    assert both_up["AAA"] == 1.0 and both_up["BBB"] == 1.0 and both_up["CASH"] == -1.0


def test_core_holding_is_always_held_and_rule_runs_the_rest():
    cfg = Config(**{**CFG.__dict__, "leverage": 2.0, "core": "FFF", "core_weight": 0.5})
    scores = pd.Series({"AAA": 0.9, "BBB": 0.8, "CCC": 0.1})
    none_up = bt.target_weights(scores, pd.Series({"AAA": False, "BBB": False}), cfg)
    assert none_up["FFF"] == 0.5 and none_up["CASH"] == 0.5        # core stays even with no picks
    both_up = bt.target_weights(scores, pd.Series({"AAA": True, "BBB": True}), cfg)
    assert both_up["AAA"] == 0.5 and both_up["BBB"] == 0.5 and both_up["FFF"] == 0.5
    assert both_up["CASH"] == -0.5 and both_up.sum() == pytest.approx(1.0)


def test_margin_loan_costs_interest(prices):
    flat = prices.copy()
    flat[:] = 100.0                                                 # nothing moves
    day = flat.index[10]
    targets = pd.DataFrame(0.0, index=[day], columns=list(CFG.tickers))
    targets.loc[day, ["AAA", "CASH"]] = [2.0, -1.0]                 # borrow 100% of the account
    cfg = Config(**{**CFG.__dict__, "cost_bps": 0.0, "margin_spread": 0.0504})
    result = bt.simulate(flat, targets, cfg)
    assert result["ret"].iloc[0] == pytest.approx(-0.0504 / 252, rel=1e-3)


def test_simulation_trades_one_day_after_signal_and_charges_costs(prices):
    signal_day = prices.index[10]
    targets = pd.DataFrame(0.0, index=[signal_day], columns=list(CFG.tickers))
    targets.loc[signal_day, "CCC"] = 1.0
    result = bt.simulate(prices, targets, CFG)

    # Bought at the close of day 11, so the first return earned is day 12's.
    assert result.index[0] == prices.index[12]
    held = prices["CCC"].iloc[12:] / prices["CCC"].iloc[11]
    np.testing.assert_allclose(result["equity"].to_numpy(), held.to_numpy())

    free = bt.simulate(prices, pd.concat([targets, targets.rename({signal_day: prices.index[40]})]),
                       CFG)
    assert free["turnover"].sum() == pytest.approx(0.0)  # re-buying the same thing costs nothing


def test_switching_positions_costs_money(prices):
    days = prices.index[[10, 40]]
    targets = pd.DataFrame(0.0, index=days, columns=list(CFG.tickers))
    targets.loc[days[0], "AAA"] = 1.0
    targets.loc[days[1], "BBB"] = 1.0
    with_cost = bt.simulate(prices, targets, CFG)
    without = bt.simulate(prices, targets, Config(**{**CFG.__dict__, "cost_bps": 0.0}))
    assert with_cost["turnover"].sum() == pytest.approx(2.0)  # sell 100%, buy 100%
    ratio = with_cost["equity"].iloc[-1] / without["equity"].iloc[-1]
    assert ratio == pytest.approx(1 - 2 * CFG.cost_bps / 1e4)


def test_walk_forward_runs_and_scores_only_eligible_assets(prices):
    features = ft.to_long(ft.build_wide_features(prices, CFG))
    labels = ft.build_labels(ft.forward_returns(prices, CFG))
    scores = ml.walk_forward(features, labels, Config(**{**CFG.__dict__, "rebalance_every": 100}))
    assert len(scores) == 4
    assert set(scores.columns) <= set(CFG.universe)
    assert scores.notna().all().all()
