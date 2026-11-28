"""Whole-account targets and orders must add up."""

import pandas as pd
import pytest

from marketalgo import plan
from marketalgo.config import Config
from marketalgo.managed import ManagedConfig

CFG = Config(managed=(("SPY", "SSO", 0.5),), fixed=())   # one sleeve: half the account in managed S&P
MCFG = ManagedConfig()
THREE = Config(managed=(("SPY", "SSO", 1 / 3), ("QQQ", "QLD", 1 / 3)), fixed=())   # S&P, Nasdaq, fund rule in thirds


def rule(**w):
    s = pd.Series(0.0, index=list(CFG.tickers))
    for k, v in w.items():
        s[k] = v
    s[CFG.cash] = 1.0 - s.drop(CFG.cash).sum()
    return s


def test_weights_add_up_and_cash_from_one_half_offsets_the_other():
    # fund rule wants two funds at 40% each of its own half; S&P half wants 0.6x
    w = plan.account_weights(rule(XLE=0.4, XLB=0.4), {"SPY": 0.6}, CFG)
    assert w.sum() == pytest.approx(1.0)
    assert w["XLE"] == pytest.approx(0.2) and w["SPY"] == pytest.approx(0.3)
    assert w[CFG.cash] == pytest.approx(0.1 + 0.2)                  # 10% from the rule + 20% from the S&P half
    assert w["SSO"] == 0


def test_exposure_above_one_goes_through_the_2x_fund_without_a_loan():
    w = plan.account_weights(rule(XLE=0.4), {"SPY": 1.5}, CFG)
    assert w["SPY"] == pytest.approx(0.25) and w["SSO"] == pytest.approx(0.25)
    assert w[CFG.cash] == pytest.approx(0.3)                         # the S&P half is fully used; no borrowing


def test_rule_borrowing_shows_as_negative_cash():
    w = plan.account_weights(rule(XLE=0.4, XLB=0.4, XLK=0.4, XLV=0.4, QQQ=0.4), {"SPY": 2.0}, CFG)
    assert w.sum() == pytest.approx(1.0)
    assert w[CFG.cash] == pytest.approx(-0.5) and w["SSO"] == pytest.approx(0.5)


def test_orders_round_to_the_nearest_share_and_skip_tiny_changes():
    w = pd.Series({"SPY": 0.315, "XLE": 0.20, "SHY": 0.485})
    prices = pd.Series({"SPY": 726.0, "XLE": 62.0, "SHY": 81.6})
    o = plan.orders(w, prices, 18_975, {"SPY": 13, "XLE": 61})
    assert o.loc["SPY", "target"] == 8 and o.loc["SPY", "trade"] == -5
    assert o.loc["XLE", "target"] == 61 and o.loc["XLE", "trade"] == 0
    assert "SHY" not in o.index                                       # spare money stays as cash
    o2 = plan.orders(w, prices, 19_300, {"SPY": 8, "XLE": 61})       # account drifted up a little
    assert (o2["trade"] == 0).all()


def test_a_holding_the_plan_no_longer_wants_is_sold_in_full():
    w = pd.Series({"SPY": 0.5, "SHY": 0.5})
    o = plan.orders(w, pd.Series({"SPY": 700.0, "XLB": 52.0}), 20_000, {"SPY": 14, "XLB": 3})
    assert o.loc["XLB", "trade"] == -3 and o.loc["SPY", "trade"] == 0


def test_review_calendar():
    anchor = pd.Timestamp("2026-12-11")
    d = pd.Timestamp
    assert not plan.review_due(d("2026-12-04"), d("2026-12-03"), anchor)
    assert plan.review_due(d("2026-12-11"), d("2026-12-10"), anchor)
    assert not plan.review_due(d("2026-12-14"), d("2026-12-11"), anchor)
    assert not plan.review_due(d("2026-12-18"), d("2026-12-17"), anchor)
    assert plan.review_due(d("2026-12-28"), d("2026-12-24"), anchor)   # Friday the 25th is a holiday


def test_frozen_positions_are_left_alone():
    w = pd.Series({"SPY": 0.30, "XLE": 0.10, "SHY": 0.60})
    prices = pd.Series({"SPY": 700.0, "XLE": 60.0})
    o = plan.orders(w, prices, 20_000, {"SPY": 14, "XLE": 61}, frozen={"XLE"})
    assert o.loc["XLE", "trade"] == 0 and o.loc["SPY", "trade"] == -5        # 8.57 shares rounds to 9


def test_exposure_held_counts_the_2x_fund_twice_and_ignores_the_rules_share():
    prices = pd.Series({"SPY": 700.0, "SSO": 60.0})
    e = plan.exposure_held({"SPY": 5, "SSO": 50}, prices, 20_000, MCFG, 0.5)
    assert e == pytest.approx((5 * 700 + 2 * 50 * 60) / 10_000)
    e2 = plan.exposure_held({"SPY": 5, "SSO": 0}, prices, 20_000, MCFG, 0.5, rule_weight_in_index=0.05)
    assert e2 == pytest.approx((5 * 700 - 1_000) / 10_000)


def test_three_way_split_adds_up():
    w = plan.account_weights(rule(XLE=0.4, XLB=0.4), {"SPY": 0.63, "QQQ": 0.5}, THREE)
    assert w.sum() == pytest.approx(1.0)
    assert w["SPY"] == pytest.approx(0.63 / 3) and w["QQQ"] == pytest.approx(0.5 / 3)
    assert w["XLE"] == pytest.approx(0.4 / 3) and w["SSO"] == 0 and w["QLD"] == 0
    assert THREE.rule_weight == pytest.approx(1 / 3)
    calm = plan.account_weights(rule(XLE=0.4, XLB=0.4, QQQ=0.4), {"SPY": 2.0, "QQQ": 1.5}, THREE)
    assert calm["SSO"] == pytest.approx(1 / 3) and calm["QLD"] == pytest.approx(0.5 / 3)
    assert calm["QQQ"] == pytest.approx(0.5 / 3 + 0.4 / 3)            # the sleeve's QQQ plus the rule's QQQ slot


def test_rounding_never_spends_more_cash_than_the_plan_allows():
    w = pd.Series({"AAA": 0.5, "BBB": 0.5, "SHY": 0.0})
    prices = pd.Series({"AAA": 600.0, "BBB": 600.0})
    o = plan.orders(w, prices, 1_900)                                 # 1.58 shares each would round up to 2 + 2 = 2,400
    assert (o["target"] * o["price"]).sum() <= 1_900


def test_default_plan_has_a_held_bitcoin_slice():
    cfg = Config()
    w = plan.account_weights(rule(XLE=0.4, XLB=0.4), {"SPY": 0.63, "QQQ": 0.5}, cfg)
    assert w.sum() == pytest.approx(1.0)
    assert w["IBIT"] == pytest.approx(0.10) and cfg.rule_weight == pytest.approx(0.30)
    assert w["SPY"] == pytest.approx(0.30 * 0.63) and w["XLE"] == pytest.approx(0.30 * 0.4)
