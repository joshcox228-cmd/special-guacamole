"""The managed-exposure rules must do exactly what they say, and never peek ahead."""

import numpy as np
import pandas as pd
import pytest

from marketalgo import managed as mg


def series(returns, start=100.0):
    dates = pd.bdate_range("2020-01-01", periods=len(returns) + 1)
    return pd.Series(start * np.cumprod(np.r_[1.0, 1 + np.asarray(returns)]), index=dates)


def test_split_uses_the_2x_fund_only_above_one():
    assert mg.split(0.6) == {"index": 0.6, "fund2x": 0.0, "cash": pytest.approx(0.4)}
    assert mg.split(1.0) == {"index": 1.0, "fund2x": 0.0, "cash": 0.0}
    s = mg.split(1.5)
    assert s == {"index": 0.5, "fund2x": 0.5, "cash": 0.0}
    assert s["index"] + 2 * s["fund2x"] == pytest.approx(1.5)        # the mix really delivers 1.5x
    assert mg.split(2.7) == {"index": 0.0, "fund2x": 1.0, "cash": 0.0}


def test_calm_market_hits_the_cap_and_wild_market_cuts_exposure():
    cfg = mg.ManagedConfig(dip_boost=0.0)
    calm = series(np.tile([0.004, -0.004], 60))                      # about 6% a year
    wild = series(np.tile([0.03, -0.03], 60))                        # about 48% a year
    assert mg.exposure_path(calm, cfg)["exposure"].iloc[-1] == pytest.approx(2.0)
    e = mg.exposure_path(wild, cfg)["exposure"].iloc[-1]
    assert e == pytest.approx(cfg.target_vol / mg.realized_vol(wild, 20).iloc[-1], abs=cfg.band)
    assert e < 0.5


def test_small_changes_inside_the_band_are_ignored():
    cfg = mg.ManagedConfig(dip_boost=0.0, band=0.10)
    rng = np.random.default_rng(3)
    p = series(rng.normal(0, 0.012, 400))
    path = mg.exposure_path(p, cfg)
    moves = path["base"].diff().abs()
    assert (moves[moves > 0] > cfg.band - 1e-12).all()               # every change is bigger than the band
    assert (path["base"] - path["wanted"]).abs().iloc[40:].max() <= cfg.band + 1e-9


def test_dip_boost_switches_on_after_a_sharp_drop_in_an_uptrend_and_off_on_the_bounce():
    up = [0.002] * 260
    p = series(up + [-0.02, -0.02, 0.0, 0.03, 0.002])
    cfg = mg.ManagedConfig()
    dip = mg.dip_signal(p, cfg)
    assert not dip.iloc[260] and dip.iloc[262]                        # on after the two down days
    assert not dip.iloc[264]                                          # off once price is back above its 5-day average
    down = series([-0.002] * 260 + [-0.02, -0.02, 0.0])
    assert not mg.dip_signal(down, cfg).any()                         # never in a downtrend


def test_exposure_today_does_not_depend_on_tomorrow():
    rng = np.random.default_rng(5)
    p = series(rng.normal(0.0003, 0.012, 600))
    cfg = mg.ManagedConfig()
    full = mg.exposure_path(p, cfg)
    cut = mg.exposure_path(p.iloc[:450], cfg)
    pd.testing.assert_frame_equal(full.iloc[:450], cut)


def test_backtest_earns_exposure_times_return_minus_financing():
    cfg = mg.ManagedConfig(dip_boost=0.0)
    p = series([0.001] * 300)                                         # dead calm: exposure sits at the 2x cap
    bill = pd.Series(0.03, index=p.index)
    out = mg.backtest(p, bill, cfg, lag=2, cost_bps=0.0)
    last = out.iloc[-1]
    assert last["exposure"] == pytest.approx(2.0)
    assert last["ret"] == pytest.approx(2 * 0.001 - (0.03 + cfg.fund_cost) / 252)
