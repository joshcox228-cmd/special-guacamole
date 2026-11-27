#!/usr/bin/env python3
"""Usage:
    python run.py backtest [--refresh] [--offset N]   replay the strategy through history
    python run.py signals                             what the model says to hold right now
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from marketalgo import backtest as bt
from marketalgo import features as ft
from marketalgo import model as ml
from marketalgo import report
from marketalgo.config import Config
from marketalgo.data import load_prices

OUT = Path(__file__).resolve().parent / "outputs"


def prepare(cfg: Config, refresh: bool):
    prices = load_prices(cfg.tickers, cfg.start, refresh=refresh)
    wide = ft.build_wide_features(prices, cfg)
    return prices, wide, ft.to_long(wide)


def run_backtest(cfg: Config, refresh: bool, n_random: int) -> None:
    prices, wide, features = prepare(cfg, refresh)
    fwd = ft.forward_returns(prices, cfg)
    print(f"Prices: {prices.index[0].date()} to {prices.index[-1].date()}, "
          f"{len(cfg.universe)} ETFs. Training walk-forward models...")
    scores = ml.walk_forward(features, ft.build_labels(fwd), cfg, progress=True)

    trend_ok = (wide["dist_sma200"] > 0).loc[scores.index]
    eligible = scores.notna()
    momentum = wide["mom_12_1"].loc[scores.index].where(eligible)
    equal_cfg = replace(cfg, top_n=len(cfg.universe), trend_filter=False)

    results = {"ML top-5": bt.simulate(prices, bt.weights_from_scores(scores, trend_ok, cfg), cfg)}
    index = results["ML top-5"].index
    results["Momentum rule top-5"] = bt.simulate(
        prices, bt.weights_from_scores(momentum, trend_ok, cfg), cfg)
    results["Equal weight (all ETFs)"] = bt.simulate(
        prices, bt.weights_from_scores(eligible.astype(float).where(eligible), trend_ok, equal_cfg),
        equal_cfg)
    results["SPY buy & hold"] = bt.buy_and_hold(prices, cfg.benchmark, index)

    print(f"\nOut-of-sample period: {index[0].date()} to {index[-1].date()} "
          f"({len(scores)} monthly rebalances, {cfg.cost_bps:g} bps cost per trade)\n")
    table = report.metrics_table({k: bt.metrics(v) for k, v in results.items()})
    print(table)

    ic = bt.information_coefficient(scores, fwd)
    t_stat = ic.mean() / ic.std() * np.sqrt(len(ic))
    mom_ic = bt.information_coefficient(momentum, fwd)
    lines = [
        "",
        "Does the model actually predict? (rank correlation of forecast vs. next month)",
        f"  ML model      mean IC {ic.mean():+.3f}   t-stat {t_stat:+.2f}   "
        f"right direction in {(ic > 0).mean():.0%} of months",
        f"  Momentum rule mean IC {mom_ic.mean():+.3f}",
    ]
    if n_random:
        sharpes = bt.random_pick_sharpes(prices, eligible, trend_ok, cfg, n_random)
        mine = bt.metrics(results["ML top-5"])["Sharpe"]
        lines += [
            "",
            f"Versus {n_random} random-pick portfolios under identical rules:",
            f"  random Sharpe: median {np.median(sharpes):.2f}, "
            f"5th-95th percentile {np.percentile(sharpes, 5):.2f} to {np.percentile(sharpes, 95):.2f}",
            f"  ML Sharpe {mine:.2f} beats {(sharpes < mine).mean():.0%} of them",
        ]
    print("\n".join(lines))

    OUT.mkdir(exist_ok=True)
    suffix = f"_offset{cfg.offset}" if cfg.offset else ""
    (OUT / f"backtest_summary{suffix}.txt").write_text(table + "\n" + "\n".join(lines) + "\n")
    pd.DataFrame({k: v["equity"] for k, v in results.items()}).to_csv(OUT / f"equity{suffix}.csv")
    report.plot(results, OUT / f"equity_curve{suffix}.png")
    print(f"\nSaved chart and tables to {OUT}/")


def run_signals(cfg: Config) -> None:
    prices, wide, features = prepare(cfg, refresh=True)
    dates = prices.index
    labels = ft.build_labels(ft.forward_returns(prices, cfg)).stack()
    labels.index.names = ["date", "ticker"]
    pos = len(dates) - 1
    model = ml.fit(features, labels, ml.training_rows(dates, pos, cfg), cfg)
    scores = ml.predict(model, features, dates[pos])
    trend_ok = wide["dist_sma200"].iloc[pos] > 0
    weights = bt.target_weights(scores, trend_ok, cfg)

    momentum = wide["mom_12_1"].iloc[pos].reindex(scores.index)
    rule_weights = bt.target_weights(momentum, trend_ok, cfg)
    table = pd.DataFrame({
        "score": scores,
        "12-1m momentum": momentum,
        "above 200d avg": trend_ok.map({True: "yes", False: "no"}),
        "ML weight": weights.reindex(scores.index),
        "momentum-rule weight": rule_weights.reindex(scores.index),
    }).sort_values("score", ascending=False)
    table.insert(0, "rank", range(1, len(table) + 1))
    cash = weights[cfg.cash]

    shown = table.copy()
    shown["score"] = shown["score"].map("{:+.3f}".format)
    shown["12-1m momentum"] = shown["12-1m momentum"].map("{:+.1%}".format)
    for col in ("ML weight", "momentum-rule weight"):
        shown[col] = shown[col].map(lambda w: f"{w:.0%}" if w else "-")
    print(f"\nSignals as of the close on {dates[pos].date()} "
          f"(act at the next close, hold about a month)\n")
    print(shown.to_string())
    print(f"\nCash ({cfg.cash}): ML {cash:.0%}, momentum rule {rule_weights[cfg.cash]:.0%}")
    print("Score = predicted rank over the next month, -0.5 (worst) to +0.5 (best).")

    OUT.mkdir(exist_ok=True)
    path = OUT / f"signals_{dates[pos].date()}.csv"
    table.to_csv(path)
    print(f"Saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--refresh", action="store_true", help="re-download prices")
    b.add_argument("--offset", type=int, default=0, help="shift the rebalance day by N trading days")
    b.add_argument("--random", type=int, default=200, help="random portfolios to compare against")
    sub.add_parser("signals")
    args = parser.parse_args()

    if args.command == "backtest":
        run_backtest(Config(offset=args.offset), args.refresh, args.random)
    else:
        run_signals(Config())


if __name__ == "__main__":
    main()
