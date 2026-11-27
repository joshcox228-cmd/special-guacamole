#!/usr/bin/env python3
"""Usage:
    python run.py backtest [--refresh] [--offset N] [--ml]   replay the strategy through history
    python run.py signals [--ml]                             what to hold right now

The main strategy is the momentum rule. --ml adds the machine-learning model
as a comparison (slower: it retrains at every monthly rebalance).
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
MAIN, ML = "Momentum rule top-5", "ML top-5"


def prepare(cfg: Config, refresh: bool):
    prices = load_prices(cfg.tickers, cfg.start, refresh=refresh)
    return prices, ft.build_wide_features(prices, cfg)


def momentum_scores(wide: dict, dates) -> pd.DataFrame:
    """The whole rule: rank by the return from 12 months ago to 1 month ago."""
    return wide["mom_12_1"].loc[dates]


def ic_line(name: str, ic: pd.Series) -> str:
    t_stat = ic.mean() / ic.std() * np.sqrt(len(ic))
    return (f"  {name:<14} mean IC {ic.mean():+.3f}   t-stat {t_stat:+.2f}   "
            f"right direction in {(ic > 0).mean():.0%} of months")


def run_backtest(cfg: Config, refresh: bool, n_random: int, with_ml: bool) -> None:
    prices, wide = prepare(cfg, refresh)
    fwd = ft.forward_returns(prices, cfg)
    signal_dates = prices.index[ml.rebalance_positions(len(prices), cfg)]
    print(f"Prices: {prices.index[0].date()} to {prices.index[-1].date()}, {len(cfg.universe)} ETFs.")

    momentum = momentum_scores(wide, signal_dates)
    eligible = momentum.notna()
    trend_ok = (wide["dist_sma200"] > 0).loc[signal_dates]
    equal_cfg = replace(cfg, top_n=len(cfg.universe), trend_filter=False)

    results = {MAIN: bt.simulate(prices, bt.weights_from_scores(momentum, trend_ok, cfg), cfg)}
    index = results[MAIN].index
    ic_lines = [ic_line("Momentum rule", bt.information_coefficient(momentum, fwd))]
    if with_ml:
        print("Training walk-forward ML models...")
        scores = ml.walk_forward(ft.to_long(wide), ft.build_labels(fwd), cfg, progress=True)
        results[ML] = bt.simulate(prices, bt.weights_from_scores(scores, trend_ok, cfg), cfg)
        ic_lines.append(ic_line("ML model", bt.information_coefficient(scores, fwd)))
    results["Equal weight (all ETFs)"] = bt.simulate(
        prices, bt.weights_from_scores(eligible.astype(float).where(eligible), trend_ok, equal_cfg),
        equal_cfg)
    results["SPY buy & hold"] = bt.buy_and_hold(prices, cfg.benchmark, index)

    print(f"\nPeriod: {index[0].date()} to {index[-1].date()} "
          f"({len(signal_dates)} monthly rebalances, {cfg.cost_bps:g} bps cost per trade)\n")
    table = report.metrics_table({k: bt.metrics(v) for k, v in results.items()})
    print(table)

    lines = ["", "Does the ranking predict? (rank correlation of score vs. next month)", *ic_lines]
    if n_random:
        sharpes = bt.random_pick_sharpes(prices, eligible, trend_ok, cfg, n_random)
        lines += ["", f"Versus {n_random} random-pick portfolios under identical rules:",
                  f"  random Sharpe: median {np.median(sharpes):.2f}, 5th-95th percentile "
                  f"{np.percentile(sharpes, 5):.2f} to {np.percentile(sharpes, 95):.2f}"]
        for name in (MAIN, ML):
            if name in results:
                mine = bt.metrics(results[name])["Sharpe"]
                lines.append(f"  {name}: Sharpe {mine:.2f} beats {(sharpes < mine).mean():.0%} of them")
    print("\n".join(lines))

    OUT.mkdir(exist_ok=True)
    suffix = f"_offset{cfg.offset}" if cfg.offset else ""
    (OUT / f"backtest_summary{suffix}.txt").write_text(table + "\n" + "\n".join(lines) + "\n")
    pd.DataFrame({k: v["equity"] for k, v in results.items()}).to_csv(OUT / f"equity{suffix}.csv")
    report.plot(results, OUT / f"equity_curve{suffix}.png", main=MAIN)
    print(f"\nSaved chart and tables to {OUT}/")


def run_signals(cfg: Config, with_ml: bool) -> None:
    prices, wide = prepare(cfg, refresh=True)
    today = prices.index[-1]
    momentum = wide["mom_12_1"].loc[today].dropna()
    trend_ok = wide["dist_sma200"].loc[today] > 0
    weights = bt.target_weights(momentum, trend_ok, cfg)

    table = pd.DataFrame({
        "12-1m momentum": momentum,
        "vs 200d avg": wide["dist_sma200"].loc[today].reindex(momentum.index),
        "uptrend": trend_ok.reindex(momentum.index).map({True: "yes", False: "no"}),
        "target weight": weights.reindex(momentum.index),
    }).sort_values("12-1m momentum", ascending=False)
    table.insert(0, "rank", range(1, len(table) + 1))

    if with_ml:
        features = ft.to_long(wide)
        labels = ft.build_labels(ft.forward_returns(prices, cfg)).stack()
        labels.index.names = ["date", "ticker"]
        pos = len(prices) - 1
        model = ml.fit(features, labels, ml.training_rows(prices.index, pos, cfg), cfg)
        scores = ml.predict(model, features, today)
        table["ML score"] = scores.reindex(table.index)
        table["ML weight"] = bt.target_weights(scores, trend_ok, cfg).reindex(table.index)

    shown = table.copy()
    for col in ("12-1m momentum", "vs 200d avg"):
        shown[col] = shown[col].map("{:+.1%}".format)
    for col in ("target weight", "ML weight"):
        if col in shown:
            shown[col] = shown[col].map(lambda w: f"{w:.0%}" if w else "-")
    if "ML score" in shown:
        shown["ML score"] = shown["ML score"].map("{:+.3f}".format)

    print(f"\nMomentum-rule signals as of the close on {today.date()} "
          f"(act at the next close, re-check in about a month)\n")
    print(shown.to_string())
    holdings = weights[weights > 0]
    print("\nTarget portfolio: " + ", ".join(f"{t} {w:.0%}" for t, w in holdings.items()))
    print(f"Rule: top {cfg.top_n} by 12-1 month momentum, {1 / cfg.top_n:.0%} each; "
          f"a pick below its 200-day average is held as {cfg.cash} (cash) instead.")

    OUT.mkdir(exist_ok=True)
    path = OUT / f"signals_{today.date()}.csv"
    table.to_csv(path)
    print(f"Saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--refresh", action="store_true", help="re-download prices")
    b.add_argument("--offset", type=int, default=0, help="shift the rebalance day by N trading days")
    b.add_argument("--random", type=int, default=200, help="random portfolios to compare against")
    s = sub.add_parser("signals")
    for p in (b, s):
        p.add_argument("--ml", action="store_true", help="also run the ML model for comparison")
    args = parser.parse_args()

    if args.command == "backtest":
        run_backtest(Config(offset=args.offset), args.refresh, args.random, args.ml)
    else:
        run_signals(Config(), args.ml)


if __name__ == "__main__":
    main()
