#!/usr/bin/env python3
"""Usage:
    python run.py signals [--account N] [--holdings "SPY:8,XLE:61"] [--cash N] [--side "GLD:3"] [--rebalance]
                                          what to hold right now, and the trades to get there
    python run.py backtest [--refresh] [--offset N] [--ml]   replay the strategy through history
    python run.py portfolio [--file portfolio.csv]           check holdings against written sell rules

The portfolio has three parts: S&P 500 and Nasdaq-100 exposure, each managed by volatility,
and a fund-rotation (momentum) rule. --ml adds the machine-learning model as a comparison (slow).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from marketalgo import backtest as bt
from marketalgo import features as ft
from marketalgo import managed as mg
from marketalgo import model as ml
from marketalgo import plan
from marketalgo import portfolio as pf
from marketalgo import report
from marketalgo.config import Config
from marketalgo.data import drop_unfinished_bar, fetch_history, load_bill_rate, load_prices

OUT = Path(__file__).resolve().parent / "outputs"
MAIN, ML = "Managed S&P + Nasdaq + fund rule", "ML top-5"
PREV, RULE = "Managed S&P + fund rule (previous)", "Fund rule alone"


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


def blend(parts: list[tuple[float, pd.Series]], rule: pd.DataFrame, rule_share: float,
          managed_turnover: pd.Series | None = None) -> pd.DataFrame:
    """Sleeves held side by side and kept at their shares of the account."""
    ret = rule_share * rule["ret"]
    for share, series in parts:
        ret = ret + share * series.reindex(rule.index).fillna(0.0)
    turnover = rule_share * rule["turnover"]
    if managed_turnover is not None:
        turnover = turnover + managed_turnover.reindex(rule.index).fillna(0.0)
    return pd.DataFrame({"ret": ret, "turnover": turnover, "equity": (1 + ret).cumprod()})


def run_backtest(cfg: Config, refresh: bool, n_random: int, with_ml: bool) -> None:
    prices, wide = prepare(cfg, refresh)
    rule_cfg = cfg
    fwd = ft.forward_returns(prices, cfg)
    signal_dates = prices.index[ml.rebalance_positions(len(prices), cfg)]
    print(f"Prices: {prices.index[0].date()} to {prices.index[-1].date()}, {len(cfg.universe)} ETFs.")

    momentum = momentum_scores(wide, signal_dates)
    eligible = momentum.notna()
    trend_ok = (wide["dist_sma200"] > 0).loc[signal_dates]
    equal_cfg = replace(rule_cfg, top_n=len(cfg.universe), trend_filter=False, leverage=1.0)

    rule = bt.simulate(prices, bt.weights_from_scores(momentum, trend_ok, rule_cfg), rule_cfg)
    index = rule.index
    bill = load_bill_rate(refresh)
    runs = {m.index: (share, mg.backtest(prices[m.index], bill, m)) for m, share in plan.sleeves(cfg)}
    parts = [(share, run["ret"]) for share, run in runs.values()]
    churn = sum(share * run["exposure"].diff().abs() for share, run in runs.values())
    results = {MAIN: blend(parts, rule, cfg.rule_weight, churn)}
    if "SPY" in runs:
        results[PREV] = blend([(0.5, runs["SPY"][1]["ret"])], rule, 0.5)
    results[RULE] = rule
    ic_lines = [ic_line("Momentum rule", bt.information_coefficient(momentum, fwd))]
    if with_ml:
        print("Training walk-forward ML models...")
        scores = ml.walk_forward(ft.to_long(wide), ft.build_labels(fwd), cfg, progress=True)
        results[ML] = bt.simulate(prices, bt.weights_from_scores(scores, trend_ok, rule_cfg), rule_cfg)
        ic_lines.append(ic_line("ML model", bt.information_coefficient(scores, fwd)))
    results["Equal weight (all ETFs)"] = bt.simulate(
        prices, bt.weights_from_scores(eligible.astype(float).where(eligible), trend_ok, equal_cfg),
        equal_cfg).reindex(index)
    results["SPY buy & hold"] = bt.buy_and_hold(prices, cfg.benchmark, index)

    print(f"\nPeriod: {index[0].date()} to {index[-1].date()}. Fund rule ({cfg.rule_weight:.0%} of the account): "
          f"{len(signal_dates)} reviews, one every {cfg.rebalance_every} trading days, {cfg.leverage:g}x on margin, "
          f"{cfg.cost_bps:g} bps per trade.")
    for m, share in plan.sleeves(cfg):
        print(f"Managed {m.index} ({share:.0%} of the account): target swing {m.target_vol:.0%} a year, cap {m.cap:g}x, "
              f"dip boost {m.dip_boost:g}x, average exposure {runs[m.index][1]['exposure'].reindex(index).mean():.2f}x.")
    table = report.metrics_table({k: bt.metrics(v) for k, v in results.items()})
    print("\n" + table)

    lines = ["", "Does the fund ranking predict? (rank correlation of score vs. next month)", *ic_lines]
    if n_random:
        sharpes = bt.random_pick_sharpes(prices, eligible, trend_ok, rule_cfg, n_random)
        lines += ["", f"Fund rule versus {n_random} random-pick portfolios under identical rules:",
                  f"  random Sharpe: median {np.median(sharpes):.2f}, 5th-95th percentile "
                  f"{np.percentile(sharpes, 5):.2f} to {np.percentile(sharpes, 95):.2f}"]
        for name in (RULE, ML):
            if name in results:
                mine = bt.metrics(results[name])["Sharpe"]
                lines.append(f"  {name}: Sharpe {mine:.2f} beats {(sharpes < mine).mean():.0%} of them")
    print("\n".join(lines))

    OUT.mkdir(exist_ok=True)
    suffix = f"_offset{cfg.offset}" if cfg.offset else ""
    (OUT / f"backtest_summary{suffix}.txt").write_text(table + "\n" + "\n".join(lines) + "\n")
    pd.DataFrame({k: v["equity"] for k, v in results.items()}).to_csv(OUT / f"equity{suffix}.csv")
    charted = {k: v for k, v in results.items() if k in report.COLORS}
    report.plot(charted, OUT / f"equity_curve{suffix}.png", main=MAIN)
    print(f"\nSaved chart and tables to {OUT}/")


def parse_positions(text: str | None) -> dict[str, float]:
    """'SPY:8, XLE:61' -> {'SPY': 8.0, 'XLE': 61.0}"""
    out = {}
    for item in (text or "").replace(";", ",").split(","):
        if item.strip():
            ticker, _, shares = item.strip().partition(":")
            out[ticker.strip().upper()] = float(shares)
    return out


def latest_price(ticker: str) -> float:
    start = (pd.Timestamp.today() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    return float(drop_unfinished_bar(fetch_history(ticker, start))["close"].iloc[-1])


def run_signals(cfg: Config, with_ml: bool, account: float | None, holdings: dict[str, float],
                cash: float | None, side: dict[str, float], rebalance: bool = False) -> None:
    prices, wide = prepare(cfg, refresh=True)
    today, previous = prices.index[-1], prices.index[-2]
    print(f"\nPLAN CHECK as of the close on {today.date()}")

    # ---- sleeves managed by volatility
    exposures, changed, managed_funds = {}, [], set()
    for n, (m, share) in enumerate(plan.sleeves(cfg), start=1):
        path = mg.exposure_path(prices[m.index], m)
        now, before = path.iloc[-1], path.iloc[-2]
        exposures[m.index] = now["exposure"]
        managed_funds |= {m.index, m.fund2x}
        moved = abs(now["exposure"] - before["exposure"]) > 1e-9
        if moved:
            changed.append(m.index)
        parts = mg.split(now["exposure"])
        print(f"\n{n}) Managed {m.index} sleeve ({share:.0%} of the account)")
        print(f"   Market swings over the last {m.lookback} days: {now['vol']:.1%} a year (target {m.target_vol:.0%}) "
              f"-> the rule calls for {now['wanted']:.2f}x; holding {now['base']:.2f}x")
        print(f"   Dip-buying boost: {'ON (+%.1fx)' % m.dip_boost if now['dip'] else 'off'}")
        print(f"   Exposure now {now['exposure']:.2f}x; previous day {before['exposure']:.2f}x -> "
              f"{'CHANGED' if moved else 'no change'}")
        print(f"   = {m.index} {share * parts['index']:.1%}, {m.fund2x} {share * parts['fund2x']:.1%}, "
              f"cash {share * parts['cash']:.1%} of the account")
        print("   Last trading days: " + ", ".join(f"{d:%b %d} {e:.2f}x" for d, e in path["exposure"].iloc[-6:].items()))

    # ---- fund rule
    momentum = wide["mom_12_1"].loc[today].dropna()
    trend_ok = wide["dist_sma200"].loc[today] > 0
    rule_weights = bt.target_weights(momentum, trend_ok, cfg)
    review = plan.review_due(today, previous, pd.Timestamp(cfg.review_anchor))
    table = pd.DataFrame({
        "12-1m momentum": momentum,
        "vs 200d avg": wide["dist_sma200"].loc[today].reindex(momentum.index),
        "uptrend": trend_ok.reindex(momentum.index).map({True: "yes", False: "no"}),
        "share of account": (rule_weights * cfg.rule_weight).reindex(momentum.index),
    }).sort_values("12-1m momentum", ascending=False)
    table.insert(0, "rank", range(1, len(table) + 1))
    if with_ml:
        features = ft.to_long(wide)
        labels = ft.build_labels(ft.forward_returns(prices, cfg)).stack()
        labels.index.names = ["date", "ticker"]
        model = ml.fit(features, labels, ml.training_rows(prices.index, len(prices) - 1, cfg), cfg)
        table["ML score"] = ml.predict(model, features, today).reindex(table.index)
    shown = table.head(8).copy()
    for col in ("12-1m momentum", "vs 200d avg"):
        shown[col] = shown[col].map("{:+.1%}".format)
    shown["share of account"] = shown["share of account"].map(lambda w: f"{w:.1%}" if w else "-")
    if "ML score" in shown:
        shown["ML score"] = shown["ML score"].map("{:+.3f}".format)
    print(f"\n{len(cfg.managed) + 1}) Fund rule ({cfg.rule_weight:.0%} of the account), reviewed every two weeks "
          f"-> review {'DUE TODAY' if review else 'not due today'}")
    print("   " + shown.to_string().replace("\n", "\n   "))

    weights = plan.account_weights(rule_weights, exposures, cfg)
    held = {t: w for t, w in weights.items() if t != cfg.cash and abs(w) > 1e-9}
    cash_w = weights[cfg.cash]
    print("\nWhole-account targets: " + ", ".join(f"{t} {w:.1%}" for t, w in held.items())
          + (f", cash {cash_w:.1%}" if cash_w >= 0 else f", BORROWED {-cash_w:.1%}"))

    # ---- orders
    tickers = sorted(set(held) | set(holdings) | set(side) | managed_funds)
    px = pd.Series({t: (float(prices[t].iloc[-1]) if t in prices.columns else latest_price(t)) for t in tickers})
    side_value = sum(px[t] * n for t, n in side.items())
    if holdings or cash is not None:
        account = sum(px[t] * n for t, n in holdings.items()) + side_value + (cash or 0.0)
    acting = bool(changed) or review
    if account:
        frozen: set[str] = set()
        if holdings and not rebalance:
            print()
            for m, share in plan.sleeves(cfg):
                have = plan.exposure_held(holdings, px, account, m, share,
                                          rule_weights.get(m.index, 0.0) * cfg.rule_weight)
                in_band = abs(have - exposures[m.index]) <= m.band
                print(f"{m.index} sleeve held now: {have:.2f}x against a target of {exposures[m.index]:.2f}x -> "
                      f"{'close enough, leave it' if in_band else 'ADJUST'}")
                if in_band and not review:
                    frozen |= {m.index, m.fund2x}
            if not review:                       # fund-rule positions only change on review days
                frozen |= {t for t in tickers if t not in managed_funds}
        book = plan.orders(weights, px, account, holdings, cash_ticker=cfg.cash, frozen=frozen,
                           min_trade=0.005 if rebalance else 0.02)
        book = book[(book["target"] > 0) | (book["held"] > 0)]
        print(f"\nAccount {account:,.0f}" + (f" (includes side bets left alone: "
              f"{', '.join(f'{t} {n:g}' for t, n in side.items())}, worth {side_value:,.0f})" if side else ""))
        out = pd.DataFrame({"price": book["price"].map("{:,.2f}".format), "target shares": book["target"].astype(int),
                            "held": book["held"].map("{:g}".format),
                            "TRADE": book["trade"].map(lambda v: "-" if v == 0 else f"{'BUY' if v > 0 else 'SELL'} {abs(v):g}"),
                            "value after": book["value"].map("{:,.0f}".format)})
        print(out.to_string())
        left = account - side_value - book["value"].sum()
        print(f"   {'Cash after trades' if left >= 0 else 'BORROWED after trades'}: {abs(left):,.0f}")
        if holdings:
            acting = bool((book["trade"] != 0).any())
    print(f"\nACTION NEEDED: {'YES' if acting else 'no'}"
          + ("" if holdings else "  (based on changes since the previous day; pass --holdings for exact trades)"))
    print("Managed sleeves change on any day; fund-rule positions only on review days"
          + (" (full rebalance requested: everything is lined up with the targets)." if rebalance else "."))

    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"signals_{today.date()}.csv"
    table.to_csv(out_path)
    print(f"Saved to {out_path}")


def run_portfolio(path: Path, cfg: Config) -> None:
    if not path.exists():
        raise SystemExit(f"{path} not found. Copy portfolio.example.csv to portfolio.csv and edit it.")
    book = pf.load_portfolio(path)
    start = (pd.Timestamp.today() - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    history = {t: drop_unfinished_bar(fetch_history(t, start)) for t in [*book.index, cfg.benchmark]}
    bench = history.pop(cfg.benchmark)["close"]
    rep = pf.review(book, history)
    as_of = rep["as_of"].max().date()

    def pct(v, signed=True):
        return "-" if pd.isna(v) else (f"{v:+.1%}" if signed else f"{v:.1%}")

    print(f"\nPortfolio check as of the close on {as_of} (flag-only: this reports what your rules say)")
    market_up = bench.iloc[-1] > bench.rolling(200).mean().iloc[-1]
    print(f"Market: {cfg.benchmark} is {'above' if market_up else 'BELOW'} its 200-day average.\n")

    for status, title in (("EXIT", "ACTIONS - stop breached, rule says exit fully"),
                          ("TRIM", "ACTIONS - trim level reached"),
                          ("NO STOP", "NEEDS A STOP"),
                          ("WATCH", "WATCH - within 3% of the stop")):
        rows = rep[rep["status"] == status]
        if len(rows):
            print(title)
            for t, r in rows.iterrows():
                extra = f" Suggested stop {r['ema_stop']:.2f} ({r['ema_note']})." \
                    if status == "NO STOP" and pd.notna(r["ema_stop"]) else ""
                print(f"  {t:<5} {r['price']:>9.2f}  {r['reason']}.{extra}")
            print()
    conflicts = pf.trend_conflicts(rep)
    if len(conflicts):
        print("TREND WARNING - no rule has fired, but below the 200-day average with negative momentum")
        for t, r in conflicts.iterrows():
            print(f"  {t:<5} {r['price']:>9.2f}  12-1m momentum {pct(r['momentum'])}, {r['ema_note']}")
        print()
    quiet = rep[(rep["status"] == "QUIET") & ~rep.index.isin(conflicts.index)]
    print(f"QUIET ({len(quiet)}): {', '.join(quiet.index) or 'none'}\n")

    table = pd.DataFrame({
        "book": rep["book"], "status": rep["status"],
        "price": rep["price"].map("{:.2f}".format),
        "gain": rep["gain"].map(pct), "weight": rep["weight"].map(lambda v: pct(v, False)),
        "stop": rep["stop"].map(lambda v: "-" if pd.isna(v) else f"{v:g}"),
        "to stop": rep["to_stop"].map(pct),
        "EMA stop": rep["ema_stop"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}"),
        "uptrend": rep["uptrend"].map({True: "yes", False: "no"}),
        "12-1m mom": rep["momentum"].map(pct),
    }).sort_values(["book", "status"])
    print(table.to_string())

    total = rep["value"].sum()
    sleeve = rep.loc[rep["book"] == "satellite", "value"].sum() / total
    print(f"\nEquity value {total:,.0f}. Satellite sleeve {sleeve:.0%} of equity "
          f"({'over' if sleeve > 0.15 else 'within'} the 15% cap; cash not included).")
    print("EMA stop = 1.25% under the nearest of the 50/88/200-day EMAs below price. "
          "Thesis-break triggers are not checked here.")

    OUT.mkdir(exist_ok=True)
    out = OUT / f"portfolio_{as_of}.csv"
    rep.to_csv(out)
    print(f"Saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--refresh", action="store_true", help="re-download prices")
    b.add_argument("--offset", type=int, default=0, help="shift the rebalance day by N trading days")
    b.add_argument("--random", type=int, default=200, help="random portfolios to compare against")
    s = sub.add_parser("signals")
    s.add_argument("--account", type=float, help="account value, to print share counts")
    s.add_argument("--holdings", help='plan holdings as "SPY:8,XLE:61,XLB:73"; gives exact trades')
    s.add_argument("--cash", type=float, help="cash balance (negative if borrowed), used with --holdings")
    s.add_argument("--side", help='positions held outside the plan, e.g. "GLD:3"; they are left alone')
    s.add_argument("--rebalance", action="store_true", help="line every holding up with its target today")
    for p in (b, s):
        p.add_argument("--ml", action="store_true", help="also run the ML model for comparison")
    pfp = sub.add_parser("portfolio")
    pfp.add_argument("--file", type=Path, default=Path(__file__).resolve().parent / "portfolio.csv")
    args = parser.parse_args()

    if args.command == "backtest":
        run_backtest(Config(offset=args.offset), args.refresh, args.random, args.ml)
    elif args.command == "signals":
        run_signals(Config(), args.ml, args.account, parse_positions(args.holdings), args.cash,
                    parse_positions(args.side), args.rebalance)
    else:
        run_portfolio(args.file, Config())


if __name__ == "__main__":
    main()
