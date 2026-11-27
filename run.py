#!/usr/bin/env python3
"""Usage:
    python run.py signals [--account N] [--holdings "SPY:8,XLE:61"] [--cash N] [--side "QQQ:3"]
                                          what to hold right now, and the trades to get there
    python run.py backtest [--refresh] [--offset N] [--ml]   replay the strategy through history
    python run.py portfolio [--file portfolio.csv]           check holdings against written sell rules

The portfolio has two halves: S&P 500 exposure managed by volatility, and a fund-rotation
(momentum) rule. --ml adds the machine-learning model as a comparison (slow).
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
MAIN, ML = "Managed S&P + fund rule", "ML top-5"
PLAIN, RULE = "Plain S&P + fund rule", "Fund rule alone"


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


def blend(core: pd.Series, rule: pd.DataFrame, share: float, core_turnover: pd.Series | None = None) -> pd.DataFrame:
    """Two halves held side by side and kept at their shares of the account."""
    core = core.reindex(rule.index).fillna(0.0)
    ret = share * core + (1 - share) * rule["ret"]
    turnover = (1 - share) * rule["turnover"]
    if core_turnover is not None:
        turnover = turnover + share * core_turnover.reindex(rule.index).fillna(0.0)
    return pd.DataFrame({"ret": ret, "turnover": turnover, "equity": (1 + ret).cumprod()})


def run_backtest(cfg: Config, refresh: bool, n_random: int, with_ml: bool) -> None:
    prices, wide = prepare(cfg, refresh)
    mcfg = mg.ManagedConfig()
    rule_cfg = replace(cfg, core_weight=0.0)
    fwd = ft.forward_returns(prices, cfg)
    signal_dates = prices.index[ml.rebalance_positions(len(prices), cfg)]
    print(f"Prices: {prices.index[0].date()} to {prices.index[-1].date()}, {len(cfg.universe)} ETFs.")

    momentum = momentum_scores(wide, signal_dates)
    eligible = momentum.notna()
    trend_ok = (wide["dist_sma200"] > 0).loc[signal_dates]
    equal_cfg = replace(rule_cfg, top_n=len(cfg.universe), trend_filter=False, leverage=1.0)

    rule = bt.simulate(prices, bt.weights_from_scores(momentum, trend_ok, rule_cfg), rule_cfg)
    index = rule.index
    managed = mg.backtest(prices[cfg.core], load_bill_rate(refresh), mcfg)
    plain = prices[cfg.core].pct_change(fill_method=None)
    results = {
        MAIN: blend(managed["ret"], rule, cfg.core_weight, managed["exposure"].diff().abs()),
        PLAIN: blend(plain, rule, cfg.core_weight),
        RULE: rule,
    }
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

    print(f"\nPeriod: {index[0].date()} to {index[-1].date()}. Fund rule: {len(signal_dates)} reviews, one every "
          f"{cfg.rebalance_every} trading days, {cfg.leverage:g}x on margin, {cfg.cost_bps:g} bps per trade.\n"
          f"Managed S&P: target swing {mcfg.target_vol:.0%} a year, cap {mcfg.cap:g}x, dip boost {mcfg.dip_boost:g}x, "
          f"average exposure {managed['exposure'].reindex(index).mean():.2f}x.\n")
    table = report.metrics_table({k: bt.metrics(v) for k, v in results.items()})
    print(table)

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
                cash: float | None, side: dict[str, float]) -> None:
    prices, wide = prepare(cfg, refresh=True)
    mcfg = mg.ManagedConfig()
    rule_cfg = replace(cfg, core_weight=0.0)
    today, previous = prices.index[-1], prices.index[-2]

    # ---- S&P 500 half
    path = mg.exposure_path(prices[cfg.core], mcfg)
    now, before = path.iloc[-1], path.iloc[-2]
    core_changed = abs(now["exposure"] - before["exposure"]) > 1e-9
    parts = mg.split(now["exposure"])

    # ---- fund-rule half
    momentum = wide["mom_12_1"].loc[today].dropna()
    trend_ok = wide["dist_sma200"].loc[today] > 0
    rule_weights = bt.target_weights(momentum, trend_ok, rule_cfg)
    review = plan.review_due(today, previous, pd.Timestamp(cfg.review_anchor))
    table = pd.DataFrame({
        "12-1m momentum": momentum,
        "vs 200d avg": wide["dist_sma200"].loc[today].reindex(momentum.index),
        "uptrend": trend_ok.reindex(momentum.index).map({True: "yes", False: "no"}),
        "share of account": (rule_weights * (1 - cfg.core_weight)).reindex(momentum.index),
    }).sort_values("12-1m momentum", ascending=False)
    table.insert(0, "rank", range(1, len(table) + 1))
    if with_ml:
        features = ft.to_long(wide)
        labels = ft.build_labels(ft.forward_returns(prices, cfg)).stack()
        labels.index.names = ["date", "ticker"]
        model = ml.fit(features, labels, ml.training_rows(prices.index, len(prices) - 1, cfg), cfg)
        table["ML score"] = ml.predict(model, features, today).reindex(table.index)

    weights = plan.account_weights(rule_weights, now["exposure"], cfg, mcfg)

    print(f"\nPLAN CHECK as of the close on {today.date()}")
    print(f"\n1) S&P 500 half ({cfg.core_weight:.0%} of the account)")
    print(f"   Market swings over the last {mcfg.lookback} days: {now['vol']:.1%} a year (target {mcfg.target_vol:.0%}) "
          f"-> the rule calls for {now['wanted']:.2f}x; holding {now['base']:.2f}x")
    print(f"   Dip-buying boost: {'ON (+%.1fx)' % mcfg.dip_boost if now['dip'] else 'off'}")
    print(f"   Exposure now {now['exposure']:.2f}x; previous day {before['exposure']:.2f}x -> "
          f"{'CHANGED' if core_changed else 'no change'}")
    print(f"   = {mcfg.index} {cfg.core_weight * parts['index']:.1%}, {mcfg.fund2x} {cfg.core_weight * parts['fund2x']:.1%}, "
          f"cash {cfg.core_weight * parts['cash']:.1%} of the account")

    shown = table.head(8).copy()
    for col in ("12-1m momentum", "vs 200d avg"):
        shown[col] = shown[col].map("{:+.1%}".format)
    shown["share of account"] = shown["share of account"].map(lambda w: f"{w:.0%}" if w else "-")
    if "ML score" in shown:
        shown["ML score"] = shown["ML score"].map("{:+.3f}".format)
    print(f"\n2) Fund-rule half ({1 - cfg.core_weight:.0%} of the account), reviewed every two weeks "
          f"-> review {'DUE TODAY' if review else 'not due today'}")
    print("   " + shown.to_string().replace("\n", "\n   "))

    held = {t: w for t, w in weights.items() if t != cfg.cash and abs(w) > 1e-9}
    cash_w = weights[cfg.cash]
    print("\nWhole-account targets: " + ", ".join(f"{t} {w:.1%}" for t, w in held.items())
          + (f", cash {cash_w:.1%}" if cash_w >= 0 else f", BORROWED {-cash_w:.1%}"))

    # ---- orders
    core_funds = {mcfg.index, mcfg.fund2x}
    tickers = sorted(set(held) | set(holdings) | set(side) | core_funds)
    px = pd.Series({t: (float(prices[t].iloc[-1]) if t in prices.columns else latest_price(t)) for t in tickers})
    side_value = sum(px[t] * n for t, n in side.items())
    if holdings or cash is not None:
        account = sum(px[t] * n for t, n in holdings.items()) + side_value + (cash or 0.0)
    acting = core_changed or review
    if account:
        frozen: set[str] = set()
        if holdings:
            have = plan.core_exposure_held(holdings, px, account, cfg, mcfg)
            in_band = abs(have - now["exposure"]) <= mcfg.band
            print(f"\nS&P half held now: {have:.2f}x against a target of {now['exposure']:.2f}x -> "
                  f"{'close enough, leave it' if in_band else 'ADJUST'}")
            if in_band:
                frozen |= core_funds
            if not review:                       # fund-rule positions only change on review days
                frozen |= {t for t in tickers if t not in core_funds}
        book = plan.orders(weights, px, account, holdings, cash_ticker=cfg.cash, frozen=frozen)
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
    print("S&P-half changes apply on any day; fund-rule changes only on review days.")

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
    s.add_argument("--side", help='positions held outside the plan, e.g. "QQQ:3"; they are left alone')
    for p in (b, s):
        p.add_argument("--ml", action="store_true", help="also run the ML model for comparison")
    pfp = sub.add_parser("portfolio")
    pfp.add_argument("--file", type=Path, default=Path(__file__).resolve().parent / "portfolio.csv")
    args = parser.parse_args()

    if args.command == "backtest":
        run_backtest(Config(offset=args.offset), args.refresh, args.random, args.ml)
    elif args.command == "signals":
        run_signals(Config(), args.ml, args.account, parse_positions(args.holdings), args.cash,
                    parse_positions(args.side))
    else:
        run_portfolio(args.file, Config())


if __name__ == "__main__":
    main()
