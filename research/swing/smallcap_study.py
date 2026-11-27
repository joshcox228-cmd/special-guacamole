"""Small-company research: do insider purchases or earnings surprises predict returns in smaller stocks?

Rules fixed before any result was seen:

  Stock list ("small"): ranks 501-2500 by 60-day median dollar volume, with at least $2M traded a day.
  Costs: 20 bps per trade (stress test: 40). Fills at the next open. 20 positions of 5% each.
  Test window: 2016-07-01 onward (delisted companies are well covered from 2016).

  E1  Good earnings surprise: earnings beat the estimate AND the stock beat the S&P 500 by 5% or more over the
      report day and the day after. Buy at the open two days after the report. Hold 60 trading days.
  I1  Insider cluster: within 30 days, at least 2 different officers or directors report open-market purchases
      totalling $100,000 or more. Buy at the open after the filing that completes the cluster. Hold 120 trading days.
  I2  Large single purchase: one officer or director reports buying $250,000 or more. Same entry and hold.

Usage: python smallcap_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import Data, STOCK_DIR, TRADING_DAYS, clean_data, liquid_universe, nw_tstat, simulate, trade_stats

pd.set_option("display.width", 250)
SC = STOCK_DIR / "smallcap"
FIELDS = ("open", "high", "low", "close", "adj", "volume")
BEGIN, MID = pd.Timestamp("2016-07-01"), pd.Timestamp("2021-07-01")
COST = 20.0


def load_small() -> Data:
    frames = {f: pd.read_parquet(SC / f"{f}.parquet").astype("float64") for f in FIELDS}
    for f in ("open", "high", "low", "close", "adj"):
        frames[f] = frames[f].where(frames[f] > 0)
    return clean_data(Data(pool=pd.read_csv(SC / "pool.csv", index_col=0), **frames))


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {"ret": np.nan, "dd": np.nan, "sharpe": np.nan}
    eq = (1 + r).cumprod()
    return {"ret": eq.iloc[-1] ** (TRADING_DAYS / len(r)) - 1, "dd": (eq / eq.cummax() - 1).min(),
            "sharpe": r.mean() * TRADING_DAYS / (r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan}


def periods(r: pd.Series) -> dict:
    a, b, c = perf(r.loc[BEGIN:MID]), perf(r.loc[MID:]), perf(r.loc[BEGIN:])
    return {"2016-21": a["ret"], "2021-26": b["ret"], "All": c["ret"], "worst drop": c["dd"], "Sharpe": c["sharpe"]}


def flags_from(events: pd.DataFrame, dates: pd.DatetimeIndex, cols: list[str], offset: int = 0):
    """events: columns symbol, date, score. Marks the first trading day on/after `date` (+offset days)."""
    col = {c: k for k, c in enumerate(cols)}
    ev = events[events.symbol.isin(col)]
    i = dates.searchsorted(ev.date.to_numpy()) + offset
    ok = i < len(dates)
    j = ev.symbol.map(col).to_numpy()[ok]; i = i[ok]
    flag = np.zeros((len(dates), len(cols)), bool); score = np.full((len(dates), len(cols)), np.nan)
    flag[i, j] = True; score[i, j] = ev.score.to_numpy()[ok]
    return pd.DataFrame(flag, index=dates, columns=cols), pd.DataFrame(score, index=dates, columns=cols)


def event_table(data: Data, entry: pd.DataFrame, peers: pd.DataFrame, horizons) -> pd.DataFrame:
    cols = list(entry.columns)
    ao = data.adj_open[cols]
    rows = []
    for k in horizons:
        fwd = ao.shift(-(k + 1)) / ao.shift(-1) - 1
        abn = fwd.sub(fwd.where(peers).mean(axis=1), axis=0).where(entry)
        row = {"days held": k}
        for label, sl in (("2016-21", slice(BEGIN, MID)), ("2021-26", slice(MID, None)), ("All", slice(BEGIN, None))):
            raw = fwd.where(entry).loc[sl].stack().dropna(); ab = abn.loc[sl].stack().dropna()
            row[f"{label} n"] = len(raw); row[f"{label} avg %"] = 100 * raw.mean(); row[f"{label} vs peers %"] = 100 * ab.mean()
            row[f"{label} t"] = nw_tstat(abn.loc[sl].mean(axis=1).dropna().to_numpy(), lags=min(k, 20))
        row["rose"] = (fwd.where(entry).loc[BEGIN:].stack().dropna() > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("days held").round(2)


def traded(data: Data, entry, score, hold, bench: dict, cost=COST, slots=20) -> dict:
    daily, tr = simulate(data, entry, score, None, slots=slots, max_hold=hold, cost_bps=cost, start=BEGIN)
    out = periods(daily["ret"]); t = trade_stats(tr)
    out.update({"invested": daily["invested"].mean(), "win rate": t.get("Win rate", np.nan), "avg trade %": 100 * t.get("Avg trade", np.nan),
                "worst trade %": 100 * t.get("Worst", np.nan), "trades/yr": t.get("Trades/yr", 0)})
    r = daily["ret"]; b = bench["pool"].reindex(r.index).fillna(0.0)
    beta = np.cov(r, b)[0, 1] / b.var(); resid = r - beta * b
    out["alpha vs small stocks %/yr"] = 100 * resid.mean() * TRADING_DAYS; out["alpha t"] = nw_tstat(resid.to_numpy())
    return out


def insider_events(ins: pd.DataFrame, min_people: int, min_value: float, window_days: int = 30, cool_off: int = 60) -> pd.DataFrame:
    """One row per signal: the filing date on which the rule first becomes true for a company."""
    out = []
    for sym, g in ins.groupby("symbol", sort=False):
        g = g.sort_values("filingDate")
        dates = g.filingDate.to_numpy(); who = g.reportingCik.to_numpy(); val = g.value.to_numpy()
        last = None; start = 0
        for k in range(len(g)):
            while dates[k] - dates[start] > np.timedelta64(window_days, "D"):
                start += 1
            people = len(set(who[start:k + 1])); total = val[start:k + 1].sum()
            if people >= min_people and total >= min_value:
                if last is None or dates[k] - last > np.timedelta64(cool_off, "D"):
                    out.append((sym, dates[k], -total)); last = dates[k]
    return pd.DataFrame(out, columns=["symbol", "date", "score"])


def main() -> None:
    data = load_small()
    cols = data.stocks(); dates = data.dates
    adj = data.adj[cols]
    dollars = (data.close[cols] * data.volume[cols]).rolling(60, min_periods=40).median()
    small = liquid_universe(data, 2500, skip_top=500)[cols] & (dollars >= 2e6)
    big = liquid_universe(data, 500)[cols]
    ret = adj.pct_change(fill_method=None)
    bench = {"pool": ret.where(small.shift(1, fill_value=False)).mean(axis=1).fillna(0.0)}
    for t in ("SPY", "IWM"):
        bench[t] = data.adj[t].pct_change(fill_method=None).fillna(0.0)
    src = data.pool["source"].reindex(cols)
    in_small = small.loc[BEGIN:]
    print(f"{len(cols)} companies ({(src == 'delisted').sum()} delisted). Small list averages {int(in_small.sum(axis=1).mean())} stocks a day; "
          f"{in_small.loc[:, (src == 'delisted').to_numpy()].sum().sum() / in_small.sum().sum():.0%} of its stock-days are companies that later delisted.")
    print("\n=== Benchmarks over the test window ===")
    print(pd.DataFrame({"S&P 500 (SPY)": periods(bench["SPY"]), "Russell 2000 small stocks (IWM)": periods(bench["IWM"]),
                        "all stocks on the small list, equal amounts": periods(bench["pool"])}).T.round(3).to_string())

    spy = data.adj["SPY"].to_numpy(); A = adj.to_numpy(); col = {c: k for k, c in enumerate(cols)}

    # ------------------------------------------------------------------ E1 earnings surprise
    E = pd.read_parquet(SC / "earnings.parquet").dropna(subset=["epsActual", "epsEstimated"])
    E = E[E.symbol.isin(col) & (E.date >= "2015-06-01") & (E.date <= dates[-3])]
    i0 = dates.searchsorted(E.date.to_numpy()); E = E[(i0 >= 2) & (i0 < len(dates) - 2)]; i0 = dates.searchsorted(E.date.to_numpy())
    j = E.symbol.map(col).to_numpy()
    E = E.assign(react=A[i0 + 1, j] / A[i0 - 1, j] - 1 - (spy[i0 + 1] / spy[i0 - 1] - 1), in_small=small.to_numpy()[i0 - 1, j],
                 in_big=big.to_numpy()[i0 - 1, j], beat=E.epsActual > E.epsEstimated).dropna(subset=["react"])
    Es = E[E.in_small]
    print(f"\n=== E1 earnings surprises in small companies: {len(Es)} reports since mid-2015; {Es.beat.mean():.0%} beat; "
          f"good surprises {int((Es.beat & (Es.react >= 0.05)).sum())} ===")
    good = Es[Es.beat & (Es.react >= 0.05)].assign(score=lambda d: -d.react)
    f_e, s_e = flags_from(good[["symbol", "date", "score"]], dates, cols, offset=1)
    print(event_table(data, f_e, small, (5, 20, 40, 60, 120)).to_string())
    rows = {f"E1 good surprise, hold {h} days": traded(data, f_e, s_e, h, bench) for h in (20, 40, 60, 120)}
    rows["E1 hold 60, costs doubled"] = traded(data, f_e, s_e, 60, bench, cost=40)
    big_good = E[E.in_big & E.beat & (E.react >= 0.05)].assign(score=lambda d: -d.react)
    f_b, s_b = flags_from(big_good[["symbol", "date", "score"]], dates, cols, offset=1)
    rows["(same rule in the 500 biggest, hold 60)"] = traded(data, f_b, s_b, 60, bench, cost=5)
    print(pd.DataFrame(rows).T.round(3).to_string())

    # ------------------------------------------------------------------ I1 / I2 insider purchases
    ins = pd.read_parquet(SC / "insider.parquet")
    ins = ins[(ins.acquisitionOrDisposition == "A") & (ins.price > 0) & (ins.securitiesTransacted > 0) & ins.filingDate.notna() & ins.symbol.isin(col)]
    role = ins.typeOfOwner.fillna("").str.lower()
    ins = ins[role.str.contains("officer") | role.str.contains("director")]
    ins = ins.drop_duplicates(["symbol", "reportingCik", "transactionDate", "securitiesTransacted", "price"])
    ins = ins.assign(value=ins.securitiesTransacted * ins.price)
    ins = ins[(ins.value < 2e8) & (ins.filingDate >= "2015-06-01") & ((ins.filingDate - ins.transactionDate).dt.days.between(0, 30))]
    print(f"\n=== Insider purchases by officers and directors: {len(ins)} filings since mid-2015; by year {ins.groupby(ins.filingDate.dt.year).size().to_dict()} ===")
    per_person = ins.groupby(["symbol", "filingDate", "reportingCik"], as_index=False)["value"].sum()

    def run_insider(label, ev, hold_main=120):
        ev = ev[ev.date >= "2015-06-01"]
        f_all, s_all = flags_from(ev, dates, cols)
        f, s = f_all & small, s_all.where(small)
        n = int(f.loc[BEGIN:].sum().sum())
        print(f"\n-- {label}: {n} signals in small companies since {BEGIN.date()} (about {n / ((dates[-1] - BEGIN).days / 365.25):.0f} a year) --")
        print(event_table(data, f, small, (5, 20, 60, 120, 250)).to_string())
        rows = {f"{label}, hold {h} days": traded(data, f, s, h, bench) for h in (20, 60, 120, 250)}
        rows[f"{label}, hold {hold_main}, costs doubled"] = traded(data, f, s, hold_main, bench, cost=40)
        fb, sb = f_all & big, s_all.where(big)
        rows[f"(same rule in the 500 biggest, hold {hold_main})"] = traded(data, fb, sb, hold_main, bench, cost=5)
        print(pd.DataFrame(rows).T.round(3).to_string())
        return f, s

    f1, s1 = run_insider("I1 insider cluster", insider_events(per_person, 2, 100_000))
    f2, s2 = run_insider("I2 large single purchase", insider_events(per_person, 1, 250_000))

    # how sensitive is the cluster rule to its thresholds?
    print("\n=== I1 with other thresholds (hold 120 days) ===")
    rows = {}
    for people, value in ((2, 50_000), (2, 250_000), (3, 100_000), (3, 500_000), (1, 100_000), (1, 1_000_000)):
        f_all, s_all = flags_from(insider_events(per_person, people, value), dates, cols)
        rows[f"{people}+ insiders, ${value:,}+"] = traded(data, f_all & small, s_all.where(small), 120, bench)
    print(pd.DataFrame(rows).T.round(3).to_string())

    # year by year for the two headline rules
    yr = lambda r: ((1 + r.loc[BEGIN:]).groupby(r.loc[BEGIN:].index.year).prod() - 1)
    d_e, _ = simulate(data, f_e, s_e, None, slots=20, max_hold=60, cost_bps=COST, start=BEGIN)
    d_i, _ = simulate(data, f1, s1, None, slots=20, max_hold=120, cost_bps=COST, start=BEGIN)
    print("\n=== Year by year ===")
    print(pd.DataFrame({"E1 hold 60": yr(d_e["ret"]), "I1 hold 120": yr(d_i["ret"]), "small stocks": yr(bench["pool"]), "IWM": yr(bench["IWM"]), "SPY": yr(bench["SPY"])}).round(3).T.to_string())
    pd.DataFrame({"E1": d_e["ret"], "I1": d_i["ret"], "pool": bench["pool"], "SPY": bench["SPY"], "IWM": bench["IWM"]}).to_parquet(SC / "study_returns.parquet")


if __name__ == "__main__":
    main()
