"""Data loading, indicators and a day-by-day trade simulator for swing-trading research.

The simulator is deliberately literal: it walks forward one day at a time, only ever
looks at data up to that day's close, and fills orders at a price that came later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STOCK_DIR = ROOT / "data" / "stocks"
START = pd.Timestamp("2003-03-01")
SPLIT = pd.Timestamp("2015-01-01")
RECENT = pd.Timestamp("2021-01-01")
TRADING_DAYS = 252


# ---------------------------------------------------------------------------- data
@dataclass
class Data:
    open: pd.DataFrame      # split-adjusted
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    adj: pd.DataFrame       # close adjusted for splits and dividends
    volume: pd.DataFrame
    pool: pd.DataFrame      # ticker -> name, sector, source

    @property
    def adj_open(self) -> pd.DataFrame:
        return self.open * (self.adj / self.close)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.adj.index

    def stocks(self) -> list[str]:
        """Tickers that are companies (not funds or indexes)."""
        is_fund = self.pool["source"].eq("benchmark") | self.pool.index.isin(FUNDS)
        return [t for t in self.adj.columns if t in self.pool.index and not is_fund.get(t, False)]


FUNDS = set("""ARKK KWEB FXI EWZ GDX GDXJ SLV GLD USO UNG SMH SOXX XBI IBB XOP XME XHB XRT KRE KBE ITB IYR HYG
LQD EEM EFA DIA VTI VOO SPY QQQ IWM MDY SHY TLT XLK XLF XLE XLV XLY XLP XLI XLB XLU XLRE XLC""".split())


def load(directory: Path = STOCK_DIR, clean: bool = True, delisted: bool = True) -> Data:
    """Yahoo data for companies that still trade, plus (if present and wanted) the delisted
    and dropped-from-index companies recovered from FMP by fmp_delisted.py / fmp_merge.py."""
    fields = ("open", "high", "low", "close", "adj", "volume")
    frames = {f: pd.read_parquet(directory / f"{f}.parquet").astype("float64") for f in fields}
    pool = pd.read_csv(directory / "pool.csv", index_col=0)
    accepted = directory / "fmp_accepted.csv"
    if delisted and accepted.exists():
        meta = pd.read_csv(accepted, index_col=0)
        names = [c for c in meta.index if c not in frames["adj"].columns]
        for f in fields:
            extra = pd.read_parquet(directory / f"fmp_{f}.parquet")[names].astype("float64")
            frames[f] = frames[f].join(extra, how="left")
        add = pd.DataFrame({"name": meta.loc[names, "name"], "sector": "", "yahoo": "",
                            "source": np.where(meta.loc[names, "why"] == "delisted", "delisted", "left index")}, index=names)
        pool = pd.concat([pool, add])
    pool = pool[~pool.index.duplicated(keep="last")]
    for f in ("open", "high", "low", "close", "adj"):
        frames[f] = frames[f].where(frames[f] > 0)
    data = Data(pool=pool, **frames)
    return clean_data(data) if clean else data


def clean_data(data: Data, jump: float = 3.0) -> Data:
    """Remove the few data faults that would otherwise dominate a backtest.

    1. Days the market was closed (no S&P 500 bar) are dropped.
    2. An opening price outside that day's high-low range is treated as unknown.
    3. A one-day rise of more than 200% is treated as a data splice (for example a
       company's old, wiped-out shares glued onto its new ones): earlier history is
       rescaled so that day's return is zero. Real collapses are left untouched, and a
       genuine +200% day is lost, so this can only make buy-side results look worse.
    """
    open_day = data.adj["SPY"].notna()
    f = {k: getattr(data, k)[open_day] for k in ("open", "high", "low", "close", "adj", "volume")}
    bad_open = (f["open"] > f["high"] * 1.001) | (f["open"] < f["low"] * 0.999)
    f["open"] = f["open"].mask(bad_open)
    filled = f["adj"].ffill()
    ratio = (filled / filled.shift(1)).fillna(1.0)
    factor = ratio.where(ratio > jump, 1.0)
    scale = factor.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)   # product of later jumps
    for k in ("open", "high", "low", "close", "adj"):
        f[k] = f[k] * scale
    f["volume"] = f["volume"] / scale
    return Data(pool=data.pool, **f)


# ---------------------------------------------------------------------- indicators
def sma(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).mean()


def rsi(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    d = df.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def liquid_universe(data: Data, top: int, lookback: int = 60, min_history: int = 252,
                    skip_top: int = 0) -> pd.DataFrame:
    """True where a stock ranks in the `top` most traded (by dollars) over the past
    `lookback` days. Uses only past data, so a stock joins the list when it actually
    became heavily traded, not because we know it is big today."""
    cols = data.stocks()
    dollars = (data.close[cols] * data.volume[cols]).rolling(lookback, min_periods=lookback * 2 // 3).median()
    seasoned = data.adj[cols].notna().rolling(min_history, min_periods=1).sum() >= min_history
    rank = dollars.where(seasoned).rank(axis=1, ascending=False)
    return ((rank <= top) & (rank > skip_top)).reindex(columns=data.adj.columns, fill_value=False)


# ----------------------------------------------------------------------- simulator
def simulate(data: Data, entry: pd.DataFrame, rank: pd.DataFrame, exit_: pd.DataFrame | None = None, *,
             slots: int = 10, weight: float | None = None, max_hold: int | None = None,
             mode: str = "next_open", cost_bps: float = 5.0, start: pd.Timestamp = START,
             entry_gate: pd.Series | None = None, seed_choice: np.random.Generator | None = None,
             delist_haircut: float = 0.0):
    """Walk forward through history.

    entry / exit_ : True where a signal is seen at that day's CLOSE.
    rank          : lower is better; used to choose when signals outnumber free slots.
    mode          : "next_open"  -> orders fill at the next day's open (evening check-in)
                    "same_close" -> orders fill at the close the signal is seen on
    weight        : share of the account per position (default 1 / slots).
    max_hold      : sell after holding this many trading days regardless (bought at Monday's
                    open with max_hold=5 means sold at the following Monday's open).
    entry_gate    : optional per-day switch; no new positions on days it is False.
    delist_haircut: a held stock whose data ends is sold at its last price; if it had lost
                    more than half its value in its final six months (a failure rather than
                    a buyout), this further fraction is taken off, since the last quoted
                    price of a failing company usually flatters what a holder really got.

    Returns (daily DataFrame with ret / invested, trades DataFrame).
    Cash earns nothing here and positions never exceed the account (no borrowing).
    """
    weight = weight if weight is not None else 1.0 / slots
    cols = list(data.adj.columns)
    dates = data.dates
    O = data.adj_open.to_numpy()
    C = data.adj.to_numpy()
    C_ff = data.adj.ffill().to_numpy()
    E = entry.reindex(index=dates, columns=cols).fillna(False).to_numpy(dtype=bool)
    R = rank.reindex(index=dates, columns=cols).to_numpy(dtype=float)
    X = (exit_.reindex(index=dates, columns=cols).fillna(False).to_numpy(dtype=bool)
         if exit_ is not None else np.zeros_like(E))
    gate = (entry_gate.reindex(dates).fillna(False).to_numpy(dtype=bool)
            if entry_gate is not None else np.ones(len(dates), bool))
    cost = cost_bps / 1e4
    finite = np.isfinite(C)
    last_bar = np.where(finite.any(axis=0), len(dates) - 1 - np.argmax(finite[::-1], axis=0), -1)
    half_year_ago = data.adj.ffill().shift(126).to_numpy()
    lag = 1 if mode == "next_open" else 0     # a sale decided tonight happens one bar later
    i0 = int(dates.searchsorted(start))

    cash, held = 1.0, {}                      # held: column -> [shares, entry_i, entry_px]
    pend_in, pend_out = [], []
    equity_prev = 1.0
    daily, trades = [], []

    def close_position(j, px, i):
        nonlocal cash
        shares, entry_i, entry_px = held.pop(j)
        cash += shares * px * (1 - cost)
        trades.append((cols[j], dates[entry_i], dates[i], i - entry_i,
                       px * (1 - cost) / (entry_px * (1 + cost)) - 1))

    def open_position(j, px, i, equity_ref):
        nonlocal cash
        # never spend more cash than is on hand: winners that have grown past their
        # slot must not turn into hidden, interest-free borrowing
        value = min(equity_ref * weight, cash / (1 + cost))
        if value <= equity_ref * weight * 0.25 or not np.isfinite(px):
            return
        held[j] = [value / px, i, px]
        cash -= value * (1 + cost)

    for i in range(i0, len(dates)):
        for j in [j for j in held if i > last_bar[j]]:           # stopped trading: paid out
            k = last_bar[j]
            failed = np.isfinite(half_year_ago[k, j]) and C[k, j] < 0.5 * half_year_ago[k, j]
            close_position(j, C[k, j] * (1 - delist_haircut if failed else 1.0), i)
            pend_out = [x for x in pend_out if x != j]
        if mode == "next_open":
            still = []
            for j in pend_out:                 # sell first, at today's open
                if np.isfinite(O[i, j]):
                    close_position(j, O[i, j], i)
                else:
                    still.append(j)            # no price today: keep trying
            pend_out = still
            for j in pend_in:                  # then buy, sized off yesterday's account value
                if len(held) < slots and j not in held:
                    open_position(j, O[i, j], i, equity_prev)
            pend_in = []

        equity = cash + sum(s * C_ff[i, j] for j, (s, _, _) in held.items())

        # ---- decisions at today's close, using only today's and earlier data
        leaving = [j for j, (_, entry_i, _) in held.items()
                   if j not in pend_out and (X[i, j] or (max_hold is not None and i - entry_i + lag >= max_hold))]
        if mode == "same_close":
            for j in leaving:
                if np.isfinite(C[i, j]):
                    close_position(j, C[i, j], i)
            leaving = []
        free = slots - (len(held) - len(leaving) - len(pend_out))
        picks = []
        if free > 0 and gate[i]:
            cand = np.flatnonzero(E[i])
            cand = [j for j in cand if j not in held]
            if seed_choice is not None:
                seed_choice.shuffle(cand)
            else:
                cand.sort(key=lambda j: (R[i, j] if np.isfinite(R[i, j]) else np.inf))
            picks = cand[:free]
        if mode == "same_close":
            for j in picks:
                open_position(j, C[i, j], i, equity)
            equity = cash + sum(s * C_ff[i, j] for j, (s, _, _) in held.items())
        else:
            pend_out += leaving
            pend_in = picks

        invested = sum(s * C_ff[i, j] for j, (s, _, _) in held.items()) / equity if equity > 0 else 0.0
        daily.append((equity / equity_prev - 1, invested, len(held)))
        equity_prev = equity

    out = pd.DataFrame(daily, index=dates[i0:], columns=["ret", "invested", "positions"])
    tr = pd.DataFrame(trades, columns=["ticker", "entry", "exit", "days", "ret"])
    return out, tr


# --------------------------------------------------------------------------- stats
def nw_tstat(x: np.ndarray, lags: int = 5) -> float:
    """t-statistic of the mean, allowing for day-to-day correlation (Newey-West)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return np.nan
    d = x - x.mean()
    var = (d @ d) / n
    for lag in range(1, lags + 1):
        var += 2 * (1 - lag / (lags + 1)) * (d[lag:] @ d[:-lag]) / n
    return x.mean() / np.sqrt(var / n) if var > 0 else np.nan


def stats(daily: pd.DataFrame, bench: pd.Series | None = None) -> dict:
    r = daily["ret"]
    years = len(r) / TRADING_DAYS
    eq = (1 + r).cumprod()
    out = {
        "Return/yr": eq.iloc[-1] ** (1 / years) - 1,
        "Vol": r.std() * np.sqrt(TRADING_DAYS),
        "Sharpe": r.mean() * TRADING_DAYS / (r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan,
        "Worst drop": (eq / eq.cummax() - 1).min(),
        "Invested": daily["invested"].mean() if "invested" in daily else np.nan,
    }
    if bench is not None:
        b = bench.reindex(r.index).fillna(0.0)
        beta = np.cov(r, b)[0, 1] / b.var() if b.var() > 0 else np.nan
        resid = r - beta * b
        out["Beta"] = beta
        out["Alpha/yr"] = resid.mean() * TRADING_DAYS
        out["Alpha t"] = nw_tstat(resid.to_numpy())
    return out


def trade_stats(tr: pd.DataFrame) -> dict:
    if not len(tr):
        return {"Trades": 0}
    wins, losses = tr.ret[tr.ret > 0], tr.ret[tr.ret <= 0]
    years = max((tr["exit"].max() - tr["entry"].min()).days / 365.25, 1e-9)
    return {
        "Trades": len(tr), "Trades/yr": len(tr) / years, "Win rate": (tr.ret > 0).mean(),
        "Avg trade": tr.ret.mean(), "Median": tr.ret.median(),
        "Avg win": wins.mean() if len(wins) else np.nan, "Avg loss": losses.mean() if len(losses) else np.nan,
        "Worst": tr.ret.min(), "Avg days": tr.days.mean(),
        "Profit factor": wins.sum() / -losses.sum() if len(losses) and losses.sum() < 0 else np.nan,
    }


def by_period(daily: pd.DataFrame, bench: pd.Series | None = None) -> dict[str, dict]:
    return {"2003-2014": stats(daily[daily.index < SPLIT], bench),
            "2015-2026": stats(daily[daily.index >= SPLIT], bench),
            "2021-2026": stats(daily[daily.index >= RECENT], bench),
            "All": stats(daily, bench)}
