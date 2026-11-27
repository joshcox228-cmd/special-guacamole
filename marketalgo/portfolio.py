"""Check real holdings against their written sell rules, and add a trend read.

Flag-only: this reports what the rules say. It never places or suggests overriding a trade.

Rules, taken from the owner's Sell Discipline Sheet:
  * Technical stop: a close below the stop price means exit fully.
  * Trim levels: a close at or above trim1 / trim2 means trim part of the position.
  * Stops are anchored to the 50/88/200-day EMAs: about 1.25% under the nearest EMA
    that sits below price. A holding below all three has no EMA support; its stop is
    left where it is and the nearest EMA above becomes the "reclaim" level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EMA_SPANS = (50, 88, 200)
STOP_CUSHION = 0.0125   # stop sits this far under the supporting EMA
WATCH_BAND = 0.03       # "close to the stop" = within 3%
COLUMNS = ["ticker", "book", "shares", "avg_cost", "stop", "trim1", "trim2"]


def load_portfolio(path: Path) -> pd.DataFrame:
    book = pd.read_csv(path, comment="#", skipinitialspace=True)
    missing = [c for c in COLUMNS if c not in book.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    book["ticker"] = book["ticker"].str.strip().str.upper()
    book["book"] = book["book"].str.strip().str.lower()
    if book["ticker"].duplicated().any():
        raise ValueError("Each ticker should appear once in the portfolio file.")
    return book.set_index("ticker")


def ema_levels(close: pd.Series) -> dict[int, float]:
    return {span: float(close.ewm(span=span, adjust=False).mean().iloc[-1])
            for span in EMA_SPANS if len(close) >= span}


def ema_stop(price: float, emas: dict[int, float]) -> tuple[float | None, str]:
    """(suggested stop, description). No stop is suggested below all EMAs:
    lowering a stop into a downtrend just defines the discipline away."""
    below = {span: level for span, level in emas.items() if level < price}
    if below:
        span = max(below, key=below.get)  # the nearest support under price
        return below[span] * (1 - STOP_CUSHION), f"under {span}-day EMA"
    if emas:
        span = min(emas, key=emas.get)    # the nearest EMA overhead
        return None, f"below all EMAs; reclaim {emas[span]:.2f} ({span}d)"
    return None, "not enough history"


def classify(price: float, stop: float, trim1: float, trim2: float) -> tuple[str, str]:
    """(status, plain-English reason) for one holding."""
    if pd.notna(stop) and price < stop:
        return "EXIT", f"closed {price / stop - 1:+.1%} vs stop {stop:g}"
    if pd.notna(trim2) and price >= trim2:
        return "TRIM", f"at or above trim 2 ({trim2:g})"
    if pd.notna(trim1) and price >= trim1:
        return "TRIM", f"at or above trim 1 ({trim1:g})"
    if pd.isna(stop):
        return "NO STOP", "no stop on file"
    if price < stop * (1 + WATCH_BAND):
        return "WATCH", f"{price / stop - 1:.1%} above stop {stop:g}"
    return "QUIET", ""


def momentum_12_1(adj: pd.Series) -> float:
    """Same measure the ETF strategy ranks on: return from 12 months ago to 1 month ago."""
    if len(adj) <= 252:
        return np.nan
    return float(adj.iloc[-22] / adj.iloc[-253] - 1)


def review(book: pd.DataFrame, history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per holding. `history[ticker]` has columns close and adj."""
    rows = []
    for ticker, pos in book.iterrows():
        hist = history[ticker]
        close, adj = hist["close"], hist["adj"]
        price = float(close.iloc[-1])
        status, reason = classify(price, pos["stop"], pos["trim1"], pos["trim2"])
        emas = ema_levels(close)
        suggested, basis = ema_stop(price, emas)
        sma200 = close.rolling(200).mean().iloc[-1]
        rows.append({
            "ticker": ticker,
            "book": pos["book"],
            "status": status,
            "reason": reason,
            "price": price,
            "value": price * pos["shares"],
            "gain": price / pos["avg_cost"] - 1,
            "stop": pos["stop"],
            "to_stop": price / pos["stop"] - 1 if pd.notna(pos["stop"]) else np.nan,
            "ema_stop": suggested,
            "ema_note": basis,
            "uptrend": bool(price > sma200) if pd.notna(sma200) else None,
            "momentum": momentum_12_1(adj),
            "as_of": close.index[-1],
        })
    out = pd.DataFrame(rows).set_index("ticker")
    out["weight"] = out["value"] / out["value"].sum()
    out["mom_rank"] = out["momentum"].rank(ascending=False)
    return out


def trend_conflicts(report: pd.DataFrame) -> pd.DataFrame:
    """Holdings the sell rules are silent on, but the trend filter would not own:
    below the 200-day average AND negative 12-1 momentum."""
    quiet = report["status"].isin(["QUIET", "WATCH", "NO STOP"])
    weak = (report["uptrend"] == False) & (report["momentum"] < 0)  # noqa: E712
    return report[quiet & weak]
