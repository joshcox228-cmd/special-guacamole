"""Walk-forward machine learning: at each rebalance, train only on the past."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .config import Config

REQUIRED = "ret_252"  # an asset is eligible once it has a full year of history


def make_model(seed: int) -> LGBMRegressor:
    # Deliberately small and heavily regularised. Market data is mostly noise;
    # a big flexible model would memorise the past instead of learning from it.
    return LGBMRegressor(
        n_estimators=200, learning_rate=0.03, num_leaves=8, max_depth=3,
        min_child_samples=100, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=5.0,
        random_state=seed, n_jobs=1, deterministic=True, force_row_wise=True,
        verbose=-1,
    )


def rebalance_positions(n_days: int, cfg: Config) -> np.ndarray:
    """Row numbers of the days on which a new signal is produced."""
    return np.arange(cfg.min_train_days + cfg.offset, n_days, cfg.rebalance_every)


def training_rows(dates: pd.DatetimeIndex, pos: int, cfg: Config) -> pd.DatetimeIndex:
    """Dates usable for training a model that predicts on dates[pos].

    A label for day d is only known `exec_lag + horizon` days later, so anything
    newer than that is excluded. Without this "purge" the model would be trained
    on outcomes that had not happened yet.
    """
    last_known = pos - (cfg.exec_lag + cfg.horizon)
    return dates[: max(last_known + 1, 0)][:: cfg.sample_every]


def fit(features: pd.DataFrame, labels: pd.Series, train_dates, cfg: Config) -> LGBMRegressor:
    x = features.loc[features.index.get_level_values("date").isin(train_dates)]
    y = labels.reindex(x.index)
    keep = y.notna() & x[REQUIRED].notna()
    return make_model(cfg.seed).fit(x[keep], y[keep])


def predict(model: LGBMRegressor, features: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    x = features.xs(date, level="date")
    x = x[x[REQUIRED].notna()]
    return pd.Series(model.predict(x), index=x.index, name=date)


def walk_forward(features: pd.DataFrame, labels: pd.DataFrame, cfg: Config,
                 progress: bool = False) -> pd.DataFrame:
    """Out-of-sample scores: rows are signal dates, columns are tickers."""
    dates = labels.index
    y = labels.stack()
    y.index.names = ["date", "ticker"]
    out = []
    positions = rebalance_positions(len(dates), cfg)
    for i, pos in enumerate(positions):
        model = fit(features, y, training_rows(dates, pos, cfg), cfg)
        out.append(predict(model, features, dates[pos]))
        if progress and (i % 24 == 0 or i == len(positions) - 1):
            print(f"  trained through {dates[pos].date()} ({i + 1}/{len(positions)})", flush=True)
    return pd.DataFrame(out)
