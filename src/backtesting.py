"""Rolling-origin direct forecasting backtesting on historical GLORYS data.

Each evaluation origin corresponds to a point in time where seven prior daily
SSH values are available and three future values can be retrieved from the
same historical dataset.  Metrics measure agreement with GLORYS reference
output, not accuracy against independent ground truth.
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.forecasting import persistence_forecast, train_random_forest

log = logging.getLogger(__name__)


def error_metrics(actual, predicted) -> dict:
    """Calculate MAE and RMSE in metres.

    Parameters
    ----------
    actual, predicted : array-like of float
        Matching arrays of reference and predicted values.

    Returns
    -------
    dict with keys ``mae_m`` and ``rmse_m``.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError("actual and predicted arrays must have identical shapes.")
    if not (np.isfinite(a).all() and np.isfinite(p).all()):
        raise ValueError("Metric arrays must contain only finite values.")
    return {
        "mae_m":  float(mean_absolute_error(a.ravel(), p.ravel())),
        "rmse_m": float(np.sqrt(mean_squared_error(a.ravel(), p.ravel()))),
    }


def rolling_backtest(
    frame: pd.DataFrame,
    min_train: int = 60,
    max_origins: int = 90,
    input_size: int = 7,
    horizon: int = 3,
) -> pd.DataFrame:
    """Evaluate persistence and Random Forest over expanding rolling origins.

    For each origin index ``t``:
    * Training data:  frame[0 : t]  (all observations strictly before the origin)
    * Input window:   frame[t-7 : t]
    * Targets:        frame[t : t+3]

    Parameters
    ----------
    frame : pd.DataFrame
        Columns ``time`` (tz-aware) and ``ssh_m`` (float).
    min_train : int
        Minimum number of prior observations before the first origin.
    max_origins : int
        Maximum number of distinct origins to evaluate (most recent).

    Returns
    -------
    pd.DataFrame with columns:
        origin_time, target_time, lead, actual_m,
        persistence_m, random_forest_m
    """
    if "time" not in frame or "ssh_m" not in frame:
        raise ValueError("frame must contain 'time' and 'ssh_m' columns.")

    values = np.asarray(frame["ssh_m"], dtype=float)
    times  = pd.to_datetime(frame["time"], utc=True)

    if not np.isfinite(values).all():
        raise ValueError("History frame contains non-finite SSH values; clean before backtesting.")

    rows: list[dict] = []
    # Earliest origin where we have min_train past obs AND input_size obs AND horizon future obs
    earliest = max(input_size, min_train)
    latest   = len(values) - horizon
    candidate_origins = list(range(earliest, latest + 1))[-max_origins:]

    for origin in candidate_origins:
        train  = values[:origin]
        window = train[-input_size:]

        # Persistence baseline — always defined
        persist = persistence_forecast(window, horizon)

        # Random Forest — retrain on all prior data at this origin
        model, _ = train_random_forest(train)
        if model is not None:
            rf = np.asarray(model.predict(window.reshape(1, -1))).reshape(-1)
        else:
            rf = persist.copy()   # fall back to persistence when RF unavailable

        for lead in range(horizon):
            rows.append({
                "origin_time":     times.iloc[origin - 1],
                "target_time":     times.iloc[origin + lead],
                "lead":            lead + 1,
                "actual_m":        float(values[origin + lead]),
                "persistence_m":   float(persist[lead]),
                "random_forest_m": float(rf[lead]),
            })

    if not rows:
        log.warning("No valid backtest origins found (min_train=%d, series length=%d).", min_train, len(values))
    else:
        log.info("Backtest: %d origins × %d leads = %d evaluation rows.", len(candidate_origins), horizon, len(rows))

    return pd.DataFrame(rows)


def summarize_backtest(
    results: pd.DataFrame,
    prediction_col: str = "random_forest_m",
) -> pd.DataFrame:
    """Aggregate per-lead MAE and RMSE across all backtest origins.

    Parameters
    ----------
    results : pd.DataFrame
        Output of ``rolling_backtest``.
    prediction_col : str
        Column to compare against ``actual_m``.

    Returns
    -------
    pd.DataFrame with columns: lead, mae_m, rmse_m
    """
    if results.empty:
        return pd.DataFrame(columns=["lead", "mae_m", "rmse_m"])
    out = []
    for lead, group in results.groupby("lead"):
        out.append({"lead": int(lead), **error_metrics(group["actual_m"], group[prediction_col])})
    summary = pd.DataFrame(out)
    # Overall row
    overall = error_metrics(results["actual_m"], results[prediction_col])
    overall["lead"] = 0   # 0 = overall
    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)
    summary["lead_label"] = summary["lead"].map({0: "Overall", 1: "Day 1", 2: "Day 2", 3: "Day 3"})
    return summary
