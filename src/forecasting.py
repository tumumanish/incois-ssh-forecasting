"""Direct three-step baseline and Random Forest forecasting.

Design
------
* ``persistence_forecast`` — repeats the last observed SSH value for every
  horizon step (simplest possible baseline).
* ``train_random_forest`` — trains a multi-output ``RandomForestRegressor``
  using a chronological (purged) split so test targets always follow training
  targets in time, preventing future-data leakage.
* ``forecast`` — dispatches to either model and returns a ``ForecastResult``.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from config.settings import (
    INPUT_WINDOW, FORECAST_HORIZON, MIN_RF_SAMPLES, RF_RANDOM_STATE,
    MODEL_VERSION,
)
from src.data_preprocessing import supervised_windows

log = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Immutable result container for one forecast run."""
    model_name:  str
    predictions: np.ndarray          # shape (horizon,)
    model:       Any | None  = None  # fitted estimator, if any
    training:    dict         = field(default_factory=dict)
    version:     str          = MODEL_VERSION


def persistence_forecast(window, horizon: int = FORECAST_HORIZON) -> np.ndarray:
    """Repeat the last observed value for each forecast step.

    Parameters
    ----------
    window : array-like, length >= 1
        Observed SSH values (metres), most recent last.
    horizon : int
        Number of forecast steps.

    Returns
    -------
    np.ndarray shape (horizon,)
    """
    values = np.asarray(window, dtype=float)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("Input window must be a non-empty 1-D array.")
    if not np.isfinite(values).all():
        raise ValueError("Input window contains non-finite values.")
    return np.repeat(values[-1], horizon)


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split (X, y) chronologically with a purge gap.

    The purge gap prevents the training targets from overlapping with the
    validation targets.  For a horizon-3 direct model, adjacent windows share
    target days, so ``horizon - 1`` windows are dropped between the two sets.

    Returns (X_train, y_train, X_val, y_val).  Returns empty validation arrays
    when the dataset is too small to split.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")
    horizon = y.shape[1] if y.ndim > 1 else 1
    nominal = max(1, int(len(X) * (1.0 - test_fraction)))
    # Purge: drop (horizon-1) windows immediately after the nominal cut.
    cut = max(1, nominal - (horizon - 1))
    if nominal >= len(X):
        log.warning("Dataset too small to form a validation split; returning empty validation.")
        return X, y, np.empty((0, *X.shape[1:]), dtype=X.dtype), np.empty((0, *y.shape[1:]), dtype=y.dtype)
    return X[:cut], y[:cut], X[nominal:], y[nominal:]


def train_random_forest(
    values,
    input_size: int = INPUT_WINDOW,
    horizon: int    = FORECAST_HORIZON,
    min_samples: int = MIN_RF_SAMPLES,
    random_state: int = RF_RANDOM_STATE,
) -> tuple[RandomForestRegressor | None, dict]:
    """Train a multi-output Random Forest on a chronological time series.

    Parameters
    ----------
    values : array-like
        Full historical SSH series (all available daily observations).
    input_size, horizon : int
        Sliding-window and forecast lengths.
    min_samples : int
        Minimum number of training examples required; returns ``None`` if
        training data is insufficient.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    (model, info) : (RandomForestRegressor | None, dict)
        ``model`` is ``None`` when there is insufficient data.
        ``info`` documents the training outcome.
    """
    arr = np.asarray(values, dtype=float)
    X, y = supervised_windows(arr, input_size, horizon)

    if len(X) < min_samples:
        log.warning("Insufficient supervised examples (%d) for Random Forest (need %d).", len(X), min_samples)
        return None, {"status": "insufficient_data", "examples": int(len(X)), "required": min_samples}

    X_tr, y_tr, X_val, y_val = chronological_split(X, y)

    if len(X_tr) < min_samples:
        log.warning(
            "Insufficient training examples after split (%d); need %d.", len(X_tr), min_samples
        )
        return None, {
            "status": "insufficient_data",
            "examples": int(len(X)),
            "training_examples": int(len(X_tr)),
        }

    model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    info: dict = {
        "status": "trained",
        "examples": int(len(X)),
        "training_examples": int(len(X_tr)),
        "validation_examples": int(len(X_val)),
        "input_size": input_size,
        "horizon": horizon,
        "n_estimators": 100,
        "random_state": random_state,
    }
    if len(X_val) > 0:
        from sklearn.metrics import mean_absolute_error
        preds = model.predict(X_val)
        info["val_mae_m"] = float(mean_absolute_error(y_val.ravel(), preds.ravel()))
        log.info("Random Forest trained. val_mae=%.4f m (%d validation examples).", info["val_mae_m"], len(X_val))
    else:
        log.info("Random Forest trained on all %d examples (no validation split).", len(X_tr))

    return model, info


def forecast(
    window,
    model: RandomForestRegressor | None = None,
    horizon: int = FORECAST_HORIZON,
) -> ForecastResult:
    """Generate a direct multi-step forecast from a seven-day input window.

    Parameters
    ----------
    window : array-like, length == INPUT_WINDOW
        Seven consecutive daily SSH values (metres), most recent last.
    model : RandomForestRegressor | None
        Fitted RF model; uses persistence when ``None``.
    horizon : int
        Number of forecast steps (must match model training).

    Returns
    -------
    ForecastResult
    """
    values = np.asarray(window, dtype=float)
    if len(values) != INPUT_WINDOW:
        raise ValueError(
            f"Forecast input must contain exactly {INPUT_WINDOW} daily SSH values; "
            f"received {len(values)}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Forecast input contains non-finite values.")

    if model is None:
        preds = persistence_forecast(values, horizon)
        return ForecastResult("Persistence", preds, None, {}, MODEL_VERSION)

    preds = np.asarray(model.predict(values.reshape(1, -1))).reshape(-1)
    if len(preds) != horizon:
        raise ValueError(
            f"Model returned {len(preds)} predictions but horizon is {horizon}."
        )
    return ForecastResult("Random Forest", preds, model, {}, MODEL_VERSION)
