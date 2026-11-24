"""Validation and construction of chronological daily SSH windows.

NOTE: ``clean_daily_series`` validates individual column types and removes
NaN rows.  It does *not* enforce day-by-day consecutive spacing across the
full multi-month history frame — that check is applied only to the seven-day
input window extracted by ``latest_input_window``.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def clean_daily_series(
    frame: pd.DataFrame,
    time_col: str = "time",
    value_col: str = "ssh_m",
) -> pd.DataFrame:
    """Return a cleaned copy of the SSH frame.

    Validates column types, parses timestamps, removes NaN rows, and checks
    for duplicate timestamps.  Does *not* require the series to be strictly
    consecutive day-by-day across the entire frame (gaps are allowed in the
    full history; consecutive-day enforcement happens only in
    ``latest_input_window``).

    Raises
    ------
    ValueError
        On missing columns, un-parseable timestamps, duplicate timestamps, or
        non-finite SSH values.
    """
    if time_col not in frame or value_col not in frame:
        raise ValueError(f"Input frame must contain '{time_col}' and '{value_col}' columns.")

    out = frame[[time_col, value_col]].copy()
    out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")

    if out[time_col].isna().any():
        raise ValueError("One or more observation timestamps could not be parsed.")
    if out[time_col].duplicated().any():
        n = int(out[time_col].duplicated().sum())
        raise ValueError(f"Duplicate observation timestamps are not allowed ({n} duplicates found).")
    if not np.isfinite(out[value_col].dropna()).all():
        raise ValueError("SSH observations must be finite numeric values (no inf allowed).")

    # Drop NaN SSH rows after checking for non-finite; NaN means missing, not corrupt.
    out = out.dropna(subset=[value_col])
    out = out.sort_values(time_col).reset_index(drop=True)
    log.debug("clean_daily_series: %d valid observations retained.", len(out))
    return out


def _check_consecutive(frame: pd.DataFrame, time_col: str = "time") -> bool:
    """Return True if every adjacent pair of timestamps is exactly 1 day apart."""
    if len(frame) < 2:
        return True
    diffs = frame[time_col].diff().dropna()
    return bool((diffs == pd.Timedelta(days=1)).all())


def latest_input_window(frame: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Extract the most recent ``window`` rows and verify they are consecutive daily values.

    Raises
    ------
    ValueError
        If fewer than ``window`` observations exist or the tail is not
        consecutive day-by-day.
    """
    clean = clean_daily_series(frame)
    if len(clean) < window:
        raise ValueError(
            f"Need at least {window} consecutive valid daily SSH observations; "
            f"found {len(clean)}."
        )
    tail = clean.tail(window).copy().reset_index(drop=True)
    if not _check_consecutive(tail):
        raise ValueError(
            f"The {window} most recent observations are not consecutive daily values. "
            "Check for missing dates near the end of the series."
        )
    return tail


def supervised_windows(
    values,
    input_size: int = 7,
    horizon: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Create (X, y) pairs using a sliding window over a 1-D value array.

    Each input window of length ``input_size`` is paired with the immediately
    following ``horizon`` values.  Windows are non-overlapping in target space
    only in the sense that they are ordered chronologically; the sliding step
    is 1 day.

    Returns empty arrays if the series is too short.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Values must be a one-dimensional series.")
    if not np.isfinite(arr).all():
        raise ValueError("All values must be finite before constructing supervised windows.")
    n = len(arr)
    n_windows = n - input_size - horizon + 1
    if n_windows <= 0:
        return np.empty((0, input_size), dtype=float), np.empty((0, horizon), dtype=float)
    X = np.stack([arr[i : i + input_size] for i in range(n_windows)])
    y = np.stack([arr[i + input_size : i + input_size + horizon] for i in range(n_windows)])
    return X, y
