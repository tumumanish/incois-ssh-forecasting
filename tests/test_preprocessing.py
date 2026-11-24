"""Tests for data preprocessing.

All fixtures are synthetic and clearly labelled as such.
Synthetic values are used only to verify software behaviour, not as ocean results.
"""
import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    clean_daily_series,
    latest_input_window,
    supervised_windows,
)

# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

def _synthetic_frame(n: int = 10, start: str = "2020-01-01") -> pd.DataFrame:
    """SYNTHETIC: n consecutive daily rows with ssh_m = 0.0, 0.1, …"""
    return pd.DataFrame({
        "time": pd.date_range(start, periods=n, freq="D", tz="UTC"),
        "ssh_m": [round(i * 0.1, 2) for i in range(n)],
    })


# ---------------------------------------------------------------------------
# clean_daily_series
# ---------------------------------------------------------------------------

class TestCleanDailySeries:
    def test_returns_correct_columns(self):
        df = clean_daily_series(_synthetic_frame())
        assert "time" in df.columns and "ssh_m" in df.columns

    def test_sorts_chronologically(self):
        df = _synthetic_frame(5).iloc[::-1].reset_index(drop=True)
        result = clean_daily_series(df)
        assert result["time"].is_monotonic_increasing

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="'ssh_m'"):
            clean_daily_series(pd.DataFrame({"time": pd.date_range("2020", periods=3, tz="UTC")}))

    def test_duplicate_timestamps_rejected(self):
        df = _synthetic_frame(5)
        df.loc[2, "time"] = df.loc[1, "time"]
        with pytest.raises(ValueError, match="Duplicate"):
            clean_daily_series(df)

    def test_non_parseable_timestamp_rejected(self):
        df = _synthetic_frame(5)
        df["time"] = df["time"].astype(str)
        df.loc[2, "time"] = "not-a-date"
        with pytest.raises(ValueError, match="parsed"):
            clean_daily_series(df)

    def test_infinite_ssh_rejected(self):
        df = _synthetic_frame(5)
        df.loc[2, "ssh_m"] = float("inf")
        with pytest.raises(ValueError, match="finite"):
            clean_daily_series(df)

    def test_gaps_allowed_in_full_history(self):
        """clean_daily_series must NOT raise for a gapped history frame —
        gap enforcement is left to latest_input_window."""
        df = _synthetic_frame(10)
        # Introduce a gap by skipping day 5
        df = pd.concat([df.iloc[:4], df.iloc[5:]], ignore_index=True)
        result = clean_daily_series(df)
        assert len(result) == 9  # one row removed, no error raised


# ---------------------------------------------------------------------------
# latest_input_window
# ---------------------------------------------------------------------------

class TestLatestInputWindow:
    def test_seven_day_window_shape(self):
        result = latest_input_window(_synthetic_frame(10))
        assert len(result) == 7

    def test_window_contains_most_recent_rows(self):
        df = _synthetic_frame(10)
        result = latest_input_window(df)
        assert result["ssh_m"].tolist() == [round(i * 0.1, 2) for i in range(3, 10)]

    def test_fewer_than_seven_raises(self):
        with pytest.raises(ValueError, match="at least 7"):
            latest_input_window(_synthetic_frame(6))

    def test_exactly_seven_ok(self):
        result = latest_input_window(_synthetic_frame(7))
        assert len(result) == 7

    def test_non_consecutive_tail_raises(self):
        # Build a 10-row frame, then remove row index 7 to create a gap in the
        # last 7 rows (days 3-9 become 3,4,5,6,8,9 — missing day 7).
        df = _synthetic_frame(10)
        df = pd.concat([df.iloc[:7], df.iloc[8:]], ignore_index=True)
        # Now the last 7 rows span days 1–6 and 9, which is not consecutive.
        with pytest.raises(ValueError, match="consecutive"):
            latest_input_window(df)

    def test_missing_values_in_tail_excluded(self):
        """NaN ssh_m in the tail means fewer than 7 clean rows → raises."""
        df = _synthetic_frame(10)
        df.loc[7, "ssh_m"] = float("nan")
        # After cleaning (NaN dropped), only 9 rows; tail has a gap → raises
        with pytest.raises(ValueError):
            latest_input_window(df)


# ---------------------------------------------------------------------------
# supervised_windows
# ---------------------------------------------------------------------------

class TestSupervisedWindows:
    def test_shape_exact(self):
        X, y = supervised_windows(range(12))
        assert X.shape == (3, 7) and y.shape == (3, 3)

    def test_no_overlap_in_start_positions(self):
        X, _ = supervised_windows(range(20))
        # Each row starts one step later
        for i in range(len(X) - 1):
            assert X[i + 1, 0] == X[i, 1]

    def test_too_short_returns_empty(self):
        X, y = supervised_windows(range(5))
        assert X.shape == (0, 7) and y.shape == (0, 3)

    def test_non_finite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            supervised_windows([1, 2, float("nan"), 4, 5, 6, 7, 8, 9, 10])

    def test_no_future_data_in_inputs(self):
        """The input at position i must not contain values at index i+7 or later."""
        arr = np.arange(15, dtype=float)
        X, y = supervised_windows(arr, input_size=7, horizon=3)
        for i, (x_row, y_row) in enumerate(zip(X, y)):
            # All x_row indices are < i + 7; all y_row indices are >= i + 7
            assert x_row[-1] < y_row[0], (
                f"Window {i}: last input {x_row[-1]} >= first target {y_row[0]} "
                "— future data leaked into inputs."
            )
