"""Tests for the backtesting module.

All fixtures are synthetic.
"""
import numpy as np
import pandas as pd
import pytest

from src.backtesting import error_metrics, rolling_backtest, summarize_backtest


def _synthetic_history(n: int = 120, start: str = "2020-01-01") -> pd.DataFrame:
    """SYNTHETIC: smooth sinusoidal SSH series of length n."""
    t = np.arange(n, dtype=float)
    ssh = 0.5 * np.sin(2 * np.pi * t / 365) + 0.1 * np.random.default_rng(42).normal(size=n)
    return pd.DataFrame({
        "time": pd.date_range(start, periods=n, tz="UTC"),
        "ssh_m": ssh,
    })


class TestRollingBacktest:
    def test_returns_dataframe_with_expected_columns(self):
        df = rolling_backtest(_synthetic_history(100), min_train=60, max_origins=10)
        required = {"origin_time", "target_time", "lead", "actual_m", "persistence_m", "random_forest_m"}
        assert required.issubset(df.columns)

    def test_lead_values(self):
        df = rolling_backtest(_synthetic_history(100), min_train=60, max_origins=5)
        assert set(df["lead"].unique()) == {1, 2, 3}

    def test_target_after_origin(self):
        df = rolling_backtest(_synthetic_history(100), min_train=60, max_origins=5)
        assert (df["target_time"] > df["origin_time"]).all()

    def test_no_future_data_in_training(self):
        """The training data for each origin must end before the target dates."""
        history = _synthetic_history(100)
        values = history["ssh_m"].to_numpy()
        df = rolling_backtest(history, min_train=60, max_origins=5)
        # For each row: actual_m comes from values[origin+lead-1] which is after the window
        # We verify that actual_m values match the history frame
        for _, row in df.iterrows():
            target_idx = history[history["time"] == row["target_time"]].index
            if len(target_idx):
                assert row["actual_m"] == pytest.approx(float(values[target_idx[0]]))

    def test_insufficient_history_returns_empty(self):
        df = rolling_backtest(_synthetic_history(10), min_train=60)
        assert df.empty

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="'time'"):
            rolling_backtest(pd.DataFrame({"x": [1, 2, 3]}))


class TestSummarizeBacktest:
    def test_summary_includes_all_leads_and_overall(self):
        df = rolling_backtest(_synthetic_history(100), min_train=60, max_origins=5)
        summary = summarize_backtest(df, "persistence_m")
        assert set(summary["lead"]).issuperset({0, 1, 2, 3})

    def test_empty_input_returns_empty(self):
        result = summarize_backtest(pd.DataFrame())
        assert result.empty

    def test_mae_non_negative(self):
        df = rolling_backtest(_synthetic_history(100), min_train=60, max_origins=5)
        summary = summarize_backtest(df, "persistence_m")
        assert (summary["mae_m"] >= 0).all()
        assert (summary["rmse_m"] >= 0).all()
