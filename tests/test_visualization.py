"""Tests for the visualization module.

All fixtures are synthetic.
"""
import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from src.visualization import (
    feature_importance_chart,
    forecast_chart,
    location_map_html,
    metrics_bar_chart,
    seven_plus_three_chart,
    ssh_history_chart,
    validation_scatter,
)


def _synthetic_history(n: int = 30, start: str = "2024-01-01") -> pd.DataFrame:
    """SYNTHETIC: simple ascending SSH series."""
    return pd.DataFrame({
        "time": pd.date_range(start, periods=n, tz="UTC"),
        "ssh_m": [round(0.4 + i * 0.001, 4) for i in range(n)],
    })


def _synthetic_window(start: str = "2024-01-24") -> pd.DataFrame:
    """SYNTHETIC: 7-day window."""
    return pd.DataFrame({
        "time": pd.date_range(start, periods=7, tz="UTC"),
        "ssh_m": [0.42, 0.421, 0.422, 0.423, 0.424, 0.425, 0.426],
    })


class TestSSHHistoryChart:
    def test_returns_figure(self):
        fig = ssh_history_chart(_synthetic_history())
        assert isinstance(fig, go.Figure)

    def test_contains_history_trace(self):
        fig = ssh_history_chart(_synthetic_history())
        assert any("GLORYS" in t.name for t in fig.data)

    def test_window_trace_added_when_provided(self):
        fig = ssh_history_chart(_synthetic_history(), window=_synthetic_window())
        trace_names = [t.name for t in fig.data]
        assert any("window" in n.lower() for n in trace_names)


class TestForecastChart:
    def test_returns_figure(self):
        history = _synthetic_history()
        window  = _synthetic_window()
        dates   = pd.date_range("2024-01-31", periods=3, tz="UTC")
        preds   = np.array([0.427, 0.428, 0.429])
        fig = forecast_chart(history, window, dates, preds, preds)
        assert isinstance(fig, go.Figure)

    def test_persistence_trace_present(self):
        history = _synthetic_history()
        window  = _synthetic_window()
        dates   = pd.date_range("2024-01-31", periods=3, tz="UTC")
        preds   = np.array([0.427, 0.428, 0.429])
        fig = forecast_chart(history, window, dates, None, preds)
        trace_names = [t.name for t in fig.data]
        assert any("ersistence" in n for n in trace_names)

    def test_rf_trace_absent_when_none(self):
        history = _synthetic_history()
        window  = _synthetic_window()
        dates   = pd.date_range("2024-01-31", periods=3, tz="UTC")
        preds   = np.array([0.427, 0.428, 0.429])
        fig = forecast_chart(history, window, dates, None, preds)
        trace_names = [t.name for t in fig.data]
        assert not any("Random Forest" in n for n in trace_names)


class TestMetricsBarChart:
    def test_returns_figure_with_data(self):
        summary = pd.DataFrame({
            "lead": [0, 1, 2, 3],
            "lead_label": ["Overall", "Day 1", "Day 2", "Day 3"],
            "mae_m": [0.05, 0.04, 0.05, 0.06],
            "rmse_m": [0.07, 0.06, 0.07, 0.08],
        })
        fig = metrics_bar_chart(summary, "mae_m")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_empty_df_returns_figure(self):
        fig = metrics_bar_chart(pd.DataFrame(), "mae_m")
        assert isinstance(fig, go.Figure)


class TestValidationScatter:
    def test_empty_returns_figure(self):
        fig = validation_scatter(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_with_data(self):
        df = pd.DataFrame({
            "prediction_m": [0.4, 0.42, 0.41],
            "reference_m": [0.41, 0.43, 0.40],
            "lead": [1, 2, 3],
            "status": ["Fully validated"] * 3,
        })
        fig = validation_scatter(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestLocationMapHtml:
    def test_returns_html_string(self):
        html = location_map_html(12.0, 88.0, 12.0, 88.0)
        assert isinstance(html, str)
        assert len(html) > 10

    def test_contains_coordinates(self):
        html = location_map_html(12.083, 88.083, 12.0, 88.0)
        # Either folium HTML or fallback text should mention the grid coordinates
        assert "12" in html and "88" in html


class TestSevenPlusThreeChart:
    def test_returns_figure(self):
        window = _synthetic_window()
        dates = pd.date_range("2024-01-31", periods=3, tz="UTC")
        persist = np.array([0.43, 0.43, 0.43])
        rf = np.array([0.428, 0.427, 0.426])
        fig = seven_plus_three_chart(window, dates, persist, rf)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3

    def test_without_rf(self):
        window = _synthetic_window()
        dates = pd.date_range("2024-01-31", periods=3, tz="UTC")
        persist = np.array([0.43, 0.43, 0.43])
        fig = seven_plus_three_chart(window, dates, persist, None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2


class TestFeatureImportanceChart:
    def test_returns_figure_with_data(self):
        importances = [0.05, 0.05, 0.1, 0.1, 0.15, 0.2, 0.35]
        fig = feature_importance_chart(importances)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_empty_returns_figure(self):
        fig = feature_importance_chart([])
        assert isinstance(fig, go.Figure)
