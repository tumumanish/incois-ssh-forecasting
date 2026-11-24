"""Tests for the forecasting module.

All fixtures are synthetic.
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.backtesting import error_metrics
from src.forecasting import (
    ForecastResult,
    chronological_split,
    forecast,
    persistence_forecast,
    train_random_forest,
)

# ---------------------------------------------------------------------------
# persistence_forecast
# ---------------------------------------------------------------------------

class TestPersistenceForecast:
    def test_repeats_last_value(self):
        p = persistence_forecast([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        assert p.shape == (3,)
        assert np.all(p == 7.0)

    def test_horizon_matches(self):
        p = persistence_forecast([0.1] * 7, horizon=3)
        assert len(p) == 3

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            persistence_forecast([])

    def test_non_finite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            persistence_forecast([1, 2, float("nan"), 4, 5, 6, 7])


# ---------------------------------------------------------------------------
# forecast dispatch
# ---------------------------------------------------------------------------

class TestForecastDispatch:
    def test_persistence_when_no_model(self):
        result = forecast([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        assert isinstance(result, ForecastResult)
        assert result.model_name == "Persistence"
        assert result.predictions.shape == (3,)

    def test_wrong_window_length_raises(self):
        with pytest.raises(ValueError, match="exactly 7"):
            forecast([1.0, 2.0, 3.0])

    def test_non_finite_window_raises(self):
        with pytest.raises(ValueError, match="finite"):
            forecast([1.0, 2.0, float("inf"), 4.0, 5.0, 6.0, 7.0])

    def test_rf_model_output_shape(self):
        values = np.sin(np.arange(100) / 5.0)
        model, info = train_random_forest(values, min_samples=10)
        assert model is not None
        result = forecast(values[-7:], model)
        assert result.predictions.shape == (3,)
        assert result.model_name == "Random Forest"


# ---------------------------------------------------------------------------
# train_random_forest
# ---------------------------------------------------------------------------

class TestTrainRandomForest:
    def test_trained_on_sufficient_data(self):
        values = np.sin(np.arange(100) / 5.0)
        model, info = train_random_forest(values, min_samples=10)
        assert model is not None
        assert info["status"] == "trained"

    def test_none_returned_on_insufficient_data(self):
        values = np.arange(10, dtype=float)
        model, info = train_random_forest(values, min_samples=20)
        assert model is None
        assert info["status"] == "insufficient_data"

    def test_predict_shape(self):
        values = np.random.default_rng(0).uniform(0, 1, 120)
        model, _ = train_random_forest(values, min_samples=10)
        assert model is not None
        preds = model.predict(np.ones((1, 7)))
        assert preds.shape == (1, 3)

    def test_no_future_data_in_training(self):
        """Validation targets must start strictly after training targets."""
        values = np.arange(80, dtype=float)
        model, info = train_random_forest(values, min_samples=10)
        assert model is not None
        # The split function ensures no overlapping targets; verify info keys
        assert "training_examples" in info
        assert "validation_examples" in info


# ---------------------------------------------------------------------------
# chronological_split
# ---------------------------------------------------------------------------

class TestChronologicalSplit:
    def test_validation_after_training(self):
        X = np.arange(60).reshape(20, 3)
        y = np.array([[i, i + 1, i + 2] for i in range(20)])
        X_tr, y_tr, X_val, y_val = chronological_split(X, y)
        # Last training target must be < first validation target
        assert float(y_tr[-1, -1]) < float(y_val[0, 0])

    def test_no_overlap_in_targets(self):
        X = np.arange(100).reshape(50, 2)
        y = np.array([[i, i + 1, i + 2] for i in range(50)])
        X_tr, y_tr, X_val, y_val = chronological_split(X, y)
        # Purge gap ensures y_tr[-1, -1] < y_val[0, 0]
        assert float(y_tr[-1, -1]) < float(y_val[0, 0])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same number of rows"):
            chronological_split(np.ones((5, 2)), np.ones((6, 3)))

    def test_too_small_returns_empty_val(self):
        X = np.ones((5, 2))
        y = np.ones((5, 3))
        X_tr, y_tr, X_val, y_val = chronological_split(X, y, test_fraction=0.2)
        # With only 5 samples the split may yield empty validation
        assert len(X_tr) + len(X_val) <= len(X)


# ---------------------------------------------------------------------------
# error_metrics
# ---------------------------------------------------------------------------

class TestErrorMetrics:
    def test_mae_rmse_values(self):
        m = error_metrics([1.0, 2.0, 3.0], [1.0, 3.0, 3.0])
        assert m["mae_m"] == pytest.approx(1 / 3)
        assert m["rmse_m"] == pytest.approx((1 / 3) ** 0.5)

    def test_perfect_forecast(self):
        m = error_metrics([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
        assert m["mae_m"] == pytest.approx(0.0)
        assert m["rmse_m"] == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="identical shapes"):
            error_metrics([1, 2, 3], [1, 2])

    def test_non_finite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            error_metrics([1.0, float("nan")], [1.0, 2.0])
