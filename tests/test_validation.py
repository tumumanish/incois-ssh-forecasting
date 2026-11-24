"""Tests for reference matching and validation.

All fixtures are synthetic.
"""
import pandas as pd
import pytest

from src.validation import match_reference, validate_one


def _ref_frame(*timestamps_and_values) -> pd.DataFrame:
    """SYNTHETIC: build a small reference frame from (timestamp, value) pairs."""
    rows = [(pd.Timestamp(t, tz="UTC"), float(v)) for t, v in timestamps_and_values]
    return pd.DataFrame(rows, columns=["time", "ssh_m"])


# ---------------------------------------------------------------------------
# match_reference
# ---------------------------------------------------------------------------

class TestMatchReference:
    def test_exact_match_utc(self):
        ref = _ref_frame(("2024-01-01", 1.25))
        assert match_reference(ref, "2024-01-01T00:00:00Z") == pytest.approx(1.25)

    def test_no_match_returns_none(self):
        ref = _ref_frame(("2024-01-01", 1.25))
        assert match_reference(ref, "2024-01-02T00:00:00Z") is None

    def test_empty_frame_returns_none(self):
        assert match_reference(pd.DataFrame(columns=["time", "ssh_m"]), "2024-01-01") is None

    def test_none_frame_returns_none(self):
        assert match_reference(None, "2024-01-01") is None

    def test_nan_reference_returns_none(self):
        """A NaN reference value must not be returned — it is not a valid measurement."""
        import numpy as np
        ref = pd.DataFrame({
            "time": [pd.Timestamp("2024-01-01", tz="UTC")],
            "ssh_m": [float("nan")],
        })
        assert match_reference(ref, "2024-01-01T00:00:00Z") is None

    def test_missing_reference_not_treated_as_zero(self):
        """Absence of a reference value must not be interpreted as zero."""
        ref = _ref_frame(("2024-01-01", 0.0))
        result = match_reference(ref, "2024-01-02T00:00:00Z")
        assert result is None, "Missing reference must return None, not 0.0"

    def test_multiple_rows_returns_one(self):
        ref = _ref_frame(("2024-01-01", 1.0), ("2024-01-02", 2.0), ("2024-01-03", 3.0))
        assert match_reference(ref, "2024-01-02") == pytest.approx(2.0)

    def test_timezone_naive_target_treated_as_utc(self):
        ref = _ref_frame(("2024-06-15", 0.5))
        result = match_reference(ref, "2024-06-15T00:00:00")  # naive
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# validate_one
# ---------------------------------------------------------------------------

class TestValidateOne:
    def test_validate_one_matched(self, tmp_path):
        from src import database
        db = tmp_path / "test.sqlite3"
        database.initialize(db)
        inputs = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=7, tz="UTC"),
            "ssh_m": [float(i) * 0.1 for i in range(7)],
        })
        preds = pd.Series([0.6, 0.7, 0.8], index=pd.date_range("2024-01-08", periods=3, tz="UTC"))
        fid = database.save_forecast(
            db, source="synthetic test fixture", dataset_id="test",
            requested_lat=12.0, requested_lon=88.0,
            grid_lat=12.0, grid_lon=88.0,
            inputs=inputs, predictions=preds, model_name="Persistence",
        )
        ref = _ref_frame(("2024-01-08", 0.62))
        val = validate_one(db, fid, 1, "2024-01-08T00:00:00Z", 0.6, ref, "test-ref", "analysis")
        assert val == pytest.approx(0.62)
        validations = database.list_validations(db)
        lead1 = validations[validations["lead"] == 1].iloc[0]
        assert lead1["status"] == "Fully validated"
        assert float(lead1["absolute_error_m"]) == pytest.approx(abs(0.6 - 0.62))

    def test_validate_one_unmatched(self, tmp_path):
        from src import database
        db = tmp_path / "test.sqlite3"
        database.initialize(db)
        inputs = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=7, tz="UTC"),
            "ssh_m": [float(i) * 0.1 for i in range(7)],
        })
        preds = pd.Series([0.6, 0.7, 0.8], index=pd.date_range("2024-01-08", periods=3, tz="UTC"))
        fid = database.save_forecast(
            db, source="synthetic test fixture", dataset_id="test",
            requested_lat=12.0, requested_lon=88.0,
            grid_lat=12.0, grid_lon=88.0,
            inputs=inputs, predictions=preds, model_name="Persistence",
        )
        ref = _ref_frame(("2024-02-01", 0.99))  # no match for 2024-01-08
        val = validate_one(db, fid, 1, "2024-01-08T00:00:00Z", 0.6, ref, "test-ref", "analysis")
        assert val is None
        validations = database.list_validations(db)
        lead1 = validations[validations["lead"] == 1].iloc[0]
        assert lead1["status"] == "Reference unavailable"
