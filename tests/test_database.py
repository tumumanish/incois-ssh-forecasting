"""Tests for database persistence.

All fixtures are synthetic and clearly labelled as such.
"""
import json
import pandas as pd
import pytest

from src import database


def _make_inputs(start: str = "2024-01-01", n: int = 7) -> pd.DataFrame:
    """SYNTHETIC: n consecutive daily rows."""
    return pd.DataFrame({
        "time":  pd.date_range(start, periods=n, tz="UTC"),
        "ssh_m": [float(i) * 0.05 for i in range(n)],
    })


def _make_predictions(start: str = "2024-01-08") -> pd.Series:
    """SYNTHETIC: 3 forecast values."""
    return pd.Series(
        [0.35, 0.36, 0.37],
        index=pd.date_range(start, periods=3, tz="UTC"),
    )


# ---------------------------------------------------------------------------
# Initialisation and round-trip
# ---------------------------------------------------------------------------

class TestForecastLedger:
    def test_initialize_creates_tables(self, tmp_path):
        db = tmp_path / "test.sqlite3"
        database.initialize(db)
        import sqlite3
        with sqlite3.connect(db) as con:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "forecasts" in tables and "validations" in tables

    def test_save_and_retrieve_forecast(self, tmp_path):
        db = tmp_path / "test.sqlite3"
        database.initialize(db)
        fid = database.save_forecast(
            db,
            source="synthetic test fixture",
            dataset_id="test-dataset",
            requested_lat=12.0, requested_lon=88.0,
            grid_lat=12.0, grid_lon=88.0,
            inputs=_make_inputs(), predictions=_make_predictions(),
            model_name="Persistence",
        )
        rows = database.list_forecasts(db)
        assert len(rows) == 1
        assert rows.iloc[0]["forecast_id"] == fid

    def test_pending_validations_created(self, tmp_path):
        db = tmp_path / "test.sqlite3"
        database.initialize(db)
        database.save_forecast(
            db,
            source="synthetic test fixture",
            dataset_id="test-dataset",
            requested_lat=12.0, requested_lon=88.0,
            grid_lat=12.0, grid_lon=88.0,
            inputs=_make_inputs(), predictions=_make_predictions(),
            model_name="Persistence",
        )
        val = database.list_validations(db)
        assert len(val) == 3
        assert set(val["status"]) == {"Pending"}
        assert set(val["lead"]) == {1, 2, 3}

    def test_duplicate_run_not_duplicated(self, tmp_path):
        db = tmp_path / "test.sqlite3"
        database.initialize(db)
        inputs = _make_inputs()
        preds  = _make_predictions()
        fid1 = database.save_forecast(
            db, source="synthetic test fixture", dataset_id="test",
            requested_lat=12.0, requested_lon=88.0,
            grid_lat=12.0, grid_lon=88.0,
            inputs=inputs, predictions=preds, model_name="Persistence",
        )
        fid2 = database.save_forecast(
            db, source="synthetic test fixture", dataset_id="test",
            requested_lat=12.0, requested_lon=88.0,
            grid_lat=12.0, grid_lon=88.0,
            inputs=inputs, predictions=preds, model_name="Persistence",
        )
        assert fid1 == fid2
        assert len(database.list_forecasts(db)) == 1


# ---------------------------------------------------------------------------
# Validation lifecycle
# ---------------------------------------------------------------------------

class TestValidationLifecycle:
    def _setup_pending(self, tmp_path):
        db = tmp_path / "test.sqlite3"
        database.initialize(db)
        fid = database.save_forecast(
            db,
            source="synthetic test fixture", dataset_id="test",
            requested_lat=12.0, requested_lon=88.0,
            grid_lat=12.0, grid_lon=88.0,
            inputs=_make_inputs(), predictions=_make_predictions(),
            model_name="Persistence",
        )
        return db, fid

    def test_update_validation_with_reference(self, tmp_path):
        db, fid = self._setup_pending(tmp_path)
        database.update_validation(
            db, fid, lead=1,
            target_time="2024-01-08T00:00:00+00:00",
            prediction=0.35, reference=0.32,
            ref_dataset="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            reference_type="operational analysis",
        )
        val = database.list_validations(db)
        lead1 = val[val["lead"] == 1].iloc[0]
        assert lead1["status"] == "Fully validated"
        assert pytest.approx(float(lead1["absolute_error_m"]), abs=1e-6) == abs(0.35 - 0.32)

    def test_reference_unavailable_not_zero(self, tmp_path):
        db, fid = self._setup_pending(tmp_path)
        database.update_validation(
            db, fid, lead=1,
            target_time="2024-01-08T00:00:00+00:00",
            prediction=0.35, reference=None,
            ref_dataset="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        )
        val = database.list_validations(db)
        lead1 = val[val["lead"] == 1].iloc[0]
        assert lead1["status"] == "Reference unavailable"
        assert lead1["reference_m"] is None or pd.isna(lead1["reference_m"])
        assert lead1["absolute_error_m"] is None or pd.isna(lead1["absolute_error_m"])

    def test_partial_validation_status(self, tmp_path):
        db, fid = self._setup_pending(tmp_path)
        database.update_validation(
            db, fid, lead=1,
            target_time="2024-01-08T00:00:00+00:00",
            prediction=0.35, reference=0.32,
            ref_dataset="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        )
        fc = database.list_forecasts(db)
        assert fc.iloc[0]["validation_status"] == "Partially validated"

    def test_fully_validated_status(self, tmp_path):
        db, fid = self._setup_pending(tmp_path)
        for lead, t in enumerate(["2024-01-08", "2024-01-09", "2024-01-10"], start=1):
            database.update_validation(
                db, fid, lead=lead,
                target_time=f"{t}T00:00:00+00:00",
                prediction=0.35 + lead * 0.01, reference=0.33 + lead * 0.01,
                ref_dataset="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            )
        fc = database.list_forecasts(db)
        assert fc.iloc[0]["validation_status"] == "Fully validated"

    def test_mark_validation_error(self, tmp_path):
        db, fid = self._setup_pending(tmp_path)
        database.mark_validation_error(db, fid, lead=2, message="Connection timeout")
        fc = database.list_forecasts(db)
        assert fc.iloc[0]["validation_status"] == "Validation error"

    def test_get_pending_validations(self, tmp_path):
        db, fid = self._setup_pending(tmp_path)
        pending = database.get_pending_validations(db)
        assert len(pending) == 3
        assert all(pending["status"] == "Pending")

    def test_original_prediction_preserved_after_validation(self, tmp_path):
        db, fid = self._setup_pending(tmp_path)
        original_preds = json.loads(database.list_forecasts(db).iloc[0]["predictions_json"])
        # Update validation — must NOT change the forecast row's predictions_json
        database.update_validation(
            db, fid, lead=1,
            target_time="2024-01-08T00:00:00+00:00",
            prediction=0.35, reference=0.32,
            ref_dataset="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        )
        updated_preds = json.loads(database.list_forecasts(db).iloc[0]["predictions_json"])
        assert original_preds == updated_preds
