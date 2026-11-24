"""SQLite forecast ledger.

Design principles
-----------------
* Original forecast rows are immutable: predictions are never overwritten.
* Validation results live in a separate ``validations`` table.
* Duplicate runs (same dataset, location, model, inputs, predictions) are
  detected by a SHA-256 ``run_key`` and silently de-duplicated.
* All timestamps are stored as ISO-8601 UTC strings.
"""
from __future__ import annotations
import hashlib
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id        TEXT PRIMARY KEY,
    generated_at_utc   TEXT NOT NULL,
    source             TEXT NOT NULL,
    dataset_id         TEXT NOT NULL,
    requested_lat      REAL NOT NULL,
    requested_lon      REAL NOT NULL,
    grid_lat           REAL NOT NULL,
    grid_lon           REAL NOT NULL,
    input_json         TEXT NOT NULL,
    predictions_json   TEXT NOT NULL,
    model_name         TEXT NOT NULL,
    model_version      TEXT NOT NULL,
    training_json      TEXT NOT NULL,
    retrieval_status   TEXT NOT NULL,
    validation_status  TEXT NOT NULL,
    run_key            TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS validations (
    validation_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_id         TEXT    NOT NULL REFERENCES forecasts(forecast_id),
    lead                INTEGER NOT NULL,
    target_time_utc     TEXT    NOT NULL,
    prediction_m        REAL    NOT NULL,
    reference_m         REAL,
    absolute_error_m    REAL,
    reference_dataset_id TEXT   NOT NULL,
    reference_type      TEXT    NOT NULL,
    status              TEXT    NOT NULL,
    checked_at_utc      TEXT    NOT NULL,
    UNIQUE(forecast_id, lead, reference_dataset_id)
);
"""

_VALID_STATUSES = frozenset({
    "Pending",
    "Partially validated",
    "Fully validated",
    "Reference unavailable",
    "Validation error",
})


def connect(path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    return con


def initialize(path) -> None:
    """Create tables if they do not already exist."""
    with connect(path) as con:
        con.executescript(_SCHEMA)
    log.debug("Database initialised at %s", path)


# ---------------------------------------------------------------------------
# Forecast persistence
# ---------------------------------------------------------------------------

def save_forecast(
    path,
    *,
    source: str,
    dataset_id: str,
    requested_lat: float,
    requested_lon: float,
    grid_lat: float,
    grid_lon: float,
    inputs: pd.DataFrame,
    predictions: pd.Series,
    model_name: str,
    model_version: str = "1.0",
    training: dict | None = None,
    retrieval_status: str = "success",
) -> str:
    """Persist a forecast record and its pending validation rows.

    Returns the forecast UUID (the existing one if this run was a duplicate).
    """
    generated = datetime.now(timezone.utc).isoformat()
    fid = str(uuid.uuid4())

    input_rows = [
        {"time": pd.Timestamp(t).isoformat(), "ssh_m": float(v)}
        for t, v in zip(inputs["time"], inputs["ssh_m"])
    ]
    pred_rows = [
        {"time": pd.Timestamp(t).isoformat(), "lead": i + 1, "ssh_m": float(v)}
        for i, (t, v) in enumerate(zip(predictions.index, predictions.values))
    ]

    canonical = json.dumps(
        [dataset_id, float(grid_lat), float(grid_lon), model_name, input_rows, pred_rows],
        sort_keys=True,
    )
    run_key = hashlib.sha256(canonical.encode()).hexdigest()

    with connect(path) as con:
        con.execute(
            """INSERT OR IGNORE INTO forecasts
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                fid, generated, source, dataset_id,
                float(requested_lat), float(requested_lon),
                float(grid_lat), float(grid_lon),
                json.dumps(input_rows), json.dumps(pred_rows),
                model_name, model_version,
                json.dumps(training or {}),
                retrieval_status, "Pending", run_key,
            ),
        )
        # Retrieve actual ID (may differ if duplicate was skipped)
        row = con.execute(
            "SELECT forecast_id FROM forecasts WHERE run_key=?", (run_key,)
        ).fetchone()
        actual_fid = row[0]

        # Create pending validation rows (once per lead)
        ref_dataset = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
        for p in pred_rows:
            con.execute(
                """INSERT OR IGNORE INTO validations
                   (forecast_id, lead, target_time_utc, prediction_m,
                    reference_dataset_id, reference_type, status, checked_at_utc)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    actual_fid, p["lead"], p["time"], p["ssh_m"],
                    ref_dataset, "operational analysis/forecast",
                    "Pending", generated,
                ),
            )
    if actual_fid != fid:
        log.info("Duplicate run detected; reusing existing forecast %s.", actual_fid)
    else:
        log.info("Forecast %s saved (%s, %d predictions).", fid, model_name, len(pred_rows))
    return actual_fid


# ---------------------------------------------------------------------------
# Validation updates
# ---------------------------------------------------------------------------

def _recompute_forecast_status(con: sqlite3.Connection, forecast_id: str) -> str:
    rows = con.execute(
        "SELECT status FROM validations WHERE forecast_id=?", (forecast_id,)
    ).fetchall()
    statuses = [r[0] for r in rows]
    validated = sum(s == "Fully validated" or s == "Validated" for s in statuses)
    # Normalise legacy "Validated" to "Fully validated" per-lead meaning
    if any(s == "Validation error" for s in statuses):
        return "Validation error"
    if validated == len(statuses) and len(statuses) > 0:
        return "Fully validated"
    if validated > 0:
        return "Partially validated"
    if all(s == "Reference unavailable" for s in statuses) and statuses:
        return "Reference unavailable"
    return "Pending"


def update_validation(
    path,
    forecast_id: str,
    lead: int,
    target_time,
    prediction: float,
    reference: float | None,
    ref_dataset: str,
    reference_type: str = "analysis",
) -> None:
    """Record the reference value and error for one forecast lead.

    ``reference=None`` means no data was found; it is recorded as
    'Reference unavailable', never as zero.
    """
    if reference is None:
        status = "Reference unavailable"
        error  = None
    else:
        status = "Fully validated"
        error  = abs(float(prediction) - float(reference))

    now = datetime.now(timezone.utc).isoformat()
    target_iso = pd.Timestamp(target_time).isoformat()

    with connect(path) as con:
        # Try update of existing pending row first
        con.execute(
            """UPDATE validations
               SET target_time_utc=?, prediction_m=?, reference_m=?,
                   absolute_error_m=?, reference_type=?, status=?, checked_at_utc=?
               WHERE forecast_id=? AND lead=? AND status='Pending'""",
            (target_iso, prediction, reference, error, reference_type, status, now,
             forecast_id, lead),
        )
        if con.total_changes == 0:
            # Upsert for already-validated rows
            con.execute(
                """INSERT INTO validations
                   (forecast_id, lead, target_time_utc, prediction_m, reference_m,
                    absolute_error_m, reference_dataset_id, reference_type, status, checked_at_utc)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(forecast_id, lead, reference_dataset_id)
                   DO UPDATE SET reference_m=excluded.reference_m,
                                 absolute_error_m=excluded.absolute_error_m,
                                 reference_type=excluded.reference_type,
                                 status=excluded.status,
                                 checked_at_utc=excluded.checked_at_utc""",
                (forecast_id, lead, target_iso, prediction, reference, error,
                 ref_dataset, reference_type, status, now),
            )
        overall = _recompute_forecast_status(con, forecast_id)
        con.execute(
            "UPDATE forecasts SET validation_status=? WHERE forecast_id=?",
            (overall, forecast_id),
        )


def mark_validation_error(
    path,
    forecast_id: str,
    lead: int,
    message: str = "Reference retrieval failed",
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connect(path) as con:
        con.execute(
            """UPDATE validations
               SET reference_type=?, status='Validation error', checked_at_utc=?
               WHERE forecast_id=? AND lead=? AND status='Pending'""",
            (str(message)[:500], now, forecast_id, lead),
        )
        con.execute(
            "UPDATE forecasts SET validation_status='Validation error' WHERE forecast_id=?",
            (forecast_id,),
        )


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def list_forecasts(path) -> pd.DataFrame:
    with connect(path) as con:
        return pd.read_sql_query(
            "SELECT * FROM forecasts ORDER BY generated_at_utc DESC", con
        )


def list_validations(path) -> pd.DataFrame:
    with connect(path) as con:
        return pd.read_sql_query(
            """SELECT v.*, f.generated_at_utc, f.grid_lat, f.grid_lon
               FROM validations v
               JOIN forecasts f USING(forecast_id)
               ORDER BY f.generated_at_utc DESC, v.lead""",
            con,
        )


def get_pending_validations(path) -> pd.DataFrame:
    with connect(path) as con:
        return pd.read_sql_query(
            """SELECT v.*, f.grid_lat, f.grid_lon, f.dataset_id
               FROM validations v
               JOIN forecasts f USING(forecast_id)
               WHERE v.status = 'Pending'
               ORDER BY f.generated_at_utc DESC, v.lead""",
            con,
        )
