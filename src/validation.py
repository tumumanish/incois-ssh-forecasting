"""Reference matching and non-destructive forecast validation.

Original forecast predictions are never overwritten.  When a matching
reference value from the operational product becomes available, it is stored
in a separate ``validations`` table with its own status lifecycle.

If the reference product contains only a forecast (not a finalized analysis)
for the target date, the comparison is labelled ``forecast-to-forecast`` and
is not an independent observation.
"""
from __future__ import annotations
import logging

import pandas as pd

from src.database import update_validation, mark_validation_error

log = logging.getLogger(__name__)


def match_reference(
    reference_frame: pd.DataFrame,
    target_time,
    tolerance: str = "0s",
) -> float | None:
    """Find the SSH value in ``reference_frame`` that matches ``target_time``.

    Parameters
    ----------
    reference_frame : pd.DataFrame
        Columns ``time`` (tz-aware) and ``ssh_m`` (float).
    target_time : str or datetime-like
        The forecast target timestamp.
    tolerance : str
        Pandas timedelta string; exact match by default.  Increase only when
        sub-daily rounding differences are expected.

    Returns
    -------
    float or None
        Matched SSH in metres, or ``None`` if not found.  Missing reference
        is never interpreted as zero.
    """
    if reference_frame is None or reference_frame.empty:
        return None

    frame = reference_frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)

    target = pd.Timestamp(target_time)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    else:
        target = target.tz_convert("UTC")

    tol = pd.Timedelta(tolerance)
    if tol == pd.Timedelta(0):
        matched = frame.loc[frame["time"] == target, "ssh_m"]
    else:
        delta = (frame["time"] - target).abs()
        matched = frame.loc[delta <= tol, "ssh_m"]
        if not matched.empty:
            matched = matched.iloc[:1]  # closest only

    if matched.empty:
        return None
    value = float(matched.iloc[0])
    return value if pd.notna(value) else None


def validate_one(
    db_path,
    forecast_id: str,
    lead: int,
    target_time,
    prediction: float,
    reference_frame: pd.DataFrame,
    reference_dataset: str,
    reference_type: str = "operational analysis/forecast",
) -> float | None:
    """Match a single forecast lead against the reference frame and persist the result.

    Parameters
    ----------
    db_path : Path
        SQLite database file path.
    forecast_id : str
        UUID of the original forecast record.
    lead : int
        Forecast lead (1, 2, or 3).
    target_time : str or datetime-like
        Target date for which the forecast was made.
    prediction : float
        Originally stored predicted SSH (metres).
    reference_frame : pd.DataFrame
        Available reference observations from the reference dataset.
    reference_dataset : str
        Copernicus dataset ID of the reference source.
    reference_type : str
        Human-readable description of what the reference represents.

    Returns
    -------
    float or None
        The matched reference value, or ``None`` if unavailable.
    """
    value = match_reference(reference_frame, target_time)
    if value is None:
        log.warning("No reference match for lead %d, target %s.", lead, target_time)
    else:
        log.info(
            "Reference matched for lead %d: pred=%.4f m, ref=%.4f m, error=%.4f m.",
            lead, prediction, value, abs(prediction - value),
        )
    update_validation(db_path, forecast_id, lead, target_time, prediction, value, reference_dataset, reference_type)
    return value


def validate_forecast(
    db_path,
    forecast_id: str,
    predictions: list[tuple[int, str, float]],
    reference_frame: pd.DataFrame,
    reference_dataset: str,
    reference_type: str = "operational analysis/forecast",
) -> dict:
    """Validate all leads of one forecast record.

    Parameters
    ----------
    predictions : list of (lead, target_time_utc, prediction_m) tuples.

    Returns
    -------
    dict mapping lead → matched reference value (or None).
    """
    results = {}
    for lead, target_time, prediction in predictions:
        try:
            val = validate_one(
                db_path, forecast_id, lead, target_time, prediction,
                reference_frame, reference_dataset, reference_type,
            )
            results[lead] = val
        except Exception as exc:
            log.error("Validation error for forecast %s lead %d: %s", forecast_id, lead, exc)
            mark_validation_error(db_path, forecast_id, lead, str(exc))
            results[lead] = None
    return results
