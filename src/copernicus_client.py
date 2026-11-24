"""Small-area access to Copernicus Marine daily sea surface height."""
from __future__ import annotations
from datetime import datetime, timezone
import logging
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from config.settings import REANALYSIS_DATASET, OPERATIONAL_DATASET, SSH_VARIABLE
from src.location_service import validate_coordinates, OceanLocation

log = logging.getLogger(__name__)

class CopernicusDataError(RuntimeError): pass

def fetch_ssh(latitude: float, longitude: float, start: str | datetime, end: str | datetime,
              dataset_id: str = REANALYSIS_DATASET, search_radius: float = 0.2) -> tuple[pd.DataFrame, OceanLocation, dict]:
    """Fetch zos over a narrow box and return its nearest valid cell time series.

    Requires Copernicus Marine Toolbox authentication. No synthetic fallback exists.
    """
    validate_coordinates(latitude, longitude)
    if dataset_id not in (REANALYSIS_DATASET, OPERATIONAL_DATASET):
        raise ValueError(f"Unconfigured Copernicus dataset: {dataset_id}")
    try:
        import copernicusmarine
        import xarray as xr
    except ImportError as exc:
        raise CopernicusDataError("Install requirements.txt to access Copernicus Marine data.") from exc
    start_dt = pd.Timestamp(start); end_dt = pd.Timestamp(end)
    if start_dt.tzinfo is None: start_dt = start_dt.tz_localize("UTC")
    if end_dt.tzinfo is None: end_dt = end_dt.tz_localize("UTC")
    if end_dt < start_dt: raise ValueError("End time must not precede start time.")
    # Small subset around requested coordinates; Toolbox returns only requested variable.
    with tempfile.TemporaryDirectory(prefix="ssh_subset_") as tmp:
        try:
            result = copernicusmarine.subset(
                dataset_id=dataset_id, variables=[SSH_VARIABLE],
                minimum_longitude=longitude-search_radius, maximum_longitude=longitude+search_radius,
                minimum_latitude=latitude-search_radius, maximum_latitude=latitude+search_radius,
                start_datetime=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                end_datetime=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                output_directory=tmp, disable_progress_bar=True)
            paths = list(Path(tmp).rglob("*.nc"))
            if not paths and result:
                candidate = getattr(result, "output_directory", None)
                if candidate: paths = list(Path(candidate).rglob("*.nc"))
            if not paths: raise CopernicusDataError("Copernicus returned no NetCDF subset for this time range and location.")
            ds = xr.open_mfdataset(paths, combine="by_coords")
            try:
                if SSH_VARIABLE not in ds: raise CopernicusDataError("Selected dataset does not contain zos.")
                units = str(ds[SSH_VARIABLE].attrs.get("units", ""))
                if units and units.lower() not in {"m", "meter", "meters", "metre", "metres"}:
                    raise CopernicusDataError(f"Unexpected zos units {units!r}; expected metres.")
                loc = _choose_cell(ds, latitude, longitude)
                lat_name = next(n for n in ("latitude", "lat") if n in ds.coords)
                lon_name = next(n for n in ("longitude", "lon") if n in ds.coords)
                series = ds[SSH_VARIABLE].sel({lat_name: loc.grid_latitude, lon_name: loc.grid_longitude}, method="nearest")
                for dim in list(series.dims):
                    if dim.lower() in ("depth", "deptht", "lev"): series = series.isel({dim: 0})
                frame = series.to_dataframe(name="ssh_m").reset_index()
                tcol = next((c for c in ("time", "time_counter") if c in frame), None)
                if not tcol: raise CopernicusDataError("Dataset has no recognizable time coordinate.")
                frame = frame.rename(columns={tcol:"time"})[["time","ssh_m"]]
                frame["time"] = pd.to_datetime(frame.time, utc=True)
                frame["ssh_m"] = pd.to_numeric(frame.ssh_m, errors="coerce")
                frame = frame.dropna(subset=["ssh_m"]).sort_values("time").drop_duplicates("time")
                meta = {"dataset_id":dataset_id, "variable":SSH_VARIABLE, "units":units or "m",
                        "temporal_representation":"daily mean", "retrieved_at_utc":datetime.now(timezone.utc).isoformat(),
                        "source":"Copernicus Marine Service", "data_type":"reanalysis" if dataset_id==REANALYSIS_DATASET else "operational analysis/forecast"}
                return frame.reset_index(drop=True), loc, meta
            finally: ds.close()
        except CopernicusDataError: raise
        except Exception as exc:
            msg = str(exc)
            if any(s in msg.lower() for s in ("credential", "login", "unauthorized", "authentication")):
                raise CopernicusDataError("Copernicus authentication failed. Run `copernicusmarine login` and retry.") from exc
            raise CopernicusDataError(f"Copernicus subset failed: {msg}") from exc

def _choose_cell(ds, latitude, longitude):
    from src.location_service import nearest_valid_cell
    return nearest_valid_cell(ds, latitude, longitude, SSH_VARIABLE)
