"""Coordinate validation and explicit grid-cell selection."""
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class OceanLocation:
    requested_latitude: float
    requested_longitude: float
    grid_latitude: float
    grid_longitude: float

def validate_coordinates(latitude: float, longitude: float) -> None:
    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("Latitude must be a finite value between -90 and 90 degrees.")
    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError("Longitude must be a finite value between -180 and 180 degrees.")

def nearest_valid_cell(dataset, latitude: float, longitude: float, variable: str = "zos") -> OceanLocation:
    """Choose closest wet coordinate containing at least one finite variable value."""
    validate_coordinates(latitude, longitude)
    lat_name = next((n for n in ("latitude", "lat") if n in dataset.coords), None)
    lon_name = next((n for n in ("longitude", "lon") if n in dataset.coords), None)
    if not lat_name or not lon_name or variable not in dataset:
        raise ValueError("Dataset lacks expected latitude/longitude coordinates or SSH variable.")
    import numpy as np
    lat = np.asarray(dataset[lat_name].values)
    lon = np.asarray(dataset[lon_name].values)
    da = dataset[variable]
    non_spatial=[d for d in da.dims if d not in (lat_name,lon_name)]
    valid_da=np.isfinite(da)
    if non_spatial: valid_da=valid_da.any(dim=non_spatial)
    da=da.isel({d:0 for d in non_spatial}) if non_spatial else da
    values = np.asarray(da.values)
    if values.ndim != 2:
        raise ValueError("SSH data does not have a two-dimensional surface grid.")
    lat_axis = list(da.dims).index(lat_name); lon_axis = list(da.dims).index(lon_name)
    if (lat_axis, lon_axis) != (0, 1): values = np.moveaxis(values, (lat_axis, lon_axis), (0, 1))
    lat_idx = np.abs(lat - latitude).argmin(); lon_idx = np.abs(lon - longitude).argmin()
    # Search available cells by geographic distance; longitude wrap is handled.
    lon_delta = np.minimum(np.abs(lon - longitude), 360 - np.abs(lon - longitude))
    distances = (lat[:, None] - latitude)**2 + (lon_delta[None, :] * np.cos(np.deg2rad(latitude)))**2
    valid = np.asarray(valid_da.values,dtype=bool)
    if not valid.any(): raise ValueError("The requested subset contains no valid ocean cells.")
    distances[~valid] = np.inf
    i, j = np.unravel_index(np.argmin(distances), distances.shape)
    return OceanLocation(latitude, longitude, float(lat[i]), float(lon[j]))
