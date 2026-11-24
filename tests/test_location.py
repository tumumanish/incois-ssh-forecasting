"""Tests for coordinate validation and nearest-cell selection.

All fixtures are synthetic xarray Datasets.
"""
import numpy as np
import pytest
import xarray as xr

from src.location_service import OceanLocation, nearest_valid_cell, validate_coordinates


# ---------------------------------------------------------------------------
# validate_coordinates
# ---------------------------------------------------------------------------

class TestValidateCoordinates:
    def test_valid_bay_of_bengal(self):
        validate_coordinates(12.0, 88.0)   # should not raise

    def test_latitude_out_of_range_raises(self):
        with pytest.raises(ValueError):
            validate_coordinates(91.0, 88.0)

    def test_negative_latitude_ok(self):
        validate_coordinates(-45.0, 88.0)

    def test_longitude_out_of_range_raises(self):
        with pytest.raises(ValueError):
            validate_coordinates(12.0, 185.0)

    def test_nan_latitude_raises(self):
        with pytest.raises(ValueError):
            validate_coordinates(float("nan"), 88.0)

    def test_inf_longitude_raises(self):
        with pytest.raises(ValueError):
            validate_coordinates(12.0, float("inf"))


# ---------------------------------------------------------------------------
# nearest_valid_cell
# ---------------------------------------------------------------------------

def _make_ds(lats, lons, data, time_steps=1, variable="zos"):
    """Helper to build a minimal synthetic xr.Dataset."""
    if time_steps > 1:
        arr = np.stack([data] * time_steps, axis=0)
        return xr.Dataset(
            {variable: (("time", "latitude", "longitude"), arr)},
            coords={"time": range(time_steps), "latitude": lats, "longitude": lons},
        )
    return xr.Dataset(
        {variable: (("latitude", "longitude"), data)},
        coords={"latitude": lats, "longitude": lons},
    )


class TestNearestValidCell:
    def test_returns_ocean_location(self):
        ds = _make_ds([12.0], [88.0], [[1.0]])
        cell = nearest_valid_cell(ds, 12.0, 88.0)
        assert isinstance(cell, OceanLocation)

    def test_selects_nearest_non_nan_cell(self):
        # lon 88.0 is NaN; lon 88.1 is valid — must pick 88.1
        ds = _make_ds([12.0], [88.0, 88.1], [[np.nan, 2.0]])
        cell = nearest_valid_cell(ds, 12.0, 88.0)
        assert cell.grid_longitude == pytest.approx(88.1)

    def test_uses_any_valid_time_step(self):
        # First time step has NaN at [0,0]; second has valid value — cell should be found
        data = np.array([[[np.nan, 2.0]], [[3.0, 4.0]]])
        ds = xr.Dataset(
            {"zos": (("time", "latitude", "longitude"), data)},
            coords={"time": [0, 1], "latitude": [12.0], "longitude": [88.0, 88.1]},
        )
        cell = nearest_valid_cell(ds, 12.0, 88.0)
        assert cell.grid_latitude == pytest.approx(12.0)
        assert cell.grid_longitude == pytest.approx(88.0)

    def test_all_nan_raises(self):
        ds = _make_ds([12.0], [88.0], [[np.nan]])
        with pytest.raises(ValueError, match="no valid ocean cells"):
            nearest_valid_cell(ds, 12.0, 88.0)

    def test_requested_coords_recorded(self):
        ds = _make_ds([12.0, 12.083], [88.0, 88.083], np.ones((2, 2)))
        cell = nearest_valid_cell(ds, 12.04, 88.04)
        assert cell.requested_latitude == pytest.approx(12.04)
        assert cell.requested_longitude == pytest.approx(88.04)

    def test_missing_coordinate_raises(self):
        ds = xr.Dataset({"zos": (("x", "y"), [[1.0]])}, coords={"x": [0], "y": [0]})
        with pytest.raises(ValueError):
            nearest_valid_cell(ds, 12.0, 88.0)
