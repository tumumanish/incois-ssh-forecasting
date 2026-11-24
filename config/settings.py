"""Configuration constants and environment-backed settings."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Copernicus Marine dataset identifiers (verified against catalogue 2026-11)
# ---------------------------------------------------------------------------
REANALYSIS_PRODUCT   = "GLOBAL_MULTIYEAR_PHY_001_030"
REANALYSIS_DATASET   = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

OPERATIONAL_PRODUCT  = "GLOBAL_ANALYSISFORECAST_PHY_001_024"
OPERATIONAL_DATASET  = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"

SSH_VARIABLE         = "zos"          # Sea surface height above geoid, metres

INPUT_WINDOW         = 7             # Days of SSH used as model input
FORECAST_HORIZON     = 3            # Forecast days generated per run
MIN_RF_SAMPLES       = 20           # Min supervised examples for Random Forest
RF_RANDOM_STATE      = 17

MODEL_VERSION        = "1.0"


@dataclass(frozen=True)
class Settings:
    latitude:      float = float(os.getenv("SSH_LATITUDE", "12.0"))
    longitude:     float = float(os.getenv("SSH_LONGITUDE", "88.0"))
    database_path: Path  = field(default_factory=lambda: ROOT / os.getenv("SSH_DATABASE_PATH", "data/forecasts.sqlite3"))
    models_path:   Path  = field(default_factory=lambda: ROOT / "models")
    log_level:     str   = os.getenv("SSH_LOG_LEVEL", "INFO")


settings = Settings()
