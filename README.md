# AI-Based Sea Surface Height Forecasting — INCOIS Prototype

A research prototype for direct three-day sea surface height (SSH) forecasting
at a configurable ocean grid point in the Bay of Bengal.  Uses Copernicus
Marine GLORYS datasets via the official `copernicusmarine` Python client.

> **No synthetic or fabricated SSH values are displayed in the dashboard.**
> All forecasts are generated from actual GLORYS data retrieved after
> successful authentication.

---

## Verified dataset identifiers

Checked against the Copernicus Marine catalogue and Product User Manuals
(November 2026):

| Use | Product | Daily dataset | Variable |
|-----|---------|--------------|----------|
| Historical training / backtesting | `GLOBAL_MULTIYEAR_PHY_001_030` | `cmems_mod_glo_phy_my_0.083deg_P1D-m` | `zos` — daily mean SSH above geoid, metres |
| Recent data / reference validation | `GLOBAL_ANALYSISFORECAST_PHY_001_024` | `cmems_mod_glo_phy_anfc_0.083deg_P1D-m` | `zos` — daily mean SSH above geoid, metres |

Both datasets are global at approximately 1/12° resolution.  The operational
product includes both finalized analysis fields and forecast fields; a
comparison of a prediction against an operational **forecast** value is
labelled **forecast-to-forecast** and is not an independent observation.

---

## Project structure

```
incois_ssh_forecasting/
├── app.py                     Streamlit dashboard (8 sections)
├── requirements.txt
├── README.md
├── .env.example               Safe placeholder; never put real credentials here
├── .gitignore
├── conftest.py                Pytest path setup
├── pytest.ini
│
├── config/
│   └── settings.py            Dataset IDs, variable names, path defaults
│
├── data/                      SQLite forecast ledger (git-ignored)
├── models/                    Saved joblib RF models (git-ignored)
│
└── src/
    ├── __init__.py
    ├── copernicus_client.py   Authenticated small-box subset retrieval
    ├── data_preprocessing.py  Cleaning, 7-day window, supervised windows
    ├── location_service.py    Coordinate validation, nearest wet cell
    ├── forecasting.py         Persistence baseline + Random Forest
    ├── backtesting.py         Rolling-origin evaluation, MAE/RMSE
    ├── validation.py          Reference matching, non-destructive update
    ├── database.py            SQLite ledger (immutable forecast rows)
    └── visualization.py       Plotly charts and Folium map helpers

tests/
├── test_preprocessing.py
├── test_forecasting.py
├── test_backtesting.py
├── test_database.py
├── test_validation.py
├── test_location.py
└── test_visualization.py
```

---

## Setup

### 1. Python environment

Python 3.11 or later is required (the `copernicusmarine` toolbox supports
Python 3.10+).

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Copernicus Marine authentication

Create a free account at [marine.copernicus.eu](https://marine.copernicus.eu/).

Store credentials using the official login command (writes to your user
profile — never to this repository):

```powershell
copernicusmarine login
```

Alternatively, export these environment variables in your shell session:

```powershell
$env:COPERNICUSMARINE_SERVICE_USERNAME = "your_username"
$env:COPERNICUSMARINE_SERVICE_PASSWORD = "your_password"
```

Copy `.env.example` to `.env` for non-secret app settings.  **Never put
Copernicus credentials inside `.env` or any file that could be committed.**

### 3. Optional environment settings (`.env`)

```ini
SSH_LATITUDE=12.0           # Bay of Bengal default
SSH_LONGITUDE=88.0
SSH_DATABASE_PATH=data/forecasts.sqlite3
SSH_LOG_LEVEL=INFO
```

---

## Running the dashboard

```powershell
streamlit run app.py
```

The dashboard opens in your browser.  All data-intensive operations
(downloads, model training) are triggered only by explicit button presses —
not on every page reload.

---

## Running the tests

```powershell
pytest -q
```

All 92 tests use synthetic fixtures clearly labelled as such.  No test
requires network access or Copernicus authentication.

---

## Workflow

1. **Fetch / refresh data** — retrieves the last 30 days from the selected
   dataset for the configured grid point (small spatial subset only).
2. **Fetch training history** — downloads GLORYS reanalysis from a
   configurable start date for model training and backtesting.
3. **Generate forecast** — extracts the 7-day input window, trains the
   Random Forest on available history, and saves all three forecast leads
   to the SQLite ledger.
4. **Run backtest** — evaluates both models on historical expanding origins
   and displays per-lead MAE and RMSE.
5. **Refresh pending validations** — retrieves operational reference data for
   any pending forecast leads and records the absolute error.
6. **Export** — download forecasts and validation results as CSV.

---

## Forecasting method

| Aspect | Detail |
|--------|--------|
| Input | 7 consecutive daily SSH values (metres) |
| Output | 3 predicted daily SSH values (direct multi-step) |
| Baseline | Persistence: last observed value repeated for all 3 leads |
| ML model | Multi-output Random Forest Regressor (100 trees) |
| Training | Chronological sliding-window examples; purged train/val split |
| Minimum training examples | 20 (falls back to persistence otherwise) |
| Evaluation | MAE and RMSE in metres; measured against GLORYS reference |

---

## Scientific caveats and limitations

- **One grid point ≠ Bay of Bengal.** A single 1/12° cell is not
  representative of the full basin.
- **GLORYS agreement ≠ independent validation.** Backtesting metrics
  measure how well the model reproduces the GLORYS reference product, not
  accuracy against independent in-situ observations.
- **Forecast-to-forecast comparison.** The operational daily product includes
  forecast fields for future dates. Comparing a prediction against those
  fields is labelled forecast-to-forecast and is not an independent
  observational check.
- **Data latency.** Operational analysis fields typically lag real time by
  1–5 days; true reanalysis data has a longer latency.
- **No gap filling.** Missing or incomplete daily values stop the affected
  operation. No arbitrary interpolation is performed.
- **Model selection.** The Random Forest is not claimed superior to
  persistence unless evaluation results on held-out data confirm it.

---

## Authentication and security

- Credentials are **never** stored in source code, logged, exported, or
  displayed in the dashboard.
- Use `copernicusmarine login` or environment variables only.
- `.env` is git-ignored. `.env.example` contains only safe placeholders.
- SQLite files and NetCDF downloads are git-ignored.
