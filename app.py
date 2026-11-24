"""INCOIS Sea Surface Height (SSH) Forecasting System.

Operational & Reanalysis Research Dashboard powered by Copernicus Marine GLORYS Data.
Provides direct multi-step forecasting (Persistence Baseline & Multi-Output Random Forest)
at configurable ocean & coastal locations across the Bay of Bengal & Indian Ocean.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# Project imports — settings and analytical pipeline
from config.settings import (
    settings,
    REANALYSIS_PRODUCT,
    REANALYSIS_DATASET,
    OPERATIONAL_PRODUCT,
    OPERATIONAL_DATASET,
    INPUT_WINDOW,
    FORECAST_HORIZON,
    SSH_VARIABLE,
)
from src import database
from src.backtesting import rolling_backtest, summarize_backtest
from src.copernicus_client import CopernicusDataError, fetch_ssh
from src.data_preprocessing import latest_input_window
from src.forecasting import forecast, persistence_forecast, train_random_forest
from src.visualization import (
    COASTAL_PRESETS,
    build_ocean_map,
    feature_importance_chart,
    forecast_chart,
    location_map_html,
    metrics_bar_chart,
    seven_plus_three_chart,
    ssh_history_chart,
    validation_scatter,
)

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="INCOIS | Sea Surface Height Forecasting",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

log = logging.getLogger(__name__)

# Initialise Database
database.initialize(settings.database_path)

# ---------------------------------------------------------------------------
# Professional Oceanographic CSS Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main container styling */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Top Header */
    .incois-header {
        background: linear-gradient(135deg, #0b2545 0%, #134074 50%, #006494 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
    }
    .incois-header h1 {
        color: #ffffff;
        font-size: 1.95rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .incois-header p {
        color: #e0f2fe;
        font-size: 0.98rem;
        margin: 0;
        line-height: 1.5;
    }
    
    /* Section Headings */
    .section-badge {
        display: inline-block;
        background-color: #006494;
        color: white;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        margin-right: 10px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .section-title {
        color: #0b2545;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
    }

    /* Cards */
    .info-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #006494;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .metric-card-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-card-value {
        font-size: 1.12rem;
        font-weight: 700;
        color: #0f172a;
        word-break: break-all;
    }
    .metric-card-sub {
        font-size: 0.78rem;
        color: #475569;
        margin-top: 4px;
    }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.84rem;
        font-weight: 600;
    }
    .status-success {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .status-pending {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .status-idle {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
    }
    
    /* Buttons */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        border-color: #006494;
        color: #006494;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
_STATE_DEFAULTS = {
    "lat": settings.latitude,
    "lon": settings.longitude,
    "selected_preset": "Visakhapatnam Coast (Offshore)",
    "ssh_frame": None,
    "ssh_location": None,
    "ssh_meta": None,
    "retrieval_status": "Idle — Awaiting User Request",
    "last_fetch_utc": None,
    "history_frame": None,
    "history_location": None,
    "history_key": None,
    "last_history_utc": None,
    "backtest_results": None,
    "forecast_id": None,
    "rf_model": None,
    "rf_training": None,
    "persist_preds": None,
    "rf_preds": None,
    "forecast_dates": None,
    "forecast_model_name": None,
}
for key, default in _STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# ---------------------------------------------------------------------------
# Sidebar: Ocean Location & Pipeline Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌊 Ocean Station & Controls")

    # 1. Preset Location Selector
    preset_names = [p["name"] for p in COASTAL_PRESETS] + ["Custom Coordinates"]
    
    # Callback to update coordinates when preset changes
    def _on_preset_change():
        chosen = st.session_state.preset_selector
        for p in COASTAL_PRESETS:
            if p["name"] == chosen:
                st.session_state.lat = p["lat"]
                st.session_state.lon = p["lon"]
                break

    selected_preset = st.selectbox(
        "📍 Coastal / Ocean Location Preset",
        preset_names,
        index=1 if "Visakhapatnam Coast (Offshore)" in preset_names else 0,
        key="preset_selector",
        on_change=_on_preset_change,
        help="Quickly pick a seacoast or ocean location in the Indian Ocean / Bay of Bengal basin.",
    )

    # 2. Latitude & Longitude Inputs (Full Precision)
    c_lat, c_lon = st.columns(2)
    with c_lat:
        lat = st.number_input(
            "Latitude (°N)",
            min_value=-90.0,
            max_value=90.0,
            value=float(st.session_state.lat),
            step=0.083,
            format="%.4f",
            key="lat_input",
            help="Configurable ocean latitude.",
        )
        st.session_state.lat = lat
    with c_lon:
        lon = st.number_input(
            "Longitude (°E)",
            min_value=-180.0,
            max_value=180.0,
            value=float(st.session_state.lon),
            step=0.083,
            format="%.4f",
            key="lon_input",
            help="Configurable ocean longitude.",
        )
        st.session_state.lon = lon

    # 3. Product & Dataset Selection
    st.markdown("---")
    st.markdown("### 📡 Copernicus Marine Source")
    source_choice = st.selectbox(
        "Dataset Category",
        ["Operational Analysis & Forecast (Recent)", "Historical Multi-Year Reanalysis (Training)"],
        index=0,
        help="Operational dataset provides recent observations & forecast comparisons. Reanalysis provides long historical records for model training.",
    )

    if source_choice.startswith("Operational"):
        active_product = OPERATIONAL_PRODUCT
        active_dataset = OPERATIONAL_DATASET
        source_label = "Operational Analysis/Forecast"
    else:
        active_product = REANALYSIS_PRODUCT
        active_dataset = REANALYSIS_DATASET
        source_label = "Historical GLORYS Reanalysis"

    st.caption(f"**Variable:** `{SSH_VARIABLE}` (Daily Mean SSH above Geoid, metres)")

    # 4. Pipeline Execution Actions
    st.markdown("---")
    st.markdown("### ⚡ Pipeline Operations")

    btn_fetch = st.button("🌐 1. Fetch Daily SSH Data", use_container_width=True, help="Download recent daily SSH subset for the selected location.")
    btn_gen = st.button("🔮 2. Generate 3-Day Forecast", use_container_width=True, help="Compute Persistence Baseline & train/predict with Random Forest.")
    
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    with st.expander("🛠️ Advanced Training & Validation Actions"):
        btn_history = st.button("📚 Fetch Reanalysis History", use_container_width=True, help="Download multi-month/year reanalysis for expanding-origin training.")
        history_start = st.date_input(
            "Reanalysis Start Date",
            value=datetime(2020, 1, 1).date(),
            help="Beginning date for historical training observations.",
        )
        btn_backtest = st.button("📊 Run Historical Backtest", use_container_width=True, help="Evaluate multi-origin rolling backtest metrics.")
        btn_validate = st.button("✅ Refresh Pending Validations", use_container_width=True, help="Compare pending ledger forecasts with new operational reference data.")

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size: 0.8rem; color: #64748b; line-height: 1.4;'>
        <b>System Assurance:</b><br>
        • Actual Copernicus Marine GLORYS data only.<br>
        • Zero fabricated or simulated observations.<br>
        • Authenticated via Copernicus Marine credentials.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="incois-header">
        <h1>🌊 AI-Based Sea Surface Height (SSH) Forecasting System</h1>
        <p>Operational Research Prototype associated with INCOIS (Indian National Centre for Ocean Information Services).
        Direct multi-step forecasting at configurable ocean locations across the Indian Ocean & Bay of Bengal using Copernicus Marine GLORYS data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Backend Action Handlers (Button-Gated Only, Free Reruns)
# ---------------------------------------------------------------------------

# Action 1: Fetch Daily SSH Data
if btn_fetch:
    with st.spinner(f"Retrieving ocean subset from Copernicus Marine for ({lat:.4f}°N, {lon:.4f}°E)…"):
        end_time = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
        start_time = end_time - pd.Timedelta(days=35)
        try:
            df, loc_info, meta_info = fetch_ssh(lat, lon, start_time, end_time, active_dataset)
            st.session_state.ssh_frame = df
            st.session_state.ssh_location = loc_info
            st.session_state.ssh_meta = meta_info
            st.session_state.last_fetch_utc = _now_utc_str()
            st.session_state.retrieval_status = "Success — Ocean Data Loaded"
            st.toast(f"Successfully retrieved {len(df)} daily observations!", icon="🌊")
        except CopernicusDataError as c_err:
            st.session_state.retrieval_status = f"Failed: {c_err}"
            st.error(f"**Data Access Notice:** {c_err}")
            if "authentication" in str(c_err).lower() or "login" in str(c_err).lower():
                st.info("💡 **Copernicus Authentication Required:** Run `copernicusmarine login` in a terminal or configure environment variables.")
        except Exception as exc:
            st.session_state.retrieval_status = f"Error: {exc}"
            st.error(f"Unexpected retrieval issue: {exc}")
            log.exception("Retrieval failure")

# Action 2: Fetch Historical Reanalysis
if 'btn_history' in locals() and btn_history:
    with st.spinner("Downloading multi-year historical reanalysis for model training…"):
        end_time = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
        try:
            hf, hl, hm = fetch_ssh(lat, lon, pd.Timestamp(history_start, tz="UTC"), end_time, REANALYSIS_DATASET)
            st.session_state.history_frame = hf
            st.session_state.history_location = hl
            st.session_state.history_key = (float(lat), float(lon))
            st.session_state.last_history_utc = _now_utc_str()
            st.success(f"Loaded {len(hf)} historical reanalysis records from {hf['time'].min().date()} to {hf['time'].max().date()}!")
        except Exception as exc:
            st.error(f"Historical reanalysis fetch error: {exc}")

# Action 3: Generate Forecast
if btn_gen:
    frame = st.session_state.ssh_frame
    if frame is None or len(frame) == 0:
        st.warning("⚠️ Please fetch daily ocean data first by clicking '**1. Fetch Daily SSH Data**' in the sidebar.")
    else:
        try:
            # Validate and extract consecutive 7-day input window
            window = latest_input_window(frame, window=INPUT_WINDOW)
            with st.spinner("Training Random Forest Regressor & generating 3-day multi-step forecasts…"):
                # Determine training pool
                loc_info = st.session_state.ssh_location
                history_pool = []
                if st.session_state.history_frame is not None and st.session_state.history_key == (float(lat), float(lon)):
                    history_pool = st.session_state.history_frame["ssh_m"].to_numpy()
                elif active_dataset == REANALYSIS_DATASET:
                    history_pool = frame["ssh_m"].to_numpy()

                rf_mod, rf_info = train_random_forest(list(history_pool))
                st.session_state.rf_model = rf_mod
                st.session_state.rf_training = rf_info

                input_vals = window["ssh_m"].to_numpy()
                persist_res = forecast(input_vals, model=None, horizon=FORECAST_HORIZON)
                rf_res = forecast(input_vals, model=rf_mod, horizon=FORECAST_HORIZON) if rf_mod is not None else None

                forecast_dates = pd.date_range(
                    window["time"].iloc[-1] + pd.Timedelta(days=1),
                    periods=FORECAST_HORIZON,
                    freq="D",
                    tz="UTC",
                )

                st.session_state.persist_preds = persist_res.predictions
                st.session_state.rf_preds = rf_res.predictions if rf_res is not None else None
                st.session_state.forecast_dates = forecast_dates
                st.session_state.forecast_model_name = rf_res.name if rf_res is not None else "Persistence"

                # Persist to SQLite Forecast Ledger
                chosen_model = rf_res if rf_res is not None else persist_res
                pred_series = pd.Series(chosen_model.predictions, index=forecast_dates)
                fid = database.save_forecast(
                    settings.database_path,
                    source=st.session_state.ssh_meta.get("source", "Copernicus Marine Service"),
                    dataset_id=active_dataset,
                    requested_lat=lat,
                    requested_lon=lon,
                    grid_lat=loc_info.grid_latitude if loc_info else lat,
                    grid_lon=loc_info.grid_longitude if loc_info else lon,
                    inputs=window,
                    predictions=pred_series,
                    model_name=chosen_model.name,
                    training=rf_info,
                )
                st.session_state.forecast_id = fid

                # Save joblib model if trained
                if rf_mod is not None:
                    m_path = settings.models_path / f"rf_{fid}.joblib"
                    m_path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(rf_mod, m_path)

                st.toast(f"Forecast successfully computed & saved (ID: {fid[:8]}…)", icon="🎯")
        except ValueError as v_err:
            st.error(f"**Input Window Requirement:** {v_err}")
        except Exception as exc:
            st.error(f"Forecasting pipeline error: {exc}")
            log.exception("Forecast execution error")

# Action 4: Run Historical Backtest
if 'btn_backtest' in locals() and btn_backtest:
    frame_to_test = (
        st.session_state.history_frame
        if st.session_state.history_frame is not None and st.session_state.history_key == (float(lat), float(lon))
        else st.session_state.ssh_frame
    )
    if frame_to_test is None or len(frame_to_test) < 65:
        st.warning("Backtesting requires at least 65 consecutive daily observations. Please use '**Fetch Reanalysis History**' to load a multi-year training period.")
    else:
        with st.spinner("Executing chronological expanding-origin backtesting (Persistence vs Random Forest)…"):
            try:
                bt_results = rolling_backtest(frame_to_test)
                st.session_state.backtest_results = bt_results
                st.success(f"Backtesting finished across {len(bt_results) // 3} chronological forecast origins!")
            except Exception as b_err:
                st.error(f"Backtest error: {b_err}")

# Action 5: Refresh Pending Validations
if 'btn_validate' in locals() and btn_validate:
    pending_df = database.get_pending_validations(settings.database_path)
    if pending_df.empty:
        st.info("No pending validation entries found in the ledger.")
    else:
        with st.spinner("Checking operational data for matching reference timestamps…"):
            from src.validation import validate_forecast
            from src.database import mark_validation_error as _mve
            all_fc = database.list_forecasts(settings.database_path)
            for f_id, grp in pending_df.groupby("forecast_id"):
                fc_match = all_fc.loc[all_fc["forecast_id"] == f_id]
                if fc_match.empty:
                    continue
                fc_row = fc_match.iloc[0]
                t_series = pd.to_datetime(grp["target_time_utc"], utc=True)
                try:
                    refs, _, _ = fetch_ssh(
                        float(fc_row["grid_lat"]), float(fc_row["grid_lon"]),
                        t_series.min(), t_series.max(),
                        OPERATIONAL_DATASET,
                        search_radius=0.05,
                    )
                    pred_tuples = [
                        (int(r["lead"]), r["target_time_utc"], float(r["prediction_m"]))
                        for _, r in grp.iterrows()
                    ]
                    ref_label = "forecast-to-forecast" if any(
                        pd.Timestamp(t) > pd.Timestamp.now(tz="UTC") for t in grp["target_time_utc"]
                    ) else "operational analysis; not independent observation"
                    validate_forecast(settings.database_path, f_id, pred_tuples, refs, OPERATIONAL_DATASET, ref_label)
                except CopernicusDataError as c_err:
                    for _, it in grp.iterrows():
                        _mve(settings.database_path, f_id, int(it["lead"]), str(c_err))
            st.rerun()

# ---------------------------------------------------------------------------
# Section 1: Project Overview & System Metadata
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title"><span class="section-badge">01</span>Project Overview & System Status</div>', unsafe_allow_html=True)

# Overview Expandable Card
with st.expander("📖 Oceanographic Methodology & Data Verification Summary", expanded=False):
    st.markdown(
        f"""
        **System Objectives:**
        1. Access official Copernicus Marine GLORYS products for sea surface height (variable `{SSH_VARIABLE}`, daily mean SSH in metres).
        2. Extract consecutive seven-day observations for a configurable grid coordinate in the Indian Ocean & Bay of Bengal.
        3. Execute direct multi-step forecasting for the next three days using a Persistence Baseline and a Multi-Output Random Forest Regressor.
        4. Maintain an immutable SQLite audit ledger with non-destructive validation tracking.

        **Verified Copernicus Marine Product Identifiers:**
        - **Operational Product:** `{OPERATIONAL_PRODUCT}`
        - **Operational Daily Dataset:** `{OPERATIONAL_DATASET}`
        - **Historical Reanalysis Product:** `{REANALYSIS_PRODUCT}`
        - **Historical Daily Dataset:** `{REANALYSIS_DATASET}`

        **Oceanographic Research Caveats:**
        - A single 1/12° grid cell (~9 km) represents a localized ocean point and not the entire Bay of Bengal basin.
        - Operational reference data containing future dates represents a *forecast-to-forecast* comparison, not independent ground-truth validation.
        - Metrics measure agreement with the GLORYS numerical model output.
        """
    )

# System Metadata Metrics Grid (Responsive, completely visible, no truncation)
loc_info = st.session_state.ssh_location
frame = st.session_state.ssh_frame

m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-label">Selected Ocean Point</div>
            <div class="metric-card-value">{lat:.4f}° N, {lon:.4f}° E</div>
            <div class="metric-card-sub">Location: <b>{selected_preset}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_col2:
    grid_coords = f"{loc_info.grid_latitude:.4f}° N, {loc_info.grid_longitude:.4f}° E" if loc_info else "Pending Fetch"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-label">Copernicus Grid Cell</div>
            <div class="metric-card-value">{grid_coords}</div>
            <div class="metric-card-sub">Resolution: ~1/12° (~9 km)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_col3:
    latest_obs = frame["time"].max().strftime("%Y-%m-%d %H:%M UTC") if frame is not None and not frame.empty else "No Data Loaded"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-label">Latest Valid Observation</div>
            <div class="metric-card-value">{latest_obs}</div>
            <div class="metric-card-sub">Last Fetch: {st.session_state.last_fetch_utc or "None"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_col4:
    retrieval_stat = st.session_state.retrieval_status
    status_cls = "status-success" if "Success" in retrieval_stat else ("status-pending" if "Idle" in retrieval_stat else "status-idle")
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-label">Data Retrieval Status</div>
            <div class="metric-card-value" style="font-size: 0.95rem; line-height: 1.4;">
                <span class="status-badge {status_cls}">{retrieval_stat}</span>
            </div>
            <div class="metric-card-sub">Dataset: <code>{active_dataset}</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Section 2: Ocean Location Map
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title"><span class="section-badge">02</span>Ocean Location Map (Bay of Bengal & Indian Ocean)</div>', unsafe_allow_html=True)

map_col, details_col = st.columns([3, 1])

with map_col:
    grid_lat_val = loc_info.grid_latitude if loc_info else None
    grid_lon_val = loc_info.grid_longitude if loc_info else None
    
    # Render interactive Folium Map
    folium_map = build_ocean_map(lat, lon, grid_lat_val, grid_lon_val, location_name=selected_preset)
    if folium_map is not None:
        map_out = st_folium(
            folium_map,
            height=460,
            use_container_width=True,
            returned_objects=["last_clicked"],
            key="ocean_folium_map",
        )
        # Capture clicks on map to set custom coordinates
        if map_out and map_out.get("last_clicked"):
            c_lat_click = round(map_out["last_clicked"]["lat"], 4)
            c_lon_click = round(map_out["last_clicked"]["lng"], 4)
            if abs(c_lat_click - lat) > 0.001 or abs(c_lon_click - lon) > 0.001:
                st.info(f"📍 **Map Click Detected:** `{c_lat_click:.4f}°N, {c_lon_click:.4f}°E`. Click button below to update target location:")
                if st.button("Apply Clicked Coordinates to Forecast Target", key="apply_click_coords"):
                    st.session_state.lat = c_lat_click
                    st.session_state.lon = c_lon_click
                    st.session_state.preset_selector = "Custom Coordinates"
                    st.rerun()
    else:
        st.warning("Folium map rendering is unavailable.")

with details_col:
    st.markdown("#### 🧭 Coordinates & Grid")
    glat_str = f"{grid_lat_val:.4f}° N" if grid_lat_val is not None else "Awaiting fetch"
    glon_str = f"{grid_lon_val:.4f}° E" if grid_lon_val is not None else "Awaiting fetch"
    st.markdown(
        f"""
        **Target Point:**
        - **Lat:** `{lat:.4f}° N`
        - **Lon:** `{lon:.4f}° E`
        - **Preset:** *{selected_preset}*
        
        **Copernicus Cell:**
        - **Grid Lat:** `{glat_str}`
        - **Grid Lon:** `{glon_str}`
        """
    )
    if loc_info:
        offset_lat = abs(lat - loc_info.grid_latitude)
        offset_lon = abs(lon - loc_info.grid_longitude)
        if offset_lat > 0.001 or offset_lon > 0.001:
            st.caption(f"ℹ️ Snapped to nearest wet ocean cell. Spatial shift: ΔLat={offset_lat:.3f}°, ΔLon={offset_lon:.3f}°.")
    else:
        st.caption("Click **'1. Fetch Daily SSH Data'** in the sidebar to verify ocean cell wetness and snap to the Copernicus grid.")

# ---------------------------------------------------------------------------
# Section 3: Historical Seven-Day SSH Data
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title"><span class="section-badge">03</span>Historical Seven-Day SSH Data (Model Inputs)</div>', unsafe_allow_html=True)

if frame is not None and not frame.empty:
    window_df = None
    try:
        window_df = latest_input_window(frame, window=INPUT_WINDOW)
    except ValueError as val_err:
        st.warning(f"⚠️ Cannot form 7-day input sequence: {val_err}")

    # Tabs for Data View & History Chart
    tab_seq, tab_full = st.tabs(["📋 Consecutive 7-Day Input Sequence", "📈 Full 30-Day Retrieval History"])

    with tab_seq:
        if window_df is not None:
            w_disp = window_df.copy()
            w_disp["Observation Date (UTC)"] = pd.to_datetime(w_disp["time"]).dt.strftime("%Y-%m-%d")
            w_disp["Day Index"] = [f"T - {INPUT_WINDOW - 1 - i}" if i < INPUT_WINDOW - 1 else "T = 0 (Origin)" for i in range(len(w_disp))]
            w_disp["Sea Surface Height (m)"] = w_disp["ssh_m"].apply(lambda v: f"{v:+.4f}")
            w_disp["Data Source"] = active_dataset
            st.dataframe(w_disp[["Day Index", "Observation Date (UTC)", "Sea Surface Height (m)", "Data Source"]], use_container_width=True, hide_index=True)
            st.caption(f"✅ Exactly {INPUT_WINDOW} consecutive daily SSH observations verified. Forecast origin is {w_disp['Observation Date (UTC)'].iloc[-1]}.")
        else:
            st.info("7-day consecutive sequence not yet available.")

    with tab_full:
        fig_full_hist = ssh_history_chart(frame, window=window_df, title=f"Daily Mean SSH at {lat:.3f}°N, {lon:.3f}°E ({active_dataset})")
        st.plotly_chart(fig_full_hist, use_container_width=True)
else:
    st.info("ℹ️ No ocean observations currently loaded. Click '**1. Fetch Daily SSH Data**' in the sidebar to retrieve the latest GLORYS dataset.")

# ---------------------------------------------------------------------------
# Section 4: Three-Day SSH Forecast
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title"><span class="section-badge">04</span>Three-Day SSH Forecast (Direct Multi-Step)</div>', unsafe_allow_html=True)

if st.session_state.forecast_dates is not None and window_df is not None:
    fc_dates = st.session_state.forecast_dates
    persist_vals = st.session_state.persist_preds
    rf_vals = st.session_state.rf_preds
    primary_model_name = st.session_state.forecast_model_name

    # 1. Professional Focused 7 + 3 Time-Series Chart
    fig_73 = seven_plus_three_chart(
        window=window_df,
        forecast_dates=fc_dates,
        persist_preds=persist_vals,
        rf_preds=rf_vals,
        title=f"Sea Surface Height: 7-Day Historical Observations + 3-Day Forecast Horizon (Lead +1, +2, +3)",
    )
    st.plotly_chart(fig_73, use_container_width=True)

    # 2. Tabular Forecast Results
    fc_table_rows = []
    for i, f_date in enumerate(fc_dates):
        p_val = persist_vals[i]
        rf_val = rf_vals[i] if rf_vals is not None else np.nan
        fc_table_rows.append({
            "Forecast Lead": f"Day {i+1} (+{i+1}d)",
            "Target Date (UTC)": f_date.strftime("%Y-%m-%d"),
            "Persistence Baseline (m)": f"{p_val:+.4f}",
            "Random Forest ML (m)": f"{rf_val:+.4f}" if not np.isnan(rf_val) else "N/A (Insufficient History)",
            "Active Selection": f"{rf_val:+.4f}" if rf_vals is not None else f"{p_val:+.4f}",
            "Primary Model": primary_model_name,
        })
    st.dataframe(pd.DataFrame(fc_table_rows), use_container_width=True, hide_index=True)
    st.caption("Dashed lines and diamond/square points represent predicted values. Shaded area marks the 3-day multi-step forecast window.")
else:
    st.info("ℹ️ Forecast has not yet been generated. Once ocean data is loaded, click '**2. Generate 3-Day Forecast**' in the sidebar.")

# ---------------------------------------------------------------------------
# Section 5: Model Comparison
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title"><span class="section-badge">05</span>Model Comparison (Persistence Baseline vs Random Forest)</div>', unsafe_allow_html=True)

if st.session_state.forecast_dates is not None and window_df is not None:
    cmp1, cmp2 = st.columns(2)
    with cmp1:
        st.markdown(
            f"""
            <div class="info-card">
                <h4 style="margin: 0 0 8px 0; color: #d62728;">Persistence Baseline Model</h4>
                <p style="font-size: 0.9rem; margin: 0 0 10px 0;">Assumes ocean sea surface height remains identical to the latest observed measurement across all forecast leads.</p>
                <b>Forecast Day 1:</b> {persist_vals[0]:+.4f} m<br>
                <b>Forecast Day 2:</b> {persist_vals[1]:+.4f} m<br>
                <b>Forecast Day 3:</b> {persist_vals[2]:+.4f} m<br>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 8px;">Complexity: O(1) · Standard oceanographic benchmark</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cmp2:
        rf_status_str = st.session_state.rf_training.get("status", "N/A") if st.session_state.rf_training else "Untrained"
        rf_d1 = f"{rf_vals[0]:+.4f} m" if rf_vals is not None else "N/A"
        rf_d2 = f"{rf_vals[1]:+.4f} m" if rf_vals is not None else "N/A"
        rf_d3 = f"{rf_vals[2]:+.4f} m" if rf_vals is not None else "N/A"
        st.markdown(
            f"""
            <div class="info-card">
                <h4 style="margin: 0 0 8px 0; color: #2ca02c;">Multi-Output Random Forest Regressor</h4>
                <p style="font-size: 0.9rem; margin: 0 0 10px 0;">Direct 3-step regressor trained on chronological rolling windows with purged train/val splits to prevent data leakage.</p>
                <b>Forecast Day 1:</b> {rf_d1}<br>
                <b>Forecast Day 2:</b> {rf_d2}<br>
                <b>Forecast Day 3:</b> {rf_d3}<br>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 8px;">Status: <b>{rf_status_str}</b> · Estimators: 100 · Min Samples: 20</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Feature Importance Chart if Random Forest was trained
    if st.session_state.rf_training and st.session_state.rf_training.get("feature_importances"):
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.plotly_chart(
            feature_importance_chart(st.session_state.rf_training["feature_importances"]),
            use_container_width=True,
        )
else:
    st.info("ℹ️ Run the forecast generator to view side-by-side model predictions.")

# ---------------------------------------------------------------------------
# Section 6: Forecast Validation & Ledger
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title"><span class="section-badge">06</span>Forecast Validation & Verification Ledger</div>', unsafe_allow_html=True)

forecasts_df = database.list_forecasts(settings.database_path)
validations_df = database.list_validations(settings.database_path)

if not forecasts_df.empty:
    val_tab1, val_tab2, val_tab3 = st.tabs(["📋 Forecast Ledger History", "🔍 Lead Validation Status", "📉 Predicted vs Reference Scatter"])

    with val_tab1:
        disp_fc = forecasts_df[[
            "forecast_id", "generated_at_utc", "dataset_id", "requested_lat", "requested_lon",
            "grid_lat", "grid_lon", "model_name", "validation_status"
        ]].copy()
        disp_fc.columns = ["Forecast ID", "Generated At (UTC)", "Dataset", "Req Lat", "Req Lon", "Grid Lat", "Grid Lon", "Model", "Validation Status"]
        st.dataframe(disp_fc, use_container_width=True, hide_index=True)

    with val_tab2:
        if not validations_df.empty:
            v_cols = [c for c in [
                "forecast_id", "lead", "target_time_utc", "prediction_m", "reference_m",
                "absolute_error_m", "reference_type", "status", "checked_at_utc"
            ] if c in validations_df.columns]
            v_disp = validations_df[v_cols].copy()
            v_disp.columns = ["Forecast ID", "Lead", "Target (UTC)", "Pred (m)", "Ref (m)", "Abs Error (m)", "Ref Type", "Status", "Checked At"]
            st.dataframe(v_disp, use_container_width=True, hide_index=True)
            st.caption("ℹ️ 'Pending' indicates the target date is in the future or Copernicus reference data has not yet been fetched. Operational forecast matches are categorized as forecast-to-forecast.")
        else:
            st.write("No validation records yet.")

    with val_tab3:
        st.plotly_chart(validation_scatter(validations_df), use_container_width=True)

    # Export controls
    exp_c1, exp_c2 = st.columns(2)
    with exp_c1:
        st.download_button(
            "💾 Export Forecast Ledger (CSV)",
            forecasts_df.to_csv(index=False),
            file_name="incois_ssh_forecast_ledger.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp_c2:
        st.download_button(
            "💾 Export Validation Records (CSV)",
            validations_df.to_csv(index=False),
            file_name="incois_ssh_validations.csv",
            mime="text/csv",
            use_container_width=True,
        )
else:
    st.info("No forecasts stored in the SQLite ledger yet.")

# ---------------------------------------------------------------------------
# Section 7: Performance Metrics
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title"><span class="section-badge">07</span>Performance Metrics (MAE & RMSE in Metres)</div>', unsafe_allow_html=True)

# Display metrics only when valid evaluation results exist
bt_results = st.session_state.backtest_results

if bt_results is not None and not bt_results.empty:
    st.caption("Evaluation results computed across chronological expanding-origin backtesting against GLORYS reference observations:")
    b_col1, b_col2 = st.columns(2)

    with b_col1:
        st.markdown("##### 📌 Persistence Baseline Performance")
        p_sum = summarize_backtest(bt_results, "persistence_m")
        if "lead_label" in p_sum:
            st.dataframe(
                p_sum[["lead_label", "mae_m", "rmse_m"]].rename(
                    columns={"lead_label": "Horizon", "mae_m": "MAE (m)", "rmse_m": "RMSE (m)"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        fig_bp = metrics_bar_chart(p_sum, "mae_m", "Persistence Baseline MAE by Lead (m)")
        st.plotly_chart(fig_bp, use_container_width=True)

    with b_col2:
        st.markdown("##### 🌲 Random Forest Regressor Performance")
        rf_sum = summarize_backtest(bt_results, "random_forest_m")
        if "lead_label" in rf_sum:
            st.dataframe(
                rf_sum[["lead_label", "mae_m", "rmse_m"]].rename(
                    columns={"lead_label": "Horizon", "mae_m": "MAE (m)", "rmse_m": "RMSE (m)"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        fig_brf = metrics_bar_chart(rf_sum, "mae_m", "Random Forest Regressor MAE by Lead (m)")
        st.plotly_chart(fig_brf, use_container_width=True)
else:
    # Check if we have validated records in the ledger
    validated_in_ledger = validations_df[pd.to_numeric(validations_df.get("absolute_error_m", pd.Series(dtype=float)), errors="coerce").notna()] if not validations_df.empty else pd.DataFrame()
    if not validated_in_ledger.empty:
        st.markdown("##### 📊 Real-Time Validation Metrics from Ledger")
        errors = validated_in_ledger["absolute_error_m"].astype(float)
        vm1, vm2, vm3 = st.columns(3)
        vm1.metric("Validated Leads", len(errors))
        vm2.metric("Mean Absolute Error (MAE)", f"{errors.mean():.4f} m")
        vm3.metric("Root Mean Squared Error (RMSE)", f"{np.sqrt((errors**2).mean()):.4f} m")
    else:
        st.markdown(
            """
            <div class="info-card">
                <b>Evaluation Status:</b> No backtesting results or validated ground truth reference values exist for this session yet.<br>
                MAE and RMSE are calculated dynamically only when valid historical backtest evaluation or operational reference data matching is performed.<br>
                <i>To calculate evaluation metrics, use '<b>Fetch Reanalysis History</b>' followed by '<b>Run Historical Backtest</b>' in the sidebar.</i>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 12px 0;">
        <b>INCOIS Sea Surface Height (SSH) Forecasting Prototype</b> · Powered by Copernicus Marine GLORYS Reanalysis & Operational Analysis/Forecast<br>
        Direct Multi-Step Forecasting (7-Day Input Window → 3-Day Forecast Horizon) · SQLite Persistent Ledger
    </div>
    """,
    unsafe_allow_html=True,
)
