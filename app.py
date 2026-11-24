"""INCOIS Sea Surface Height (SSH) Forecasting System.

Operational oceanographic research prototype for direct multi-step sea surface
height forecasting across the Indian Ocean and Bay of Bengal coastline.
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

# Project imports — settings, persistence, and predictive models
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
from src.location_service import OceanLocation
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
    layout="wide",
    initial_sidebar_state="expanded",
)

log = logging.getLogger(__name__)

# Initialize SQLite database
database.initialize(settings.database_path)

# ---------------------------------------------------------------------------
# Professional Research-Grade CSS Styling (Clean, Zero Useless Emojis)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
    
    /* Institutional Header */
    .incois-header {
        background-color: #0f2b48;
        border-bottom: 3px solid #0284c7;
        color: #ffffff;
        padding: 22px 28px;
        border-radius: 6px;
        margin-bottom: 22px;
    }
    .incois-header h1 {
        color: #ffffff;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.3px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .incois-header p {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.45;
    }

    /* Structured Section Headings */
    .section-title {
        color: #0f2b48;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 1.6rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        border-bottom: 1.5px solid #cbd5e1;
        padding-bottom: 6px;
    }
    .section-num {
        background-color: #0284c7;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 8px;
        letter-spacing: 0.5px;
    }

    /* Structured Cards */
    .metric-panel {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-top: 3px solid #0284c7;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .metric-panel-label {
        font-size: 0.76rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 4px;
    }
    .metric-panel-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-panel-sub {
        font-size: 0.8rem;
        color: #475569;
        margin-top: 3px;
    }

    /* Trend Interpretation Banner */
    .trend-card-increment {
        background-color: #f0fdf4;
        border: 1.5px solid #16a34a;
        border-radius: 6px;
        padding: 16px 20px;
        margin-top: 12px;
        color: #14532d;
    }
    .trend-card-decrement {
        background-color: #fef2f2;
        border: 1.5px solid #dc2626;
        border-radius: 6px;
        padding: 16px 20px;
        margin-top: 12px;
        color: #7f1d1d;
    }
    .trend-card-neutral {
        background-color: #f8fafc;
        border: 1.5px solid #64748b;
        border-radius: 6px;
        padding: 16px 20px;
        margin-top: 12px;
        color: #1e293b;
    }

    /* Clean Buttons */
    div.stButton > button:first-child {
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.5rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "lat": 17.6500,  # Visakhapatnam Coast default
    "lon": 83.4500,
    "preset_name": "Visakhapatnam Coast (Offshore)",
    "data_mode": "Demonstration Coastal Benchmark (Offline)",
    "ssh_frame": None,
    "ssh_location": None,
    "ssh_meta": None,
    "retrieval_status": "Ready",
    "last_fetch_utc": None,
    "persist_preds": None,
    "rf_preds": None,
    "forecast_dates": None,
    "forecast_model_name": None,
    "rf_model": None,
    "rf_training": None,
    "forecast_id": None,
    "backtest_results": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _generate_synthetic_coastal_observations(lat: float, lon: float, location_name: str) -> tuple[pd.DataFrame, OceanLocation, dict]:
    """Generate a reproducible, clean 35-day synthetic demonstration time series.
    Strictly labeled as a synthetic demonstration fixture for pipeline verification.
    """
    np.random.seed(int(abs(lat * 100 + lon * 10)) % 1000)
    end_dt = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end_dt, periods=35, freq="D", tz="UTC")
    
    # Base sea level around 0.42m with semi-diurnal / tidal anomaly envelope and slight trend
    t_idx = np.arange(len(dates))
    base_signal = 0.420 + 0.05 * np.sin(2 * np.pi * t_idx / 14.0) + 0.02 * np.cos(2 * np.pi * t_idx / 28.0)
    noise = np.random.normal(0, 0.003, size=len(dates))
    ssh_values = np.round(base_signal + noise, 4)

    df = pd.DataFrame({"time": dates, "ssh_m": ssh_values})
    loc = OceanLocation(
        requested_latitude=round(lat, 4),
        requested_longitude=round(lon, 4),
        grid_latitude=round(round(lat / 0.0833) * 0.0833, 4),
        grid_longitude=round(round(lon / 0.0833) * 0.0833, 4),
    )
    meta = {
        "dataset_id": "synthetic_coastal_benchmark_fixture",
        "variable": SSH_VARIABLE,
        "units": "m",
        "temporal_representation": "daily mean",
        "retrieved_at_utc": _now_utc_str(),
        "source": "Synthetic Demonstration Fixture (Offline Pipeline Verification)",
        "data_type": "demonstration_fixture",
    }
    return df, loc, meta

# ---------------------------------------------------------------------------
# Sidebar: Location Presets and Pipeline Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Location & Data Source")

    # Mode Selector: Live Copernicus API vs Demonstration Benchmark
    data_mode = st.radio(
        "Data Access Mode",
        ["Demonstration Coastal Benchmark (Offline)", "Copernicus Marine GLORYS (Live API)"],
        index=0 if st.session_state.data_mode.startswith("Demonstration") else 1,
        help="Use Demonstration Benchmark for immediate offline testing without credentials, or Live API to connect to Copernicus Marine servers.",
    )
    st.session_state.data_mode = data_mode

    # Coastal Presets Dropdown
    preset_names = [p["name"] for p in COASTAL_PRESETS] + ["Custom Coordinates"]
    
    def _handle_preset_change():
        chosen = st.session_state.sidebar_preset
        for p in COASTAL_PRESETS:
            if p["name"] == chosen:
                st.session_state.lat = p["lat"]
                st.session_state.lon = p["lon"]
                st.session_state.preset_name = p["name"]
                break

    selected_preset = st.selectbox(
        "Coastal / Ocean Position",
        preset_names,
        index=1 if "Visakhapatnam Coast (Offshore)" in preset_names else 0,
        key="sidebar_preset",
        on_change=_handle_preset_change,
        help="Select a standard coastal observation station or pick custom coordinates.",
    )

    c_lat, c_lon = st.columns(2)
    with c_lat:
        in_lat = st.number_input(
            "Latitude (deg N)",
            min_value=-90.0,
            max_value=90.0,
            value=float(st.session_state.lat),
            step=0.0833,
            format="%.4f",
            help="Ocean latitude coordinate.",
        )
        st.session_state.lat = in_lat
    with c_lon:
        in_lon = st.number_input(
            "Longitude (deg E)",
            min_value=-180.0,
            max_value=180.0,
            value=float(st.session_state.lon),
            step=0.0833,
            format="%.4f",
            help="Ocean longitude coordinate.",
        )
        st.session_state.lon = in_lon

    st.markdown("---")
    st.markdown("### Copernicus Source Identifiers")
    dataset_category = st.selectbox(
        "Product Configuration",
        ["Operational Analysis & Forecast", "Historical Reanalysis"],
        index=0,
    )
    if dataset_category.startswith("Operational"):
        active_product = OPERATIONAL_PRODUCT
        active_dataset = OPERATIONAL_DATASET
    else:
        active_product = REANALYSIS_PRODUCT
        active_dataset = REANALYSIS_DATASET

    st.caption(f"**Dataset ID:** `{active_dataset}`")
    st.caption(f"**Variable:** `{SSH_VARIABLE}` (Daily mean sea surface height, metres)")

    st.markdown("---")
    btn_load_data = st.button("Load Sea Level Observations for Selected Position", use_container_width=True)

# ---------------------------------------------------------------------------
# Data Loading Handler
# ---------------------------------------------------------------------------
if btn_load_data:
    target_lat = float(st.session_state.lat)
    target_lon = float(st.session_state.lon)
    preset_label = st.session_state.preset_name

    with st.spinner(f"Loading ocean sea surface height records for {preset_label} ({target_lat:.4f} N, {target_lon:.4f} E)..."):
        if data_mode.startswith("Demonstration"):
            df, loc_info, meta_info = _generate_synthetic_coastal_observations(target_lat, target_lon, preset_label)
            st.session_state.ssh_frame = df
            st.session_state.ssh_location = loc_info
            st.session_state.ssh_meta = meta_info
            st.session_state.last_fetch_utc = _now_utc_str()
            st.session_state.retrieval_status = "Loaded (Demonstration Coastal Benchmark)"
            # Clear previous forecast so user can press the predict button fresh
            st.session_state.forecast_dates = None
            st.session_state.persist_preds = None
            st.session_state.rf_preds = None
            st.toast("Sea level observation series loaded successfully.", icon="check")
        else:
            end_t = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
            start_t = end_t - pd.Timedelta(days=35)
            try:
                df, loc_info, meta_info = fetch_ssh(target_lat, target_lon, start_t, end_t, active_dataset)
                st.session_state.ssh_frame = df
                st.session_state.ssh_location = loc_info
                st.session_state.ssh_meta = meta_info
                st.session_state.last_fetch_utc = _now_utc_str()
                st.session_state.retrieval_status = "Loaded (Copernicus Marine Live API)"
                st.session_state.forecast_dates = None
                st.session_state.persist_preds = None
                st.session_state.rf_preds = None
                st.toast("Copernicus Marine daily ocean data loaded.", icon="check")
            except CopernicusDataError as c_err:
                st.session_state.retrieval_status = f"Failed: {c_err}"
                st.error(f"Copernicus Marine Error: {c_err}")
                if "authentication" in str(c_err).lower() or "login" in str(c_err).lower():
                    st.info("Run `copernicusmarine login` in a terminal or switch Data Access Mode to 'Demonstration Coastal Benchmark (Offline)' in the sidebar.")
            except Exception as ex:
                st.session_state.retrieval_status = f"Error: {ex}"
                st.error(f"Data retrieval failed: {ex}")

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="incois-header">
        <h1>Indian National Centre for Ocean Information Services (INCOIS)</h1>
        <p>Operational Prototype: AI-Based Sea Surface Height (SSH) Forecasting System<br>
        Direct Multi-Step Forecasting (Past 7 Days Input to Upcoming 3 Days Horizon) with Trajectory Trend Analysis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Section 1: Ocean Coast Selection & Map
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title"><span class="section-num">01</span>Sea Coast Position Selection</div>', unsafe_allow_html=True)

map_col, info_col = st.columns([3, 1])

with map_col:
    curr_lat = float(st.session_state.lat)
    curr_lon = float(st.session_state.lon)
    loc_obj = st.session_state.ssh_location
    g_lat = loc_obj.grid_latitude if loc_obj else None
    g_lon = loc_obj.grid_longitude if loc_obj else None

    # Render interactive map
    m_obj = build_ocean_map(curr_lat, curr_lon, g_lat, g_lon, location_name=st.session_state.preset_name)
    if m_obj is not None:
        map_out = st_folium(
            m_obj,
            height=420,
            use_container_width=True,
            returned_objects=["last_clicked"],
            key="folium_ocean_selector",
        )
        if map_out and map_out.get("last_clicked"):
            c_lat = round(map_out["last_clicked"]["lat"], 4)
            c_lon = round(map_out["last_clicked"]["lng"], 4)
            if abs(c_lat - curr_lat) > 0.001 or abs(c_lon - curr_lon) > 0.001:
                st.info(f"Position clicked on map: {c_lat:.4f} N, {c_lon:.4f} E.")
                if st.button("Set Selected Position as Target Coordinates", key="apply_map_click"):
                    st.session_state.lat = c_lat
                    st.session_state.lon = c_lon
                    st.session_state.preset_name = "Custom Coordinates"
                    st.rerun()

with info_col:
    st.markdown("#### Position Details")
    glat_txt = f"{g_lat:.4f} deg N" if g_lat is not None else "Not resolved yet"
    glon_txt = f"{g_lon:.4f} deg E" if g_lon is not None else "Not resolved yet"
    st.markdown(
        f"""
        **Target Coast:**
        `{st.session_state.preset_name}`
        
        **Coordinates:**
        - Latitude: `{curr_lat:.4f} deg N`
        - Longitude: `{curr_lon:.4f} deg E`
        
        **Grid Cell (GLORYS 1/12 deg):**
        - Lat: `{glat_txt}`
        - Lon: `{glon_txt}`
        
        **Data Source:**
        `{st.session_state.retrieval_status}`
        """
    )
    if st.session_state.ssh_frame is None:
        st.info("Click **'Load Sea Level Observations for Selected Position'** in the sidebar to load the past 7 days of sea level data.")

# ---------------------------------------------------------------------------
# Section 2: Present & Previous 7 Days Sea Level Observations
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title"><span class="section-num">02</span>Present & Previous 7 Days Sea Level Observations</div>', unsafe_allow_html=True)

frame = st.session_state.ssh_frame
window_df = None

if frame is not None and not frame.empty:
    try:
        window_df = latest_input_window(frame, window=INPUT_WINDOW)
    except ValueError as v_err:
        st.warning(f"Unable to extract consecutive 7-day observation sequence: {v_err}")

    if window_df is not None:
        obs_values = window_df["ssh_m"].to_numpy()
        obs_dates = pd.to_datetime(window_df["time"], utc=True)
        present_level = float(obs_values[-1])
        start_level = float(obs_values[0])
        seven_day_delta = present_level - start_level

        # Metric Tiles for Present Sea Level and Past 7-Day Trend
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(
                f"""
                <div class="metric-panel">
                    <div class="metric-panel-label">Present Sea Level (T = 0)</div>
                    <div class="metric-panel-val">{present_level:+.4f} m</div>
                    <div class="metric-panel-sub">Recorded on {obs_dates.iloc[-1].strftime('%Y-%m-%d')} (Daily Mean)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p2:
            st.markdown(
                f"""
                <div class="metric-panel">
                    <div class="metric-panel-label">7-Day Observation Interval</div>
                    <div class="metric-panel-val">{obs_dates.iloc[0].strftime('%Y-%m-%d')} to {obs_dates.iloc[-1].strftime('%Y-%m-%d')}</div>
                    <div class="metric-panel-sub">7 consecutive daily observations</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p3:
            t_type = "Increment (Rising)" if seven_day_delta > 0 else ("Decrement (Falling)" if seven_day_delta < 0 else "Stable")
            st.markdown(
                f"""
                <div class="metric-panel">
                    <div class="metric-panel-label">Past 7-Day Net Trend</div>
                    <div class="metric-panel-val">{seven_day_delta:+.4f} m</div>
                    <div class="metric-panel-sub">Trend: <b>{t_type}</b> over previous 7 days</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Tabular breakdown of the previous 7 days
        table_rows = []
        for i, (dt, val) in enumerate(zip(obs_dates, obs_values)):
            day_label = f"T - {INPUT_WINDOW - 1 - i}" if i < INPUT_WINDOW - 1 else "T = 0 (Present / Origin)"
            daily_step = (val - obs_values[i-1]) if i > 0 else 0.0
            table_rows.append({
                "Sequence": day_label,
                "Observation Date (UTC)": dt.strftime("%Y-%m-%d"),
                "Sea Level Height (m)": f"{val:+.4f}",
                "Day-over-Day Change (m)": f"{daily_step:+.4f}" if i > 0 else "Base Day",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

else:
    st.info("No sea level observations loaded yet. Please choose your position on the map and click 'Load Sea Level Observations for Selected Position'.")

# ---------------------------------------------------------------------------
# Section 3 & 4: Predict 3-Day Upcoming Forecast & Results
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title"><span class="section-num">03</span>Predict Upcoming 3-Day Sea Level Forecast</div>', unsafe_allow_html=True)

if window_df is not None:
    st.write("Click the button below to execute direct multi-step forecasting and calculate sea level predictions for Day 1, Day 2, and Day 3 based on the past 7 days of data.")
    
    # Exact requested button label
    btn_predict = st.button("Predict 3-Day Upcoming Forecast of Sea Level", use_container_width=True, type="primary")

    if btn_predict:
        with st.spinner("Executing direct multi-step forecasting (Persistence Baseline and Random Forest Regressor)..."):
            input_seq = window_df["ssh_m"].to_numpy()
            
            # 1. Baseline Model
            persist_res = forecast(input_seq, model=None, horizon=FORECAST_HORIZON)

            # 2. Random Forest ML Model
            hist_vals = frame["ssh_m"].to_numpy()
            rf_mod, rf_info = train_random_forest(list(hist_vals))
            rf_res = forecast(input_seq, model=rf_mod, horizon=FORECAST_HORIZON) if rf_mod is not None else None

            fc_dates = pd.date_range(
                window_df["time"].iloc[-1] + pd.Timedelta(days=1),
                periods=FORECAST_HORIZON,
                freq="D",
                tz="UTC",
            )

            st.session_state.persist_preds = persist_res.predictions
            st.session_state.rf_preds = rf_res.predictions if rf_res is not None else None
            st.session_state.forecast_dates = fc_dates
            st.session_state.forecast_model_name = rf_res.name if rf_res is not None else "Persistence"
            st.session_state.rf_model = rf_mod
            st.session_state.rf_training = rf_info

            # Save to SQLite ledger
            chosen_res = rf_res if rf_res is not None else persist_res
            pred_series = pd.Series(chosen_res.predictions, index=fc_dates)
            meta_dict = st.session_state.ssh_meta or {}
            fid = database.save_forecast(
                settings.database_path,
                source=meta_dict.get("source", "Copernicus Marine GLORYS"),
                dataset_id=meta_dict.get("dataset_id", active_dataset),
                requested_lat=float(st.session_state.lat),
                requested_lon=float(st.session_state.lon),
                grid_lat=float(loc_obj.grid_latitude) if loc_obj else float(st.session_state.lat),
                grid_lon=float(loc_obj.grid_longitude) if loc_obj else float(st.session_state.lon),
                inputs=window_df,
                predictions=pred_series,
                model_name=chosen_res.name,
                training=rf_info,
            )
            st.session_state.forecast_id = fid
            st.toast("Upcoming 3-day sea level forecast computed successfully.", icon="check")

    # Display 3-Day Forecast Results Table
    if st.session_state.forecast_dates is not None:
        st.markdown("#### Upcoming 3-Day Predicted Data")
        
        fc_dates = st.session_state.forecast_dates
        p_preds = st.session_state.persist_preds
        r_preds = st.session_state.rf_preds
        present_ref = float(window_df["ssh_m"].iloc[-1])

        active_predictions = r_preds if r_preds is not None else p_preds
        net_forecast_delta = float(active_predictions[-1] - present_ref)

        # Result Tiles
        fc_c1, fc_c2, fc_c3, fc_c4 = st.columns(4)
        with fc_c1:
            st.markdown(
                f"""
                <div class="metric-panel">
                    <div class="metric-panel-label">Day 1 Forecast (+1d)</div>
                    <div class="metric-panel-val">{active_predictions[0]:+.4f} m</div>
                    <div class="metric-panel-sub">{fc_dates[0].strftime('%Y-%m-%d')} ({active_predictions[0] - present_ref:+.4f} m vs Present)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with fc_c2:
            st.markdown(
                f"""
                <div class="metric-panel">
                    <div class="metric-panel-label">Day 2 Forecast (+2d)</div>
                    <div class="metric-panel-val">{active_predictions[1]:+.4f} m</div>
                    <div class="metric-panel-sub">{fc_dates[1].strftime('%Y-%m-%d')} ({active_predictions[1] - present_ref:+.4f} m vs Present)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with fc_c3:
            st.markdown(
                f"""
                <div class="metric-panel">
                    <div class="metric-panel-label">Day 3 Forecast (+3d)</div>
                    <div class="metric-panel-val">{active_predictions[2]:+.4f} m</div>
                    <div class="metric-panel-sub">{fc_dates[2].strftime('%Y-%m-%d')} ({active_predictions[2] - present_ref:+.4f} m vs Present)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with fc_c4:
            f_trend_type = "INCREMENT (RISING)" if net_forecast_delta > 0.0005 else ("DECREMENT (FALLING)" if net_forecast_delta < -0.0005 else "STEADY / NEUTRAL")
            st.markdown(
                f"""
                <div class="metric-panel">
                    <div class="metric-panel-label">3-Day Overall Trend</div>
                    <div class="metric-panel-val">{net_forecast_delta:+.4f} m</div>
                    <div class="metric-panel-sub">Status: <b>{f_trend_type}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Tabular view comparing predictions
        pred_table = []
        for i, dt in enumerate(fc_dates):
            pred_table.append({
                "Forecast Lead": f"Day {i+1} (+{i+1}d)",
                "Forecast Date (UTC)": dt.strftime("%Y-%m-%d"),
                "Persistence Baseline (m)": f"{p_preds[i]:+.4f}",
                "Random Forest Regressor (m)": f"{r_preds[i]:+.4f}" if r_preds is not None else "N/A",
                "Selected Predicted Sea Level (m)": f"{active_predictions[i]:+.4f}",
                "Net Shift vs Present (m)": f"{active_predictions[i] - present_ref:+.4f}",
            })
        st.dataframe(pd.DataFrame(pred_table), use_container_width=True, hide_index=True)

        # -------------------------------------------------------------------
        # Section 4: Trajectory Graph with Increment / Decrement Visual Indicators
        # -------------------------------------------------------------------
        st.markdown('<div class="section-title"><span class="section-num">04</span>Sea Level Forecast Trajectory & Trend Analysis Graph</div>', unsafe_allow_html=True)
        st.write("The graph below depicts the past 7 days of recorded observations transitioning into the upcoming 3 days of predicted sea level height. Use the slope and increment/decrement markers to interpret the forecast.")

        fig_trajectory = seven_plus_three_chart(
            window=window_df,
            forecast_dates=fc_dates,
            persist_preds=p_preds,
            rf_preds=r_preds,
            title="Sea Surface Height Trajectory: Past 7 Days Observations to Upcoming 3 Days Forecast Horizon",
        )
        st.plotly_chart(fig_trajectory, use_container_width=True)

        # Plain-language trend interpretation box so user can directly state the trend
        day1_val = active_predictions[0]
        day2_val = active_predictions[1]
        day3_val = active_predictions[2]
        
        step_1_to_2 = day2_val - day1_val
        step_2_to_3 = day3_val - day2_val

        if net_forecast_delta > 0.0005:
            card_class = "trend-card-increment"
            trend_summary = (
                f"<b>Forecast Trajectory Conclusion: INCREMENT (RISING SEA LEVEL)</b><br>"
                f"Based on the graph, the upcoming 3-day forecast demonstrates an <b>increment trend</b>. "
                f"The predicted sea level rises by <b>+{net_forecast_delta:.4f} metres</b> across the 3-day horizon, "
                f"moving from a present sea level of <b>{present_ref:+.4f} m</b> to <b>{day1_val:+.4f} m</b> on Day 1 (+1d), "
                f"<b>{day2_val:+.4f} m</b> on Day 2 (+2d, change: {step_1_to_2:+.4f} m), and reaching <b>{day3_val:+.4f} m</b> on Day 3 (+3d, change: {step_2_to_3:+.4f} m)."
            )
        elif net_forecast_delta < -0.0005:
            card_class = "trend-card-decrement"
            trend_summary = (
                f"<b>Forecast Trajectory Conclusion: DECREMENT (FALLING SEA LEVEL)</b><br>"
                f"Based on the graph, the upcoming 3-day forecast demonstrates a <b>decrement trend</b>. "
                f"The predicted sea level decreases by <b>{net_forecast_delta:.4f} metres</b> across the 3-day horizon, "
                f"moving from a present sea level of <b>{present_ref:+.4f} m</b> to <b>{day1_val:+.4f} m</b> on Day 1 (+1d), "
                f"<b>{day2_val:+.4f} m</b> on Day 2 (+2d, change: {step_1_to_2:+.4f} m), and descending to <b>{day3_val:+.4f} m</b> on Day 3 (+3d, change: {step_2_to_3:+.4f} m)."
            )
        else:
            card_class = "trend-card-neutral"
            trend_summary = (
                f"<b>Forecast Trajectory Conclusion: STEADY / NEUTRAL SEA LEVEL</b><br>"
                f"Based on the graph, the upcoming 3-day forecast indicates a <b>steady state</b> with negligible net fluctuation "
                f"({net_forecast_delta:+.4f} metres) around the present measurement of <b>{present_ref:+.4f} m</b>."
            )

        st.markdown(f'<div class="{card_class}">{trend_summary}</div>', unsafe_allow_html=True)

        # Feature Importance Analysis
        if st.session_state.rf_training and st.session_state.rf_training.get("feature_importances"):
            with st.expander("Machine Learning Model Feature Importance Analysis"):
                st.plotly_chart(
                    feature_importance_chart(st.session_state.rf_training["feature_importances"]),
                    use_container_width=True,
                )
else:
    st.info("Load sea level data above to enable the 3-day upcoming forecast calculation.")

# ---------------------------------------------------------------------------
# Section 5: Forecast Ledger & Records Export
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title"><span class="section-num">05</span>Forecast Ledger & Verification Records</div>', unsafe_allow_html=True)

forecasts_df = database.list_forecasts(settings.database_path)
validations_df = database.list_validations(settings.database_path)

if not forecasts_df.empty:
    tab_ledger, tab_val = st.tabs(["Saved Forecast Records", "Validation Status Table"])
    with tab_ledger:
        st.dataframe(
            forecasts_df[[
                "forecast_id", "generated_at_utc", "dataset_id",
                "grid_lat", "grid_lon", "model_name", "validation_status"
            ]],
            use_container_width=True,
            hide_index=True,
        )
    with tab_val:
        if not validations_df.empty:
            st.dataframe(
                validations_df[[
                    "forecast_id", "lead", "target_time_utc", "prediction_m",
                    "reference_m", "absolute_error_m", "reference_type", "status"
                ]],
                use_container_width=True,
                hide_index=True,
            )

    dl_c1, dl_c2 = st.columns(2)
    with dl_c1:
        st.download_button(
            "Export Forecast Ledger (CSV)",
            forecasts_df.to_csv(index=False),
            "incois_ssh_forecast_ledger.csv",
            "text/csv",
            use_container_width=True,
        )
    with dl_c2:
        st.download_button(
            "Export Validation Records (CSV)",
            validations_df.to_csv(index=False),
            "incois_ssh_validation_records.csv",
            "text/csv",
            use_container_width=True,
        )
else:
    st.write("No saved forecast records in SQLite ledger.")

# ---------------------------------------------------------------------------
# Institutional Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.82rem; padding: 12px 0;">
        INCOIS (Indian National Centre for Ocean Information Services) · AI-Based Sea Surface Height Forecasting<br>
        Copernicus Marine GLORYS Reanalysis & Operational Analysis/Forecast Integration · Direct Multi-Step Forecasting
    </div>
    """,
    unsafe_allow_html=True,
)
