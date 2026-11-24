"""Plotly chart builders and Folium map helpers for the SSH dashboard.

All functions return Plotly Figure objects, Folium Map objects, or HTML strings —
no Streamlit-specific calls are made here, keeping the visualisation layer
independently testable.
"""
from __future__ import annotations
import logging
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

log = logging.getLogger(__name__)

# Colour palette (oceanography theme)
_HIST_COLOUR   = "#1f77b4"   # steel blue
_WINDOW_COLOUR = "#ff7f0e"   # amber — 7-day input window
_PRED_RF       = "#2ca02c"   # green — Random Forest
_PRED_PERSIST  = "#d62728"   # red — persistence baseline
_REF_COLOUR    = "#9467bd"   # purple — GLORYS reference

# Standard Indian Ocean & Bay of Bengal coastal presets
COASTAL_PRESETS = [
    {"name": "Central Bay of Bengal (Open Ocean)", "lat": 12.000, "lon": 88.000, "desc": "INCOIS standard deep-water reference point"},
    {"name": "Visakhapatnam Coast (Offshore)", "lat": 17.650, "lon": 83.450, "desc": "Andhra Pradesh coastal ocean point"},
    {"name": "Chennai Coast (Offshore)", "lat": 13.100, "lon": 80.350, "desc": "Coromandel Coast offshore waters"},
    {"name": "Paradeep Coast (Offshore)", "lat": 20.250, "lon": 86.750, "desc": "Odisha coastal offshore waters"},
    {"name": "Kolkata / Sagar Island (Offshore)", "lat": 21.300, "lon": 88.100, "desc": "Northern Bay of Bengal head waters"},
    {"name": "Machilipatnam Coast (Offshore)", "lat": 16.150, "lon": 81.300, "desc": "Krishna-Godavari shelf offshore"},
    {"name": "Gopalpur / Puri Coast (Offshore)", "lat": 19.250, "lon": 85.050, "desc": "Ganjam shelf offshore waters"},
    {"name": "Port Blair / Andaman Sea", "lat": 11.650, "lon": 92.800, "desc": "Andaman & Nicobar island basin"},
    {"name": "Kanyakumari / Indian Ocean", "lat": 7.900, "lon": 77.600, "desc": "Equatorial Northern Indian Ocean"},
    {"name": "Kochi / Arabian Sea (Offshore)", "lat": 9.950, "lon": 76.100, "desc": "Southwestern Indian coastline"},
]


def ssh_history_chart(
    history: pd.DataFrame,
    window: pd.DataFrame | None = None,
    title: str = "Daily SSH History (Copernicus GLORYS)",
) -> go.Figure:
    """Plot full history with the 7-day input window highlighted."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["time"], y=history["ssh_m"],
        mode="lines", name="GLORYS Historical Observations",
        line=dict(color=_HIST_COLOUR, width=1.5),
        hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>SSH</b>: %{y:.4f} m<extra></extra>",
    ))
    if window is not None and not window.empty:
        fig.add_trace(go.Scatter(
            x=window["time"], y=window["ssh_m"],
            mode="lines+markers", name="7-Day Input Window",
            line=dict(color=_WINDOW_COLOUR, width=2.5),
            marker=dict(size=7, color=_WINDOW_COLOUR, symbol="circle"),
            hovertemplate="<b>Input Window</b>: %{x|%Y-%m-%d}<br><b>SSH</b>: %{y:.4f} m<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#1e3d59")),
        xaxis=dict(title="Date (UTC)", gridcolor="#edf2f7", showgrid=True),
        yaxis=dict(title="Sea Surface Height (m above geoid)", gridcolor="#edf2f7", showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=55, r=25, t=55, b=45),
        hovermode="x unified",
    )
    return fig


def seven_plus_three_chart(
    window: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    persist_preds: np.ndarray,
    rf_preds: np.ndarray | None = None,
    title: str = "7-Day Historical Input Observations & 3-Day Forecast",
) -> go.Figure:
    """Professional Plotly chart focused specifically on 7 observations and 3 predictions.

    Clearly distinguishes historical observations from predicted values,
    and displays persistence and Random Forest predictions separately.
    """
    fig = go.Figure()

    # 1. Seven historical daily observations (solid blue line, distinct circles)
    obs_times = pd.to_datetime(window["time"], utc=True)
    obs_vals = window["ssh_m"].to_numpy()

    fig.add_trace(go.Scatter(
        x=obs_times,
        y=obs_vals,
        mode="lines+markers+text",
        name="Historical Observations (Past 7 Days)",
        line=dict(color=_HIST_COLOUR, width=3),
        marker=dict(size=9, color=_HIST_COLOUR, symbol="circle", line=dict(color="white", width=1.5)),
        text=[f"{v:.3f}m" for v in obs_vals],
        textposition="top center",
        textfont=dict(size=10, color=_HIST_COLOUR),
        hovertemplate="<b>Observed SSH</b>: %{y:.4f} m<br><b>Date</b>: %{x|%Y-%m-%d}<extra></extra>",
    ))

    # Origin point connection
    origin_time = obs_times.iloc[-1]
    origin_val = float(obs_vals[-1])
    fc_times = list(pd.to_datetime(forecast_dates, utc=True))
    all_persist_x = [origin_time] + fc_times
    all_persist_y = [origin_val] + list(persist_preds)

    # 2. Persistence forecast (dashed red line, diamond markers)
    fig.add_trace(go.Scatter(
        x=all_persist_x,
        y=all_persist_y,
        mode="lines+markers",
        name="Persistence Forecast (Baseline)",
        line=dict(color=_PRED_PERSIST, width=2.5, dash="dash"),
        marker=dict(size=10, color=_PRED_PERSIST, symbol="diamond", line=dict(color="white", width=1)),
        hovertemplate="<b>Persistence</b>: %{y:.4f} m<br><b>Target</b>: %{x|%Y-%m-%d}<extra></extra>",
    ))

    # 3. Random Forest forecast (dashed green line, square markers)
    target_preds = rf_preds if rf_preds is not None else persist_preds
    delta_trend = float(target_preds[-1] - origin_val)
    if delta_trend > 0.0005:
        trend_label = f"Upcoming Trend: Increment (+{delta_trend:.4f} m)"
        trend_border = "#16a34a"
        trend_bg = "#f0fdf4"
    elif delta_trend < -0.0005:
        trend_label = f"Upcoming Trend: Decrement ({delta_trend:.4f} m)"
        trend_border = "#dc2626"
        trend_bg = "#fef2f2"
    else:
        trend_label = f"Upcoming Trend: Steady ({delta_trend:+.4f} m)"
        trend_border = "#475569"
        trend_bg = "#f8fafc"

    if rf_preds is not None:
        all_rf_x = [origin_time] + fc_times
        all_rf_y = [origin_val] + list(rf_preds)
        fig.add_trace(go.Scatter(
            x=all_rf_x,
            y=all_rf_y,
            mode="lines+markers+text",
            name="Random Forest Regressor (ML)",
            line=dict(color=_PRED_RF, width=2.5, dash="dot"),
            marker=dict(size=10, color=_PRED_RF, symbol="square", line=dict(color="white", width=1)),
            text=[""] + [f"{v:.3f}m ({v - origin_val:+.3f}m)" for v in rf_preds],
            textposition="bottom right",
            textfont=dict(size=10, color=_PRED_RF),
            hovertemplate="<b>Random Forest</b>: %{y:.4f} m<br><b>Target</b>: %{x|%Y-%m-%d}<extra></extra>",
        ))

    # Vertical dividing line at forecast origin
    fig.add_vline(
        x=origin_time,
        line_width=2,
        line_dash="dash",
        line_color="#718096",
        annotation_text="<b>Forecast Origin (T=0)</b>",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#4a5568"),
    )

    # Shaded forecast horizon region with Trend Annotation
    if len(fc_times) > 0:
        fig.add_vrect(
            x0=origin_time,
            x1=fc_times[-1],
            fillcolor="#ebf8ff",
            opacity=0.45,
            layer="below",
            line_width=0,
            annotation_text=f"<b>{trend_label}</b>",
            annotation_position="top right",
            annotation_font=dict(size=11, color=trend_border),
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1a365d")),
        xaxis=dict(title="Observation & Forecast Timeline (UTC)", gridcolor="#edf2f7", showgrid=True),
        yaxis=dict(title="Sea Surface Height (m)", gridcolor="#edf2f7", showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5),
        template="plotly_white",
        margin=dict(l=60, r=30, t=65, b=50),
        hovermode="x unified",
    )
    return fig


def forecast_chart(
    history: pd.DataFrame,
    window: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    rf_preds: np.ndarray | None,
    persist_preds: np.ndarray,
    validation_df: pd.DataFrame | None = None,
    title: str = "Complete Time-Series Context: Observations, Forecasts & Reference",
) -> go.Figure:
    """Combined history + forecast + (optional) reference validation chart."""
    fig = go.Figure()

    # Historical observations (dimmed)
    fig.add_trace(go.Scatter(
        x=history["time"], y=history["ssh_m"],
        mode="lines", name="GLORYS observations",
        line=dict(color=_HIST_COLOUR, width=1.5, dash="solid"),
        opacity=0.55,
        hovertemplate="<b>History</b>: %{y:.4f} m<extra></extra>",
    ))
    # Input window
    fig.add_trace(go.Scatter(
        x=window["time"], y=window["ssh_m"],
        mode="lines+markers", name="Input window (7 days)",
        line=dict(color=_WINDOW_COLOUR, width=2.5),
        marker=dict(size=7, color=_WINDOW_COLOUR),
        hovertemplate="<b>Input</b>: %{y:.4f} m<extra></extra>",
    ))

    # Persistence forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=persist_preds,
        mode="lines+markers", name="Persistence forecast",
        line=dict(color=_PRED_PERSIST, width=2, dash="dash"),
        marker=dict(symbol="diamond", size=9),
        hovertemplate="<b>Persistence</b>: %{y:.4f} m<extra></extra>",
    ))

    # Random Forest forecast
    if rf_preds is not None:
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=rf_preds,
            mode="lines+markers", name="Random Forest forecast",
            line=dict(color=_PRED_RF, width=2, dash="dot"),
            marker=dict(symbol="circle", size=9),
            hovertemplate="<b>Random Forest</b>: %{y:.4f} m<extra></extra>",
        ))

    # Reference values (if available)
    if validation_df is not None and not validation_df.empty:
        ref_times = pd.to_datetime(validation_df["target_time_utc"], utc=True)
        ref_vals  = pd.to_numeric(validation_df["reference_m"], errors="coerce").dropna()
        if not ref_vals.empty:
            fig.add_trace(go.Scatter(
                x=ref_times, y=ref_vals,
                mode="markers", name="GLORYS reference",
                marker=dict(symbol="x", size=12, color=_REF_COLOUR, line=dict(width=2.5)),
                hovertemplate="<b>GLORYS Reference</b>: %{y:.4f} m<extra></extra>",
            ))

    # Vertical separator at forecast origin
    origin = window["time"].iloc[-1]
    fig.add_vline(
        x=origin, line_dash="dot", line_color="grey",
        annotation_text="Forecast origin",
        annotation_position="top left",
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#1e3d59")),
        xaxis=dict(title="Date (UTC)", gridcolor="#edf2f7"),
        yaxis=dict(title="SSH (m)", gridcolor="#edf2f7"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=55, r=25, t=55, b=45),
    )
    return fig


def metrics_bar_chart(
    summary_df: pd.DataFrame,
    metric: str = "mae_m",
    title: str = "MAE by Forecast Lead (m)",
) -> go.Figure:
    """Bar chart of per-lead and overall metrics."""
    if summary_df.empty or metric not in summary_df:
        fig = go.Figure()
        fig.update_layout(title="No backtest results available.", template="plotly_white")
        return fig

    df = summary_df.copy()
    if "lead_label" not in df:
        df["lead_label"] = df["lead"].map({0: "Overall", 1: "Day 1", 2: "Day 2", 3: "Day 3"})

    fig = go.Figure(go.Bar(
        x=df["lead_label"],
        y=df[metric].round(4),
        text=[f"{v:.4f} m" for v in df[metric]],
        textposition="auto",
        marker_color=[_PRED_RF if l != "Overall" else _HIST_COLOUR for l in df["lead_label"]],
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1e3d59")),
        xaxis_title="Forecast Lead",
        yaxis_title=metric.upper().replace("_M", " (m)"),
        template="plotly_white",
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def validation_scatter(
    validations: pd.DataFrame,
    title: str = "Predicted vs Reference SSH (m)",
) -> go.Figure:
    """Scatter plot of predicted SSH against GLORYS reference SSH."""
    fig = go.Figure()
    if validations.empty or "reference_m" not in validations.columns:
        fig.update_layout(title="No validated results yet.", template="plotly_white")
        return fig

    validated = validations[pd.to_numeric(validations["reference_m"], errors="coerce").notna()]
    if validated.empty:
        fig.update_layout(title="No validated results yet.", template="plotly_white")
        return fig

    for lead, grp in validated.groupby("lead"):
        fig.add_trace(go.Scatter(
            x=grp["reference_m"].astype(float),
            y=grp["prediction_m"].astype(float),
            mode="markers",
            name=f"Day {lead}",
            marker=dict(size=9),
            hovertemplate=f"Day {lead}<br>Ref: %{{x:.4f}} m<br>Pred: %{{y:.4f}} m<extra></extra>",
        ))
    # 1:1 line
    all_vals = pd.to_numeric(validated[["reference_m", "prediction_m"]].values.ravel(), errors="coerce")
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) >= 2:
        lo, hi = float(all_vals.min()), float(all_vals.max())
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines", name="1:1 Perfect Agreement",
            line=dict(dash="dot", color="grey"),
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1e3d59")),
        xaxis_title="GLORYS Reference SSH (m)",
        yaxis_title="Predicted SSH (m)",
        template="plotly_white",
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def build_ocean_map(
    req_lat: float,
    req_lon: float,
    grid_lat: float | None = None,
    grid_lon: float | None = None,
    location_name: str = "Selected Location",
):
    """Build a rich interactive Folium map centered on the target ocean coordinate."""
    try:
        import folium
    except ImportError:
        return None

    # Center on requested or grid coordinate
    center_lat = grid_lat if grid_lat is not None else req_lat
    center_lon = grid_lon if grid_lon is not None else req_lon

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # 1. Bay of Bengal & Indian Ocean Region boundaries (subtle guidance)
    folium.Rectangle(
        bounds=[[4.0, 78.0], [23.5, 98.0]],
        color="#2b6cb0",
        weight=1.5,
        fill=True,
        fill_color="#ebf8ff",
        fill_opacity=0.08,
        dash_array="6",
        tooltip="Bay of Bengal & Northern Indian Ocean Domain",
    ).add_to(m)

    # 2. Add coastal & ocean preset reference stations as clickable circles
    for p in COASTAL_PRESETS:
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=5,
            color="#3182ce",
            fill=True,
            fill_color="#63b3ed",
            fill_opacity=0.8,
            tooltip=f"<b>{p['name']}</b><br>Lat: {p['lat']:.3f}°N, Lon: {p['lon']:.3f}°E<br><i>{p['desc']}</i>",
            popup=f"<b>{p['name']}</b><br>Lat: {p['lat']:.3f}°N, Lon: {p['lon']:.3f}°E<br>{p['desc']}",
        ).add_to(m)

    # 3. Requested Ocean Location marker (Orange/Red Pin)
    folium.Marker(
        [req_lat, req_lon],
        tooltip=f"<b>Requested: {location_name}</b><br>{req_lat:.4f}°N, {req_lon:.4f}°E",
        popup=f"<b>Requested Ocean Location</b><br>Lat: {req_lat:.4f}°N<br>Lon: {req_lon:.4f}°E",
        icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
    ).add_to(m)

    # 4. If Copernicus grid cell has been identified (snapped wet cell)
    if grid_lat is not None and grid_lon is not None:
        folium.Marker(
            [grid_lat, grid_lon],
            tooltip=f"<b>Copernicus GLORYS Grid Cell (1/12°)</b><br>{grid_lat:.4f}°N, {grid_lon:.4f}°E",
            popup=f"<b>Actual Ocean Grid Cell Used</b><br>Lat: {grid_lat:.4f}°N<br>Lon: {grid_lon:.4f}°E",
            icon=folium.Icon(color="blue", icon="tint", prefix="fa"),
        ).add_to(m)

        # If snapped cell differs, draw a line showing offset
        if abs(req_lat - grid_lat) > 0.001 or abs(req_lon - grid_lon) > 0.001:
            folium.PolyLine(
                locations=[[req_lat, req_lon], [grid_lat, grid_lon]],
                color="#dd6b20",
                weight=2.5,
                dash_array="4",
                tooltip=f"Offset: Snapped to nearest valid ocean cell ({grid_lat:.3f}°N, {grid_lon:.3f}°E)",
            ).add_to(m)

    # Enable click to show coordinate popup anywhere on map
    folium.LatLngPopup().add_to(m)
    return m


def feature_importance_chart(
    importances: list[float] | np.ndarray,
    title: str = "Random Forest: Predictive Influence of Past 7 Days",
) -> go.Figure:
    """Bar chart displaying relative feature importance across the 7-day input window."""
    vals = [float(v) for v in importances]
    if not vals:
        fig = go.Figure()
        fig.update_layout(title="No feature importance available.", template="plotly_white")
        return fig

    labels = [
        f"T - {len(vals) - 1 - i}" if i < len(vals) - 1 else "T = 0 (Origin)"
        for i in range(len(vals))
    ]
    pct_vals = [round(v * 100, 2) for v in vals]

    fig = go.Figure(go.Bar(
        x=labels,
        y=pct_vals,
        text=[f"{p:.1f}%" for p in pct_vals],
        textposition="auto",
        marker=dict(
            color=pct_vals,
            colorscale="Greens",
            showscale=False,
        ),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1e3d59")),
        xaxis_title="Input Day in 7-Day Observation Window",
        yaxis_title="Relative Predictive Importance (%)",
        template="plotly_white",
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def location_map_html(grid_lat: float, grid_lon: float, req_lat: float, req_lon: float) -> str:
    """Return an HTML string with a Folium map showing the selected grid cell."""
    m = build_ocean_map(req_lat, req_lon, grid_lat, grid_lon)
    if m is not None:
        return m._repr_html_()
    return (
        f"<p>Folium not available. Grid cell: "
        f"<b>{grid_lat:.3f}°N, {grid_lon:.3f}°E</b> "
        f"(requested {req_lat:.3f}°N, {req_lon:.3f}°E).</p>"
    )
