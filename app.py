import json
import os
from pathlib import Path
import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# Try importing Google GenAI SDK types
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# Internal modular architecture
from src.grid import LOCALITIES, generate_patiala_h3_grid
from src.safety import (
    DEMOGRAPHIC_FACTORS,
    DEMOGRAPHIC_LABELS,
    calculate_indian_aqi,
    calculate_safety_index,
    get_safety_summary,
)
from src.advisor import (
    resolve_gemini_api_key,
    resolve_groq_api_key,
    build_pehredar_system_prompt,
    generate_response_with_resilience,
)
from src.predictor import (
    generate_and_save_per_hex_forecast,
    train_lightgbm_models,
)
try:
    from src.router import plan_routes, geocode_place, get_known_localities, GeocodeFallbackNeeded
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False
    def get_known_localities():
        return list(LOCALITIES.keys())
    class GeocodeFallbackNeeded(Exception):
        pass

# -----------------------------------------------------------------------------
# Page Configuration & Clean Eco Pehredar Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pehredar 🛡️ | Patiala Air Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fbf8 0%, #edf7ed 100%);
        color: #1b3320;
    }
    .pehredar-header {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 60%, #388e3c 100%);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        color: #ffffff;
        box-shadow: 0 4px 18px rgba(27, 94, 32, 0.22);
        margin-bottom: 1.2rem;
    }
    .pehredar-header h1 {
        margin: 0;
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff !important;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .pehredar-header p {
        margin: 0.35rem 0 0 0;
        color: #dcedc8;
        font-size: 1.02rem;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #dcedc8;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
    .metric-title {
        font-size: 0.82rem;
        color: #558b2f;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1b3320;
        margin-top: 0.15rem;
    }
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #dcedc8;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
    }
    .stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #f1f8e9;
        border: 1px solid #c5e1a5;
    }
    .stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #ffffff;
        border-left: 4px solid #2e7d32;
    }
    .stChatMessage [data-testid="stCaptionContainer"] {
        color: #558b2f !important;
        font-size: 0.78rem;
        margin-top: 0.3rem;
    }
    .stButton > button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #1b5e20;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path(__file__).resolve().parent / "data"

# -----------------------------------------------------------------------------
# Sidebar Configuration & DEMO_MODE Toggle
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ Pehredar Settings")
    st.caption("AI Neighbourhood Air Pollution System • Patiala, Punjab")

    # DEMO_MODE toggle (runs fully from cache, no network calls)
    demo_mode = st.toggle("🔒 DEMO_MODE (Run from Cache)", value=True, help="When enabled, runs entirely from offline cached files in data/ without invoking external APIs.")

    if demo_mode:
        st.success("🟢 DEMO_MODE Active: Using offline local cache", icon="📁")
    else:
        st.info("🌐 Live API Mode Active", icon="📡")

    st.divider()

    st.markdown("#### ⚙️ AI Advisor Configuration")
    model_choice = st.selectbox(
        "Gemini Model",
        options=["gemini-3.8-flash", "gemini-3.6-flash"],
        index=0,
        help="Google Gemini Flash models provide fast, intelligent, and lightweight responses.",
    )

    temperature = st.slider(
        "Creativity / Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
    )

    st.divider()

    gemini_key, gemini_source = resolve_gemini_api_key()
    groq_key, groq_source = resolve_groq_api_key()

    if gemini_key:
        st.success(f"🌱 Gemini API: Active ({gemini_source})", icon="✅")
    else:
        st.warning("⚠️ No Gemini Key found. Add GEMINI_API_KEY to .env or enter below:")
        manual_gem = st.text_input("Enter Gemini API Key:", type="password", key="manual_gemini_key")
        if manual_gem:
            gemini_key = manual_gem
            st.rerun()

    if groq_key:
        st.info(f"⚡ Groq Fallback: Ready ({groq_source})", icon="ℹ️")
    else:
        st.caption("Optional: Set GROQ_API_KEY in .env for gpt-oss-120b fallback.")

    st.divider()

    if st.button("🔄 Reload Forecast Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------------------------------------------------
# Data Loading & Forecast Ingestion
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_application_state(is_demo: bool):
    """
    Loads 72h forecast grid, H3 hexagon mesh, and evaluation metrics.
    """
    metrics_path = DATA_DIR / "metrics.json"
    grid_parquet = DATA_DIR / "forecast_grid.parquet"

    # Ensure models and forecast grid are available
    if not grid_parquet.exists() or not metrics_path.exists():
        train_lightgbm_models(demo_mode=is_demo)
        df_forecast = generate_and_save_per_hex_forecast(demo_mode=is_demo)
    else:
        df_forecast = pd.read_parquet(grid_parquet)

    # Load metrics
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    # Generate H3 base metadata
    h3_cells = generate_patiala_h3_grid()

    return df_forecast, metrics, h3_cells


try:
    df_forecast, metrics, h3_cells = load_application_state(demo_mode)
    data_loaded_ok = True
except Exception as e:
    data_loaded_ok = False
    st.error(f"Failed to load Pehredar model forecasts: {e}")

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="pehredar-header">
        <h1>Pehredar 🛡️</h1>
        <p>AI-Powered Neighbourhood-Level Air Pollution Prediction System for Patiala, Punjab</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not data_loaded_ok:
    st.warning("⚠️ Data could not be loaded. Please ensure data/ has grid_3x3.parquet or disable DEMO_MODE to fetch.", icon="⚠️")
    st.stop()

# -----------------------------------------------------------------------------
# KPI Overview Metrics (Horizon t+1)
# -----------------------------------------------------------------------------
h1_df = df_forecast[df_forecast["horizon_hours"] == 1]
avg_pm25 = round(float(h1_df["pm2_5"].mean()), 1)
avg_aqi, aqi_cat, aqi_color = calculate_indian_aqi(avg_pm25)
avg_safety_healthy = round(float(h1_df["safety_healthy"].mean()), 1)
avg_safety_asthma = round(float(h1_df["safety_respiratory"].mean()), 1)

wind_dir = float(h1_df["wind_direction_10m"].iloc[0])
wind_spd = float(h1_df["wind_speed_10m"].iloc[0])
is_farm_belt_wind = 200.0 <= wind_dir <= 320.0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Patiala City Avg AQI (t+1h)</div>
            <div class="metric-value" style="color: {aqi_color};">AQI {avg_aqi}</div>
            <div style="font-size: 0.85rem; color: #555;">PM2.5: {avg_pm25} µg/m³ • <b>{aqi_cat}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Safety Index (General Public)</div>
            <div class="metric-value" style="color: {'#2e7d32' if avg_safety_healthy >= 7 else '#f57c00' if avg_safety_healthy >= 4 else '#d32f2f'};">
                {avg_safety_healthy} / 10
            </div>
            <div style="font-size: 0.85rem; color: #555;">Asthma / Vulnerable: <b>{avg_safety_asthma} / 10</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    stubble_badge = "🔴 Active West Smoke Corridor" if is_farm_belt_wind else "🟢 Favorable Wind Vector"
    stubble_color = "#d32f2f" if is_farm_belt_wind else "#2e7d32"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Malwa Farm Belt Trajectory</div>
            <div class="metric-value" style="color: {stubble_color}; font-size: 1.45rem;">{stubble_badge}</div>
            <div style="font-size: 0.85rem; color: #555;">Wind: {int(wind_dir)}° ({wind_spd} km/h)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    model_mae = metrics.get("model_mae", 15.0)
    base_mae = metrics.get("baseline_mae", 18.0)
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">LightGBM Model Accuracy</div>
            <div class="metric-value" style="color: #1b5e20;">MAE {model_mae} µg/m³</div>
            <div style="font-size: 0.85rem; color: #555;">Persistence Baseline: {base_mae} µg/m³</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Feature Tabs
# -----------------------------------------------------------------------------
tab_map, tab_trend, tab_safety, tab_advisor, tab_airguard, tab_eval = st.tabs([
    "🗺️ Neighbourhood H3 Grid Map",
    "📈 72h Forecast & Uncertainty Band",
    "🛡️ Safety Index Calculator",
    "💬 Pehredar AI Advisor",
    "🚦 AI Air Guard",
    "📊 Model Evaluation & SHAP",
])

# -----------------------------------------------------------------------------
# TAB 1: Neighbourhood H3 Grid Map (Folium)
# -----------------------------------------------------------------------------
with tab_map:
    st.subheader("Patiala Resolution-8 H3 Hexagonal Air Quality Mesh (~0.7 km²)")
    st.markdown(
        "Each hexagon represents a discrete **~0.7 km² micro-zone** over Patiala. "
        "Move the **Forecast Horizon Slider** to project air quality up to 72 hours in advance."
    )

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.2, 1.2])

    with ctrl_col1:
        selected_horizon = st.slider(
            "Forecast Horizon (Hours Ahead):",
            min_value=1,
            max_value=72,
            value=1,
            step=1,
            help="Select forecast horizon from t+1h to t+72h.",
        )

    with ctrl_col2:
        demographic_view = st.selectbox(
            "Safety Index Demographic:",
            options=["Healthy Adult", "Children / Elderly / Pregnant", "Asthma / Respiratory"],
            index=0,
        )

    with ctrl_col3:
        target_locality_focus = st.selectbox(
            "Locality Filter:",
            options=["All Localities"] + list(LOCALITIES.keys()),
            index=0,
        )

    # Filter forecast data for selected horizon
    horizon_data = df_forecast[df_forecast["horizon_hours"] == selected_horizon].copy()

    # Determine safety col
    if "Asthma" in demographic_view:
        safety_col = "safety_respiratory"
    elif "Children" in demographic_view:
        safety_col = "safety_vulnerable"
    else:
        safety_col = "safety_healthy"

    # Merge boundary coordinates from h3_cells
    h3_lookup = {c["hex_id"]: c for c in h3_cells}

    map_col, side_col = st.columns([2.0, 1.0])

    with map_col:
        # Build Folium map centered on Patiala (30.34, 76.39)
        m = folium.Map(
            location=[30.342, 76.395],
            zoom_start=12,
            tiles="OpenStreetMap",
            control_scale=True,
        )

        for _, row in horizon_data.iterrows():
            h_id = row["hex_id"]
            if h_id not in h3_lookup:
                continue

            cell_info = h3_lookup[h_id]
            coords = cell_info["boundary"]
            color = row["aqi_color"]
            loc_name = row["locality"]
            pm_val = row["pm2_5"]
            aqi_val = row["aqi"]
            cat_val = row["aqi_category"]
            safe_val = row[safety_col]

            # Highlight selected locality if filtered
            if target_locality_focus != "All Localities" and loc_name != target_locality_focus:
                fill_opacity = 0.15
                weight = 0.5
            else:
                fill_opacity = 0.65
                weight = 1.6

            tooltip_html = f"""
            <b>{loc_name}</b><br>
            Hex: <code>{h_id[-6:]}</code><br>
            PM2.5: <b>{pm_val} µg/m³</b><br>
            NAQI: <b>{aqi_val} ({cat_val})</b><br>
            Safety: <b>{safe_val}/10</b>
            """

            folium.Polygon(
                locations=coords,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=fill_opacity,
                weight=weight,
                tooltip=tooltip_html,
            ).add_to(m)

        # Marker for landmark localities
        for loc_name, (l_lat, l_lon) in LOCALITIES.items():
            if target_locality_focus == "All Localities" or loc_name == target_locality_focus:
                folium.CircleMarker(
                    location=[l_lat, l_lon],
                    radius=4,
                    color="#1b5e20",
                    fill=True,
                    fill_color="#ffffff",
                    fill_opacity=1.0,
                    tooltip=f"📍 {loc_name}",
                ).add_to(m)

        st_folium(m, width="100%", height=560, returned_objects=[])

    with side_col:
        st.markdown(f"#### 📍 Locality Rankings (t+{selected_horizon}h)")
        
        loc_summary = horizon_data.groupby("locality").agg({
            "pm2_5": "mean",
            "aqi": "mean",
            safety_col: "mean",
            "aqi_category": lambda s: s.mode()[0] if not s.empty else "N/A",
            "aqi_color": lambda s: s.mode()[0] if not s.empty else "#333",
        }).reset_index()

        loc_summary["pm2_5"] = loc_summary["pm2_5"].round(1)
        loc_summary["aqi"] = loc_summary["aqi"].round(0).astype(int)
        loc_summary[safety_col] = loc_summary[safety_col].round(1)
        loc_summary = loc_summary.sort_values("aqi")

        for _, r in loc_summary.iterrows():
            st.markdown(
                f"""
                <div style="background: #ffffff; border-radius: 8px; border-left: 4px solid {r['aqi_color']}; padding: 0.5rem 0.8rem; margin-bottom: 0.4rem; font-size: 0.88rem;">
                    <b>{r['locality']}</b>: AQI {r['aqi']} ({r['aqi_category']})<br>
                    <span style="color: #666;">PM2.5: {r['pm2_5']} µg/m³ • Safety: <b>{r[safety_col]}/10</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # Route Planning Panel
    # -------------------------------------------------------------------------
    st.divider()
    with st.expander("🗺️ Plan a Safe Route", expanded=False):
        st.markdown(
            "Find the **safest route** between two places in Patiala based on predicted air quality along the path."
        )

        if not ROUTER_AVAILABLE:
            st.error("⚠️ Router module not available. Check src/router.py is installed correctly.")
        else:
            known_places = get_known_localities()
            options_with_other = known_places + ["Other…"]

            r_col1, r_col2 = st.columns(2)
            with r_col1:
                orig_default_idx = known_places.index("Thapar University") if "Thapar University" in known_places else 0
                route_from_select = st.selectbox(
                    "📍 From (origin):",
                    options=options_with_other,
                    index=orig_default_idx,
                    key="route_from_select",
                    help="Select a recognized Patiala locality or choose 'Other…' to type a custom place.",
                )
                if route_from_select == "Other…":
                    route_from = st.text_input(
                        "Enter custom origin name:",
                        value="",
                        placeholder="e.g. Near Thapar Gate 2",
                        key="route_from_custom",
                    )
                else:
                    route_from = route_from_select

                route_depart = st.selectbox(
                    "🕐 Departure time:",
                    options=["Now", "+1h", "+3h", "+6h", "+12h", "+24h"],
                    index=0,
                    key="route_depart",
                )
                route_profile = st.selectbox(
                    "👤 Health profile:",
                    options=["Healthy Adult", "Children / Elderly / Pregnant", "Asthma / Respiratory"],
                    index=0,
                    key="route_profile",
                )

            with r_col2:
                dest_default_idx = known_places.index("Baradari") if "Baradari" in known_places else min(1, len(known_places)-1)
                route_to_select = st.selectbox(
                    "🏁 To (destination):",
                    options=options_with_other,
                    index=dest_default_idx,
                    key="route_to_select",
                    help="Select a recognized Patiala locality or choose 'Other…' to type a custom place.",
                )
                if route_to_select == "Other…":
                    route_to = st.text_input(
                        "Enter custom destination name:",
                        value="",
                        placeholder="e.g. Kali Devi Temple, Patiala",
                        key="route_to_custom",
                    )
                else:
                    route_to = route_to_select

                route_mode = st.selectbox(
                    "🚶 Travel mode:",
                    options=["Walk 🚶", "Bike 🚴", "Car 🚗"],
                    index=0,
                    key="route_mode",
                )

            find_route_btn = st.button("🔍 Find Safest Route", key="find_route_btn", use_container_width=True)

            if find_route_btn:
                if not route_from.strip():
                    st.warning("Please specify a departure origin.")
                elif not route_to.strip():
                    st.warning("Please specify a destination.")
                else:
                    # Map UI labels to internal values
                    depart_offset_map = {"Now": 0, "+1h": 1, "+3h": 3, "+6h": 6, "+12h": 12, "+24h": 24}
                    mode_map = {"Walk 🚶": "walk", "Bike 🚴": "bike", "Car 🚗": "car"}
                    profile_map = {
                        "Healthy Adult": "healthy",
                        "Children / Elderly / Pregnant": "vulnerable",
                        "Asthma / Respiratory": "respiratory",
                    }

                    depart_offset = depart_offset_map.get(route_depart, 0)
                    travel_mode = mode_map.get(route_mode, "walk")
                    health_prof = profile_map.get(route_profile, "healthy")

                    with st.spinner("🗺️ Finding safest route and scoring air quality..."):
                        try:
                            route_results = plan_routes(
                                from_place=route_from,
                                to_place=route_to,
                                departure_offset_hours=depart_offset,
                                mode=travel_mode,
                                health_profile=health_prof,
                                df_forecast=df_forecast,
                            )

                            if not route_results:
                                st.warning("No routes found between the specified places.")
                            else:
                                best = route_results[0]
                                origin_ll = best["origin_latlon"]
                                dest_ll = best["dest_latlon"]
                                has_offline_route = any(not rr.get("is_real_route", True) for rr in route_results)

                                # Build a new Folium map with routes overlaid
                                route_map = folium.Map(
                                    location=[
                                        (origin_ll[0] + dest_ll[0]) / 2,
                                        (origin_ll[1] + dest_ll[1]) / 2,
                                    ],
                                    zoom_start=13,
                                    tiles="OpenStreetMap",
                                    control_scale=True,
                                )

                                # Draw each route (safest = thicker green, others = grey)
                                route_colors = ["#2e7d32", "#9e9e9e", "#757575"]
                                route_weights = [6, 3, 3]
                                for i, rr in enumerate(route_results):
                                    coords_ll = [[c[1], c[0]] for c in rr["geometry_coords"]]  # GeoJSON lon,lat → folium lat,lon
                                    color = route_colors[i] if i < len(route_colors) else "#9e9e9e"
                                    weight = route_weights[i] if i < len(route_weights) else 3
                                    label_suffix = " (approximate route - offline)" if not rr.get("is_real_route", True) else ""
                                    label = f"Route {rr['rank']} {'(Safest ✅)' if i == 0 else ''} | Safety {rr['safety_score']}/10{label_suffix}"
                                    folium.PolyLine(
                                        locations=coords_ll,
                                        color=color,
                                        weight=weight,
                                        opacity=0.85,
                                        tooltip=label,
                                    ).add_to(route_map)

                                # Origin & destination markers
                                folium.Marker(
                                    location=list(origin_ll),
                                    tooltip=f"📍 {best['origin_name'][:60]}",
                                    icon=folium.Icon(color="green", icon="play"),
                                ).add_to(route_map)
                                folium.Marker(
                                    location=list(dest_ll),
                                    tooltip=f"🏁 {best['dest_name'][:60]}",
                                    icon=folium.Icon(color="red", icon="flag"),
                                ).add_to(route_map)

                                st_folium(route_map, width="100%", height=420, returned_objects=[], key="route_map_render")

                                if has_offline_route:
                                    st.info("ℹ️ **Route note:** Running in offline mode using approximate corridor sampled every ~100m. Scored with live Safety Index.")

                                # Results table
                                st.markdown("#### 📊 Route Comparison")
                                table_rows = []
                                for rr in route_results:
                                    type_label = "approximate route (offline)" if not rr.get("is_real_route", True) else "OSRM turn-by-turn"
                                    table_rows.append({
                                        "Route": f"Route {rr['rank']}{'  ✅ Safest' if rr['rank'] == 1 else ''}",
                                        "Type": type_label,
                                        "Time (min)": rr["duration_min"],
                                        "Distance (km)": rr["distance_km"],
                                        "Safety Score": f"{rr['safety_score']}/10",
                                        "Worst Air Quality Point": f"{rr['worst_locality']} (AQI {rr['worst_aqi']})",
                                    })
                                st.dataframe(table_rows, use_container_width=True, hide_index=True)

                                # Recommendation sentence
                                best_aqi_cat_str = ""
                                if df_forecast is not None:
                                    from src.safety import calculate_indian_aqi
                                    _, best_aqi_cat_str, _ = calculate_indian_aqi(best.get("worst_pm25", 75.0))

                                st.success(
                                    f"✅ **Route {best['rank']} is safest** with Safety Score **{best['safety_score']}/10**. "
                                    f"Worst air quality at **{best['worst_locality']}** "
                                    f"(AQI {best['worst_aqi']}, {best_aqi_cat_str}). "
                                    f"Recommended for **{route_profile}** — {route_mode}."
                                )

                        except (GeocodeFallbackNeeded, ValueError) as ve:
                            st.warning(f"📍 Location not found. Please select from known Patiala localities below:")
                            f_c1, f_c2 = st.columns(2)
                            with f_c1:
                                st.selectbox("Pick recognized origin:", options=known_places, key="pick_known_orig")
                            with f_c2:
                                st.selectbox("Pick recognized destination:", options=known_places, key="pick_known_dest")
                            st.info("Select localities from the dropdown above and re-click 'Find Safest Route'.")
                        except RuntimeError as re:
                            st.error(f"🛣️ Routing note: {re}", icon="⚠️")
                        except Exception as ex:
                            st.error(f"Unexpected routing error: {ex}", icon="❌")

# -----------------------------------------------------------------------------
# TAB 2: 72-Hour Prediction & Uncertainty Band (Plotly)
# -----------------------------------------------------------------------------
with tab_trend:
    st.subheader("72-Hour Multi-Horizon Forecast with Uncertainty Bands")
    st.markdown(
        "Direct multi-horizon projection from LightGBM. The shaded band shows the **10th to 90th percentile prediction interval** "
        "(80% confidence interval) capturing meteorological uncertainty."
    )

    plot_loc = st.selectbox(
        "Select Locality to View Detailed Forecast:",
        options=list(LOCALITIES.keys()),
        index=list(LOCALITIES.keys()).index("Baradari") if "Baradari" in LOCALITIES else 0,
    )

    # Filter for chosen locality
    loc_fc = df_forecast[df_forecast["locality"] == plot_loc].copy()
    loc_fc_agg = loc_fc.groupby("horizon_hours").agg({
        "pm2_5": "mean",
        "pm2_5_q10": "mean",
        "pm2_5_q90": "mean",
        "aqi": "mean",
        "target_time": "first",
    }).reset_index()

    # Plotly Figure
    fig = go.Figure()

    # Shaded Quantile Interval (q10 to q90)
    fig.add_trace(go.Scatter(
        x=loc_fc_agg["horizon_hours"],
        y=loc_fc_agg["pm2_5_q90"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=loc_fc_agg["horizon_hours"],
        y=loc_fc_agg["pm2_5_q10"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(46, 125, 50, 0.18)",
        line=dict(width=0),
        name="80% Prediction Interval (q10 - q90)",
    ))

    # Mean Expected PM2.5
    fig.add_trace(go.Scatter(
        x=loc_fc_agg["horizon_hours"],
        y=loc_fc_agg["pm2_5"],
        mode="lines+markers",
        line=dict(color="#1b5e20", width=3),
        marker=dict(size=5, color="#1b5e20"),
        name="Expected PM2.5 (Mean Forecast)",
        hovertemplate="Horizon: +%{x}h<br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
    ))

    # Reference lines for Indian AQI category thresholds (Moderate: 60, Poor: 90, Very Poor: 120, Severe: 250)
    fig.add_hline(y=60, line_dash="dot", line_color="#fbc02d", annotation_text="Satisfactory Threshold (60 µg/m³)")
    fig.add_hline(y=90, line_dash="dash", line_color="#f57c00", annotation_text="Moderate Threshold (90 µg/m³)")
    fig.add_hline(y=120, line_dash="dash", line_color="#d32f2f", annotation_text="Poor Threshold (120 µg/m³)")

    fig.update_layout(
        title=f"72-Hour PM2.5 Projection for {plot_loc}, Patiala",
        xaxis_title="Forecast Horizon (Hours from now)",
        yaxis_title="PM2.5 Concentration (µg/m³)",
        hovermode="x unified",
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: Demographic Safety Index Calculator
# -----------------------------------------------------------------------------
with tab_safety:
    st.subheader("Demographic Environmental Safety Index (1 to 10 Scale)")
    st.markdown(
        """
        The **Pehredar Safety Index** provides an intuitive 1 to 10 score (where **10 = Pristine / Safest**, and **1 = Extreme Hazard**) 
        calibrated against vulnerability factors:
        $$\\text{Safety Index} = \\text{clip}\\left(10 - \\frac{\\text{AQI} \\times \\text{factor}}{50},\\; 1,\\; 10\\right)$$
        """
    )

    s_col1, s_col2 = st.columns([1.2, 1.0])

    with s_col1:
        calc_aqi = st.slider("Select or Test an AQI Level:", min_value=10, max_value=500, value=int(avg_aqi), step=5)
        
        s_healthy = calculate_safety_index(calc_aqi, "healthy")
        s_vuln = calculate_safety_index(calc_aqi, "vulnerable")
        s_resp = calculate_safety_index(calc_aqi, "respiratory")

        st.markdown(
            f"""
            <div style="background: #ffffff; border-radius: 12px; padding: 1.4rem; border: 1px solid #dcedc8; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                <h4 style="margin: 0; color: #1b5e20;">Safety Ratings for AQI {calc_aqi}</h4>
                <div style="margin: 1rem 0; font-size: 1.05rem; line-height: 2.0;">
                    • <b>Healthy Adult (factor 1.0)</b>: <span style="font-size: 1.4rem; font-weight: 700; color: {'#2e7d32' if s_healthy >= 7 else '#f57c00' if s_healthy >= 4 else '#d32f2f'};">{s_healthy} / 10</span><br>
                    • <b>Children / Elderly / Pregnant (factor 1.25)</b>: <span style="font-size: 1.4rem; font-weight: 700; color: {'#2e7d32' if s_vuln >= 7 else '#f57c00' if s_vuln >= 4 else '#d32f2f'};">{s_vuln} / 10</span><br>
                    • <b>Asthma / Lung Condition (factor 1.5)</b>: <span style="font-size: 1.4rem; font-weight: 700; color: {'#2e7d32' if s_resp >= 7 else '#f57c00' if s_resp >= 4 else '#d32f2f'};">{s_resp} / 10</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s_col2:
        safety_summ = get_safety_summary(calc_aqi)
        st.markdown("#### Clinical Action Guidelines")
        st.info(f"💡 **Advisory**: {safety_summ['advisory']}")
        st.markdown(
            """
            - **N95 Respirators**: Essential when Safety Index drops below **4.0**.
            - **School Morning Outdoor Activity**: Shift indoors when children's Safety Index is below **5.0**.
            - **Air Purifier Operation**: Recommended when indoor baseline index drops below **6.0**.
            """
        )

# -----------------------------------------------------------------------------
# TAB 4: Pehredar AI Advisor (Gemini → Groq Fallback)
# -----------------------------------------------------------------------------
with tab_advisor:
    st.subheader("💬 Pehredar AI Environmental Health Advisor")
    st.markdown(
        "Ask questions about Patiala's neighborhood air quality, outdoor safety, jogging windows, "
        "and school protocols. Powered by Google Gemini Flash with automatic resilient Groq fallback."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Sat Sri Akal! I am **Pehredar 🛡️**, your neighbourhood environmental health advisor for Patiala. "
                    "I track hyper-local H3 air quality predictions across Tripuri, Leela Bhawan, Baradari, "
                    "Thapar University, and Model Town. How can I protect your health today?"
                ),
                "model": "Pehredar 🛡️",
            }
        ]

    for msg in st.session_state.messages:
        avatar = "🛡️" if msg["role"] == "assistant" else "🧑‍💻"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("model"):
                st.caption(f"Answered by {msg['model']}")

    user_query = st.chat_input("Ask Pehredar about Patiala air quality or health precautions...")

    if user_query:
        if not gemini_key and not groq_key:
            st.error("🔑 API Key missing! Please enter your Gemini API key in the left sidebar or .env file.", icon="🚨")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_query)

        # Ground system prompt with live Pehredar forecast snapshot
        loc_recs = loc_summary.to_dict(orient="records") if "loc_summary" in locals() else []
        wx_summary = {
            "wind_speed_10m": wind_spd,
            "wind_direction_10m": wind_dir,
            "is_stubble_wind": is_farm_belt_wind,
        }
        system_instruction = build_pehredar_system_prompt(
            locality_summary=loc_recs,
            weather_summary=wx_summary,
            metrics_info=metrics,
        )

        formatted_gemini_contents = []
        if types is not None:
            for msg in st.session_state.messages:
                role = "user" if msg["role"] == "user" else "model"
                if not formatted_gemini_contents and role == "model":
                    continue
                if formatted_gemini_contents and formatted_gemini_contents[-1].role == role:
                    formatted_gemini_contents[-1].parts.append(
                        types.Part.from_text(text=msg["content"])
                    )
                else:
                    formatted_gemini_contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part.from_text(text=msg["content"])],
                        )
                    )

        openai_messages = []
        if system_instruction:
            openai_messages.append({"role": "system", "content": system_instruction.strip()})
        for msg in st.session_state.messages:
            openai_messages.append({"role": msg["role"], "content": msg["content"]})

        with st.chat_message("assistant", avatar="🛡️"):
            stream_gen, answering_model = generate_response_with_resilience(
                selected_gemini_model=model_choice,
                gemini_key=gemini_key,
                groq_api_key=groq_key,
                formatted_gemini_contents=formatted_gemini_contents,
                openai_messages=openai_messages,
                temp=temperature,
                sys_instruction=system_instruction,
            )

            if stream_gen is None or answering_model is None:
                st.error("Pehredar is busy, please try again in a moment.")
            else:
                full_reply = st.write_stream(stream_gen)
                st.caption(f"Answered by {answering_model}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_reply,
                    "model": answering_model,
                })

# -----------------------------------------------------------------------------
# TAB 5: AI Air Guard — Conversational Safe-Travel Assistant
# -----------------------------------------------------------------------------
with tab_airguard:
    st.subheader("🚦 AI Air Guard — Personalized Safe-Travel Advisor")
    st.markdown(
        "AI Air Guard guides you step-by-step to find the **safest time and route** for your journey, "
        "using real Pehredar air quality predictions. Powered by Gemini with Groq fallback."
    )

    # ---- Session state init ----
    if "ag_step" not in st.session_state:
        st.session_state.ag_step = 0
    if "ag_data" not in st.session_state:
        st.session_state.ag_data = {}
    if "ag_result_shown" not in st.session_state:
        st.session_state.ag_result_shown = False

    # Helper: reset state machine
    def _ag_reset():
        st.session_state.ag_step = 0
        st.session_state.ag_data = {}
        st.session_state.ag_result_shown = False

    # Progress tracker
    step_labels = ["📍 Destination", "🕐 Time", "🩺 Health", "🚶 Mode", "✅ Result"]
    progress_pct = min(1.0, st.session_state.ag_step / 4)
    st.progress(progress_pct, text=f"Step {min(st.session_state.ag_step + 1, 5)}/5: {step_labels[min(st.session_state.ag_step, 4)]}")

    st.markdown("---")

    # ---- Step 0: Destination & Origin ----
    if st.session_state.ag_step == 0:
        st.markdown("### 📍 Where are you going?")
        known_places_ag = get_known_localities()
        options_ag = known_places_ag + ["Other…"]

        ag_col1, ag_col2 = st.columns(2)
        with ag_col1:
            orig_idx_ag = known_places_ag.index("Thapar University") if "Thapar University" in known_places_ag else 0
            ag_orig_choice = st.selectbox(
                "From (origin):",
                options=options_ag,
                index=orig_idx_ag,
                key="ag_orig_select",
                help="Choose a recognized Patiala locality or select 'Other…' to type.",
            )
            if ag_orig_choice == "Other…":
                ag_origin = st.text_input(
                    "Type custom origin:",
                    value=st.session_state.ag_data.get("origin_custom", ""),
                    placeholder="e.g. Near Thapar Gate 2",
                    key="ag_origin_custom_input",
                )
            else:
                ag_origin = ag_orig_choice

        with ag_col2:
            dest_idx_ag = known_places_ag.index("Baradari") if "Baradari" in known_places_ag else min(1, len(known_places_ag)-1)
            ag_dest_choice = st.selectbox(
                "To (destination):",
                options=options_ag,
                index=dest_idx_ag,
                key="ag_dest_select",
                help="Choose a recognized Patiala locality or select 'Other…' to type.",
            )
            if ag_dest_choice == "Other…":
                ag_dest = st.text_input(
                    "Type custom destination:",
                    value=st.session_state.ag_data.get("dest_custom", ""),
                    placeholder="e.g. Kali Devi Temple, Patiala",
                    key="ag_dest_custom_input",
                )
            else:
                ag_dest = ag_dest_choice

        if st.button("Next →", key="ag_btn_0"):
            if not ag_dest.strip():
                st.warning("Please enter or select a destination.")
            else:
                st.session_state.ag_data["origin"] = ag_origin.strip() or "Thapar University"
                st.session_state.ag_data["destination"] = ag_dest.strip()
                st.session_state.ag_step = 1
                st.rerun()

    # ---- Step 1: Departure Time ----
    elif st.session_state.ag_step == 1:
        st.markdown(f"### 🕐 When do you plan to leave?")
        st.caption(f"Going to: **{st.session_state.ag_data.get('destination', '')}**")
        ag_time = st.selectbox(
            "Departure time:",
            options=["Now", "+1h", "+3h", "+6h", "+12h", "+24h"],
            index=st.session_state.ag_data.get("depart_idx", 0),
            key="ag_time_input",
        )
        time_offset_map = {"Now": 0, "+1h": 1, "+3h": 3, "+6h": 6, "+12h": 12, "+24h": 24}
        ag_col_back, ag_col_next = st.columns(2)
        with ag_col_back:
            if st.button("← Back", key="ag_back_1"):
                st.session_state.ag_step = 0
                st.rerun()
        with ag_col_next:
            if st.button("Next →", key="ag_btn_1"):
                st.session_state.ag_data["departure_label"] = ag_time
                st.session_state.ag_data["departure_offset_hours"] = time_offset_map[ag_time]
                st.session_state.ag_step = 2
                st.rerun()

    # ---- Step 2: Health Profile ----
    elif st.session_state.ag_step == 2:
        st.markdown("### 🩺 Do you have any health conditions?")
        st.caption("This helps us compute a personalised Safety Index for your journey.")
        ag_health = st.selectbox(
            "Health condition:",
            options=["None (Healthy Adult)", "Children / Elderly / Pregnant", "Asthma / Heart / Lung Condition"],
            index=0,
            key="ag_health_input",
        )
        health_profile_map = {
            "None (Healthy Adult)": "healthy",
            "Children / Elderly / Pregnant": "vulnerable",
            "Asthma / Heart / Lung Condition": "respiratory",
        }
        ag_col_back, ag_col_next = st.columns(2)
        with ag_col_back:
            if st.button("← Back", key="ag_back_2"):
                st.session_state.ag_step = 1
                st.rerun()
        with ag_col_next:
            if st.button("Next →", key="ag_btn_2"):
                st.session_state.ag_data["health_label"] = ag_health
                st.session_state.ag_data["health_profile"] = health_profile_map[ag_health]
                st.session_state.ag_step = 3
                st.rerun()

    # ---- Step 3: Travel Mode ----
    elif st.session_state.ag_step == 3:
        st.markdown("### 🚶 How will you travel?")
        ag_mode = st.selectbox(
            "Travel mode:",
            options=["Walk 🚶", "Bike 🚴", "Car 🚗"],
            index=0,
            key="ag_mode_input",
        )
        mode_map = {"Walk 🚶": "walk", "Bike 🚴": "bike", "Car 🚗": "car"}
        ag_col_back, ag_col_next = st.columns(2)
        with ag_col_back:
            if st.button("← Back", key="ag_back_3"):
                st.session_state.ag_step = 2
                st.rerun()
        with ag_col_next:
            if st.button("🔍 Get My Air Guard Report →", key="ag_btn_3"):
                st.session_state.ag_data["mode_label"] = ag_mode
                st.session_state.ag_data["mode"] = mode_map[ag_mode]
                st.session_state.ag_step = 4
                st.rerun()

    # ---- Step 4: Result ----
    elif st.session_state.ag_step == 4:
        ag = st.session_state.ag_data
        st.markdown(
            f"### ✅ Air Guard Report\n"
            f"**{ag.get('origin', 'Patiala')}** → **{ag.get('destination', '')}** | "
            f"{ag.get('departure_label', 'Now')} | {ag.get('mode_label', 'Walk')} | {ag.get('health_label', 'Healthy')}"
        )

        route_results = None
        dest_forecast = None
        error_msg = None

        if not ROUTER_AVAILABLE:
            error_msg = "Router module unavailable."
        else:
            with st.spinner("🗺️ Fetching routes and air quality data..."):
                try:
                    route_results = plan_routes(
                        from_place=ag.get("origin", "Patiala Bus Stand, Patiala"),
                        to_place=ag.get("destination", "Baradari, Patiala"),
                        departure_offset_hours=ag.get("departure_offset_hours", 0),
                        mode=ag.get("mode", "walk"),
                        health_profile=ag.get("health_profile", "healthy"),
                        df_forecast=df_forecast,
                    )
                except (ValueError, RuntimeError) as e:
                    error_msg = str(e)
                except Exception as e:
                    error_msg = f"Unexpected error: {e}"

        if error_msg:
            st.warning(f"📍 {error_msg}")
            st.markdown("Please select a recognized destination locality below to proceed:")
            fallback_dest_pick = st.selectbox(
                "Select recognized destination locality:",
                options=get_known_localities(),
                key="ag_fallback_dest_pick",
            )
            if st.button("🔄 Use this destination & Generate Report", key="ag_use_fallback_dest_btn"):
                st.session_state.ag_data["destination"] = fallback_dest_pick
                st.rerun()
        else:
            best_route = route_results[0] if route_results else None

            # Get destination hex forecast for departure + trip duration
            depart_time = pd.Timestamp.now() + pd.Timedelta(hours=ag.get("departure_offset_hours", 0))
            trip_duration_h = (best_route["duration_min"] / 60) if best_route else 0.5
            arrival_time = depart_time + pd.Timedelta(hours=trip_duration_h)

            # Try to find destination forecast
            try:
                from src.router import get_hex_forecast_for_time as _ghft
                d_lat, d_lon, _ = geocode_place(ag.get("destination", "Baradari, Patiala"))
                dest_forecast = _ghft(d_lat, d_lon, arrival_time, df_forecast)
            except Exception:
                dest_forecast = None

            # Find safest time in next 24h at destination
            safest_time_label = "N/A"
            safest_si = 0.0
            if dest_forecast:
                try:
                    # Check each hour offset 0..24
                    best_si = -1.0
                    best_hour = 0
                    for h_off in range(0, 25):
                        t_check = pd.Timestamp.now() + pd.Timedelta(hours=h_off)
                        fc_h = _ghft(d_lat, d_lon, t_check, df_forecast)
                        if ag.get("health_profile") == "healthy":
                            si_h = fc_h["safety_healthy"]
                        elif ag.get("health_profile") == "vulnerable":
                            si_h = fc_h["safety_vulnerable"]
                        else:
                            si_h = fc_h["safety_respiratory"]
                        if si_h > best_si:
                            best_si = si_h
                            best_hour = h_off
                    safest_si = round(best_si, 1)
                    if best_hour == 0:
                        safest_time_label = "Right now"
                    else:
                        safest_time_label = f"In ~{best_hour}h"
                except Exception:
                    pass

            # ---- Data Card ----
            dest_name_short = ag.get("destination", "Destination")
            health_label = ag.get("health_label", "Healthy Adult")
            profile_key = ag.get("health_profile", "healthy")

            if dest_forecast:
                dest_aqi = dest_forecast["aqi"]
                dest_cat = dest_forecast["aqi_category"]
                dest_color = dest_forecast["aqi_color"]
                dest_loc = dest_forecast["locality"]
                if profile_key == "healthy":
                    dest_si = dest_forecast["safety_healthy"]
                elif profile_key == "vulnerable":
                    dest_si = dest_forecast["safety_vulnerable"]
                else:
                    dest_si = dest_forecast["safety_respiratory"]
            else:
                dest_aqi = avg_aqi
                dest_cat = aqi_cat
                dest_color = aqi_color
                dest_si = avg_safety_healthy
                dest_loc = ag.get("destination", "Patiala")

            # Metrics card
            st.markdown("#### 📊 Air Quality at Destination (on arrival)")
            card_c1, card_c2, card_c3, card_c4 = st.columns(4)
            with card_c1:
                st.metric("Destination AQI", f"{dest_aqi}", f"{dest_cat}")
            with card_c2:
                si_color_label = "🟢" if dest_si >= 7 else "🟡" if dest_si >= 4 else "🔴"
                st.metric("Safety Index", f"{dest_si}/10", f"{si_color_label} {health_label[:20]}")
            with card_c3:
                st.metric("Safest Time (24h)", safest_time_label, f"SI {safest_si}/10")
            with card_c4:
                if best_route:
                    route_display = f"Route {best_route['rank']}"
                    sub_label = f"Safety {best_route['safety_score']}/10"
                    if not best_route.get("is_real_route", True):
                        sub_label += " · offline approx"
                    st.metric(
                        "Recommended Route",
                        route_display,
                        sub_label,
                    )
                else:
                    st.metric("Recommended Route", "N/A", "")

            st.markdown("---")

            # ---- Build grounded prompt for LLM ----
            route_summary_lines = []
            if route_results:
                for rr in route_results:
                    route_summary_lines.append(
                        f"  Route {rr['rank']}: {rr['distance_km']} km, {rr['duration_min']} min, "
                        f"Safety Score {rr['safety_score']}/10, "
                        f"worst air at {rr['worst_locality']} (AQI {rr['worst_aqi']})."
                    )

            precaution_map = {
                "healthy": "healthy adult — wear a simple mask if AQI > 150, avoid prolonged outdoor exposure above AQI 200",
                "vulnerable": "child/elderly/pregnant — wear N95 mask outdoors, limit time outside, avoid peak pollution hours",
                "respiratory": "person with asthma/heart/lung condition — carry inhaler, wear N95, minimise outdoor time, keep windows closed",
            }

            grounded_prompt = f"""
The user wants to travel from "{ag.get('origin', 'Patiala')}" to "{ag.get('destination', 'Patiala')}".
Departure: {ag.get('departure_label', 'Now')}, Mode: {ag.get('mode_label', 'Walk')}, Health profile: {health_label}.

REAL DATA FROM PEHREDAR FORECAST (do NOT invent numbers):
- AQI at destination ({dest_loc}) on arrival: {dest_aqi} ({dest_cat}), Safety Index for {health_label}: {dest_si}/10
- Safest time in next 24h to travel to this area: {safest_time_label} (Safety Index {safest_si}/10)
- Current citywide PM2.5: {avg_pm25} µg/m³, AQI: {avg_aqi} ({aqi_cat})
- Farm belt stubble smoke: {"Active — avoid outdoor exposure" if is_farm_belt_wind else "Low — favorable wind"}
- Routes found:
{"".join(route_summary_lines) if route_summary_lines else "  Route data unavailable."}

Health precautions for this profile ({profile_key}): {precaution_map.get(profile_key, "Take general precautions.")}

Please answer in the user's language (detect from their question; default English).
Structure your reply as:
1. Air quality at destination (AQI, category, Safety Index)
2. Safest time window recommendation
3. Recommended route and reason
4. Specific health precautions for this profile (N95, inhaler, limit outdoor time if relevant)
Keep it concise, friendly, and actionable. Use real numbers only.
""".strip()

            # Build LLM message chain
            ag_gemini_contents = []
            ag_openai_messages = []
            ag_system = build_pehredar_system_prompt(
                locality_summary=loc_summary.to_dict(orient="records") if "loc_summary" in locals() else [],
                weather_summary={"wind_speed_10m": wind_spd, "wind_direction_10m": wind_dir, "is_stubble_wind": is_farm_belt_wind},
                metrics_info=metrics,
            )

            if types is not None:
                ag_gemini_contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=grounded_prompt)],
                    )
                )

            ag_openai_messages = [
                {"role": "system", "content": ag_system},
                {"role": "user", "content": grounded_prompt},
            ]

            if not gemini_key and not groq_key:
                st.warning("⚠️ No API key found. Add GEMINI_API_KEY or GROQ_API_KEY to .env to enable AI advice.")
            else:
                st.markdown("#### 💬 AI Air Guard Recommendation")
                with st.chat_message("assistant", avatar="🚦"):
                    stream_gen, answering_model = generate_response_with_resilience(
                        selected_gemini_model=model_choice,
                        gemini_key=gemini_key,
                        groq_api_key=groq_key,
                        formatted_gemini_contents=ag_gemini_contents,
                        openai_messages=ag_openai_messages,
                        temp=temperature,
                        sys_instruction=ag_system,
                    )
                    if stream_gen is None:
                        st.error("AI Air Guard is busy. Please try again in a moment.")
                    else:
                        st.write_stream(stream_gen)
                        st.caption(f"Answered by {answering_model} · Data: Pehredar LightGBM Forecast")

        st.markdown("---")
        if st.button("🔄 Start Over", key="ag_reset_btn", use_container_width=True):
            _ag_reset()
            st.rerun()

# -----------------------------------------------------------------------------
# TAB 6: Model Evaluation & SHAP Feature Importances
# -----------------------------------------------------------------------------
with tab_eval:
    st.subheader("Model Validation & Explainability (Hackathon Technical Benchmark)")
    st.markdown(
        "Model performance evaluated against a **'same as 24h ago' persistence baseline** on an independent October Diwali test window."
    )

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown("#### Evaluation Metrics")
        eval_table = pd.DataFrame([
            {"Metric": "Test Window", "Value": str(metrics.get("test_window", "October Diwali"))},
            {"Metric": "Test Samples", "Value": str(metrics.get("test_samples", "N/A"))},
            {"Metric": "LightGBM Model MAE", "Value": f"{metrics.get('model_mae', 'N/A')} µg/m³"},
            {"Metric": "Baseline 24h Persistence MAE", "Value": f"{metrics.get('baseline_mae', 'N/A')} µg/m³"},
            {"Metric": "LightGBM Model RMSE", "Value": f"{metrics.get('model_rmse', 'N/A')} µg/m³"},
            {"Metric": "Baseline 24h Persistence RMSE", "Value": f"{metrics.get('baseline_rmse', 'N/A')} µg/m³"},
            {"Metric": "80% Prediction Interval Coverage", "Value": f"{metrics.get('prediction_interval_coverage_pct', 'N/A')}%"},
        ])
        st.table(eval_table)

    with m_col2:
        st.markdown("#### Feature Importances")
        feat_dict = metrics.get("feature_importances", {})
        if feat_dict:
            df_feat = pd.DataFrame([
                {"Feature": k, "Importance": v} for k, v in feat_dict.items()
            ]).sort_values("Importance", ascending=True)

            fig_feat = go.Figure(go.Bar(
                x=df_feat["Importance"],
                y=df_feat["Feature"],
                orientation="h",
                marker_color="#2e7d32",
            ))
            fig_feat.update_layout(
                title="LightGBM Split Feature Importances",
                xaxis_title="Split Count",
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.caption("Feature importances will display once models are trained.")
