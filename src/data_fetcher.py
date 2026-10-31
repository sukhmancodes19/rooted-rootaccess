"""
Data Fetcher Module for Pehredar
Fetches hourly Air Quality and Weather data from Open-Meteo APIs:
  1. 3x3 Spatial Grid across Patiala bounding box (past 92 days + 4 forecast days)
  2. Historical Oct-Nov data for Diwali training (2023, 2024, 2025)
  3. Caches all API responses to data/ and supports DEMO_MODE (offline cache run)
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 3x3 spatial grid points across Patiala bounding box (lat 30.29-30.40, lon 76.32-76.47)
GRID_LATS = [30.29, 30.345, 30.40]
GRID_LONS = [76.32, 76.395, 76.47]

GRID_3X3_POINTS = [
    {"point_id": f"pt_{r}_{c}", "lat": round(lat, 4), "lon": round(lon, 4)}
    for r, lat in enumerate(GRID_LATS)
    for c, lon in enumerate(GRID_LONS)
]

CITY_CENTRE = {"lat": 30.34, "lon": 76.39}

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://api.open-meteo.com/v1/forecast"
WX_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_point_aq_and_weather(
    lat: float,
    lon: float,
    past_days: int = 92,
    forecast_days: int = 4,
    timeout: int = 15,
) -> pd.DataFrame:
    """
    Fetches hourly AQ (pm2_5, pm10) and Weather (temp, humidity, wind, boundary_layer_height)
    for a given coordinate point.
    """
    # 1. Fetch Air Quality
    aq_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["pm2_5", "pm10"],
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "Asia/Kolkata",
    }
    r_aq = requests.get(AQ_URL, params=aq_params, timeout=timeout)
    r_aq.raise_for_status()
    aq_json = r_aq.json()
    hourly_aq = aq_json.get("hourly", {})

    df_aq = pd.DataFrame({
        "time": pd.to_datetime(hourly_aq.get("time", [])),
        "pm2_5": hourly_aq.get("pm2_5", []),
        "pm10": hourly_aq.get("pm10", []),
    })

    # 2. Fetch Weather
    wx_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "boundary_layer_height",
        ],
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "Asia/Kolkata",
    }
    r_wx = requests.get(WX_URL, params=wx_params, timeout=timeout)
    r_wx.raise_for_status()
    wx_json = r_wx.json()
    hourly_wx = wx_json.get("hourly", {})

    df_wx = pd.DataFrame({
        "time": pd.to_datetime(hourly_wx.get("time", [])),
        "temperature_2m": hourly_wx.get("temperature_2m", []),
        "relative_humidity_2m": hourly_wx.get("relative_humidity_2m", []),
        "wind_speed_10m": hourly_wx.get("wind_speed_10m", []),
        "wind_direction_10m": hourly_wx.get("wind_direction_10m", []),
        "boundary_layer_height": hourly_wx.get("boundary_layer_height", []),
    })

    # Merge on time
    df_merged = pd.merge(df_aq, df_wx, on="time", how="inner")
    df_merged["lat"] = lat
    df_merged["lon"] = lon
    return df_merged


def fetch_3x3_grid_data(demo_mode: bool = False) -> pd.DataFrame:
    """
    Fetches past 92 days + 4 forecast days for 3x3 grid of points across Patiala.
    Saves to data/grid_3x3.parquet.
    If demo_mode=True or network fails, loads from data/grid_3x3.parquet.
    """
    parquet_path = DATA_DIR / "grid_3x3.parquet"

    if demo_mode:
        if parquet_path.exists():
            print("[Pehredar DEMO_MODE] Loading 3x3 grid data from cache.")
            return pd.read_parquet(parquet_path)
        else:
            print("[Pehredar DEMO_MODE Warning] Cache not found, fetching live data once to seed cache...")

    dfs = []
    print(f"Fetching Open-Meteo AQ and weather for {len(GRID_3X3_POINTS)} grid points...")
    for pt in GRID_3X3_POINTS:
        try:
            pt_df = fetch_point_aq_and_weather(pt["lat"], pt["lon"])
            pt_df["point_id"] = pt["point_id"]
            dfs.append(pt_df)
            time.sleep(0.3)  # Polite API spacing
        except Exception as e:
            print(f"Error fetching point {pt['point_id']} ({pt['lat']}, {pt['lon']}): {e}")

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_parquet(parquet_path, index=False)
        print(f"Saved {len(combined_df)} records to {parquet_path}")
        return combined_df
    elif parquet_path.exists():
        print("Network request failed, falling back to cached grid_3x3.parquet.")
        return pd.read_parquet(parquet_path)
    else:
        raise RuntimeError("Failed to fetch 3x3 grid data and no cached data exists.")


def fetch_diwali_historical_data(demo_mode: bool = False) -> pd.DataFrame:
    """
    Fetches Open-Meteo Air Quality and Weather for Oct-Nov 2023, 2024, and 2025
    at the city centre (30.34, 76.39) for Diwali historical learning.
    Skips any unavailable year and prints which years worked.
    Saves to data/historical_diwali.parquet.
    """
    parquet_path = DATA_DIR / "historical_diwali.parquet"

    if demo_mode:
        if parquet_path.exists():
            print("[Pehredar DEMO_MODE] Loading historical Diwali data from cache.")
            return pd.read_parquet(parquet_path)
        else:
            print("[Pehredar DEMO_MODE Warning] Historical cache not found, fetching live...")

    years = [2023, 2024, 2025]
    successful_years = []
    dfs = []

    for yr in years:
        start_date = f"{yr}-10-01"
        end_date = f"{yr}-11-30"

        try:
            # 1. Historical Air Quality
            aq_params = {
                "latitude": CITY_CENTRE["lat"],
                "longitude": CITY_CENTRE["lon"],
                "hourly": ["pm2_5", "pm10"],
                "start_date": start_date,
                "end_date": end_date,
                "timezone": "Asia/Kolkata",
            }
            r_aq = requests.get(AQ_URL, params=aq_params, timeout=15)
            if r_aq.status_code != 200:
                print(f"Year {yr} AQ archive returned status {r_aq.status_code}, skipping.")
                continue

            aq_json = r_aq.json().get("hourly", {})
            df_aq = pd.DataFrame({
                "time": pd.to_datetime(aq_json.get("time", [])),
                "pm2_5": aq_json.get("pm2_5", []),
                "pm10": aq_json.get("pm10", []),
            })

            # 2. Historical Weather Archive
            wx_params = {
                "latitude": CITY_CENTRE["lat"],
                "longitude": CITY_CENTRE["lon"],
                "hourly": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "boundary_layer_height",
                ],
                "start_date": start_date,
                "end_date": end_date,
                "timezone": "Asia/Kolkata",
            }
            r_wx = requests.get(WX_ARCHIVE_URL, params=wx_params, timeout=15)
            if r_wx.status_code != 200:
                # Fallback to forecast API in case year is recent
                r_wx = requests.get(WX_URL, params=wx_params, timeout=15)

            if r_wx.status_code == 200:
                wx_json = r_wx.json().get("hourly", {})
                df_wx = pd.DataFrame({
                    "time": pd.to_datetime(wx_json.get("time", [])),
                    "temperature_2m": wx_json.get("temperature_2m", []),
                    "relative_humidity_2m": wx_json.get("relative_humidity_2m", []),
                    "wind_speed_10m": wx_json.get("wind_speed_10m", []),
                    "wind_direction_10m": wx_json.get("wind_direction_10m", []),
                    "boundary_layer_height": wx_json.get("boundary_layer_height", []),
                })
            else:
                df_wx = pd.DataFrame({"time": df_aq["time"]})

            merged = pd.merge(df_aq, df_wx, on="time", how="left")
            merged["lat"] = CITY_CENTRE["lat"]
            merged["lon"] = CITY_CENTRE["lon"]
            merged["year"] = yr
            merged["point_id"] = f"hist_city_centre_{yr}"
            dfs.append(merged)
            successful_years.append(yr)
            print(f"Successfully fetched Oct-Nov historical data for year {yr} ({len(merged)} records)")
            time.sleep(0.3)
        except Exception as e:
            print(f"Failed to fetch year {yr}: {e}")

    print(f"Diwali historical fetch summary: Successfully loaded years {successful_years}")

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_parquet(parquet_path, index=False)
        return combined_df
    elif parquet_path.exists():
        print("Historical fetch failed, loading from cache.")
        return pd.read_parquet(parquet_path)
    else:
        raise RuntimeError("No historical data could be fetched or found in cache.")


if __name__ == "__main__":
    print("Testing data_fetcher...")
    df_grid = fetch_3x3_grid_data(demo_mode=False)
    print("Grid shape:", df_grid.shape)
    df_hist = fetch_diwali_historical_data(demo_mode=False)
    print("Hist shape:", df_hist.shape)
