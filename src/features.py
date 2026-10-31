"""
Feature Engineering Module for Pehredar
Constructs atmospheric, temporal, agricultural, and cultural features:
  - Temporal: hour, day of week, weekend flag (traffic patterns)
  - Autoregressive: PM2.5 lags (1h, 3h, 6h, 24h) and rolling means (6h, 24h)
  - Meteorological: temperature, humidity, wind speed, wind direction, boundary layer height
  - Agrarian: is_stubble_season (Oct 1 - Nov 30), wind_from_farm_belt (200 deg - 320 deg)
  - Cultural / Diwali: days_from_diwali (clipped to +/-10) and is_diwali_night (18:00 to 03:00)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

# Official Diwali dates for relevant years
DIWALI_DATES = [
    pd.Timestamp("2023-11-12").date(),
    pd.Timestamp("2024-10-31").date(),
    pd.Timestamp("2025-10-20").date(),
    pd.Timestamp("2026-11-08").date(),
]

FEATURE_COLS = [
    "horizon_hours",
    "pm2_5_lag_1",
    "pm2_5_lag_3",
    "pm2_5_lag_6",
    "pm2_5_lag_24",
    "pm2_5_roll_mean_6",
    "pm2_5_roll_mean_24",
    "hour",
    "day_of_week",
    "is_weekend",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "boundary_layer_height",
    "wind_from_farm_belt",
    "is_stubble_season",
    "days_from_diwali",
    "is_diwali_night",
]


def compute_diwali_features(timestamps: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Computes:
      1. days_from_diwali: distance in days to nearest Diwali date, clipped to [-10, 10]
      2. is_diwali_night: 1 if between 18:00 on Diwali night to 03:00 the following morning
    """
    days_list = []
    night_list = []

    for ts in timestamps:
        t_date = ts.date()
        t_hour = ts.hour

        # Find closest diwali
        diffs = [(t_date - d).days for d in DIWALI_DATES]
        closest_diff = min(diffs, key=abs)
        clipped_diff = max(-10, min(10, closest_diff))
        days_list.append(float(clipped_diff))

        # Check if within diwali night window (18:00 - 03:00)
        is_night = 0
        for d in DIWALI_DATES:
            next_day = d + pd.Timedelta(days=1)
            if (t_date == d and t_hour >= 18) or (t_date == next_day and t_hour <= 3):
                is_night = 1
                break
        night_list.append(is_night)

    return pd.Series(days_list, index=timestamps.index), pd.Series(night_list, index=timestamps.index)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds base temporal, meteorological, and calendar features to an hourly dataframe.
    """
    df = df.copy()
    if not np.issubdtype(df["time"].dtype, np.datetime64):
        df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values("time").reset_index(drop=True)

    # Temporal features (traffic & diurnal rhythms)
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Stubble season (1 Oct to 30 Nov)
    df["is_stubble_season"] = df["time"].dt.month.isin([10, 11]).astype(int)

    # Wind from Malwa farm fire belt (West / South-West quadrant: 200 deg to 320 deg)
    wind_dir = df.get("wind_direction_10m", pd.Series(270.0, index=df.index))
    df["wind_from_farm_belt"] = ((wind_dir >= 200.0) & (wind_dir <= 320.0)).astype(int)

    # Diwali proximity and night flags
    days_diwali, is_diwali_night = compute_diwali_features(df["time"])
    df["days_from_diwali"] = days_diwali
    df["is_diwali_night"] = is_diwali_night

    # Fill boundary layer height if missing
    if "boundary_layer_height" not in df.columns or df["boundary_layer_height"].isna().all():
        df["boundary_layer_height"] = 400.0 + 600.0 * np.sin(np.pi * (df["hour"] - 6) / 18).clip(0, 1)
    else:
        df["boundary_layer_height"] = df["boundary_layer_height"].ffill().bfill().fillna(500.0)

    # Fill weather missing values
    for col in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    return df


def add_autoregressive_lags(df: pd.DataFrame, group_col: str = None) -> pd.DataFrame:
    """
    Computes PM2.5 lags (1h, 3h, 6h, 24h) and rolling means (6h, 24h).
    Lags are strictly shifted to avoid data leakage.
    """
    df = df.copy()

    def _calc_lags(sub_df):
        s = sub_df["pm2_5"]
        sub_df["pm2_5_lag_1"] = s.shift(1)
        sub_df["pm2_5_lag_3"] = s.shift(3)
        sub_df["pm2_5_lag_6"] = s.shift(6)
        sub_df["pm2_5_lag_24"] = s.shift(24)
        sub_df["pm2_5_roll_mean_6"] = s.shift(1).rolling(6, min_periods=3).mean()
        sub_df["pm2_5_roll_mean_24"] = s.shift(1).rolling(24, min_periods=12).mean()
        return sub_df

    if group_col and group_col in df.columns:
        df = df.groupby(group_col, group_keys=False).apply(_calc_lags)
    else:
        df = _calc_lags(df)

    return df


def prepare_multi_horizon_dataset(
    df: pd.DataFrame,
    max_horizon: int = 72,
    horizon_samples: List[int] = None,
) -> pd.DataFrame:
    """
    Vectorized preparation of multi-horizon training instances (t + h for h in 1..max_horizon).
    Fast and efficient pandas implementation.
    """
    df = add_base_features(df)
    df = add_autoregressive_lags(df, group_col="point_id" if "point_id" in df.columns else None)

    if horizon_samples is None:
        # Full 1..72 coverage with dense sampling
        horizon_samples = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 30, 36, 42, 48, 56, 64, 72]

    # Target weather/temporal features to shift to target hour t+h
    target_cols = [
        "hour",
        "day_of_week",
        "is_weekend",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "boundary_layer_height",
        "wind_from_farm_belt",
        "is_stubble_season",
        "days_from_diwali",
        "is_diwali_night",
    ]

    frames = []
    groups = [df] if "point_id" not in df.columns else [g for _, g in df.groupby("point_id")]

    for grp in groups:
        grp = grp.sort_values("time").reset_index(drop=True)
        for h in horizon_samples:
            sub = grp.copy()
            sub["horizon_hours"] = h
            sub["origin_time"] = sub["time"]
            sub["target_time"] = sub["time"].shift(-h)
            sub["y_true"] = sub["pm2_5"].shift(-h)

            # Target weather features at t+h
            for col in target_cols:
                sub[col] = sub[col].shift(-h)

            # Persistence baseline: PM2.5 24 hours prior to target (t+h-24)
            # If h >= 24, (t+h-24) is shifted by -(h-24)
            # If h < 24, (t+h-24) is shifted by +(24-h) into the past
            if h >= 24:
                sub["y_baseline_24h"] = sub["pm2_5"].shift(-(h - 24))
            else:
                sub["y_baseline_24h"] = sub["pm2_5"].shift(24 - h)

            # Drop rows with NaN in origin lags or target
            valid = sub.dropna(subset=[
                "pm2_5_lag_1", "pm2_5_lag_24", "y_true", "y_baseline_24h"
            ] + target_cols).copy()

            frames.append(valid)

    if not frames:
        return pd.DataFrame()

    out_df = pd.concat(frames, ignore_index=True)
    return out_df


if __name__ == "__main__":
    from data_fetcher import fetch_3x3_grid_data
    df = fetch_3x3_grid_data(demo_mode=True)
    feats = prepare_multi_horizon_dataset(df, max_horizon=72)
    print("Vectorized dataset shape:", feats.shape)
    print("Columns:", list(feats.columns))
