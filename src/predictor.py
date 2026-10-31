"""
Predictor Module for Pehredar
Trains LightGBM models (Mean, Quantile 0.1, Quantile 0.9) to predict PM2.5 at t+h (h = 1..72)
using weather at target hour and autoregressive origin features.
Evaluates against 'same as 24h ago' persistence baseline on an October Diwali window.
Interpolates 72-hour forecasts to all H3 hexagons using Inverse Distance Weighting (IDW).
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features import FEATURE_COLS, prepare_multi_horizon_dataset
from src.grid import generate_patiala_h3_grid
from src.safety import calculate_indian_aqi, calculate_safety_index

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_lightgbm_models(demo_mode: bool = False) -> Dict[str, Any]:
    """
    Trains LightGBM mean, q10, and q90 models.
    Performs time-based train/test split on October Diwali test window.
    Compares against 'same as 24h ago' persistence baseline.
    Saves models to models/ and metrics to data/metrics.json.
    """
    metrics_path = DATA_DIR / "metrics.json"

    # If demo_mode and already trained, return cached metrics
    if demo_mode and metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Loading data for LightGBM model training...")
    grid_path = DATA_DIR / "grid_3x3.parquet"
    hist_path = DATA_DIR / "historical_diwali.parquet"

    dfs = []
    if grid_path.exists():
        dfs.append(pd.read_parquet(grid_path))
    if hist_path.exists():
        dfs.append(pd.read_parquet(hist_path))

    if not dfs:
        raise FileNotFoundError("Neither grid_3x3.parquet nor historical_diwali.parquet found in data/.")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined raw records: {len(combined_df)}")

    # Prepare multi-horizon features
    print("Generating multi-horizon feature matrix (h=1..72)...")
    dataset = prepare_multi_horizon_dataset(combined_df, max_horizon=72)
    print(f"Generated {len(dataset)} training samples.")

    # Time-based split: test on the October Diwali window
    # In our historical dataset, Oct 15 - Nov 05, 2025 contains Diwali (2025-10-20)
    # or fallback to Oct 2024 (Diwali 2024-10-31)
    dataset["target_time"] = pd.to_datetime(dataset["target_time"])
    
    # Check available October windows
    target_times = dataset["target_time"]
    mask_2025 = (target_times >= "2025-10-15") & (target_times <= "2025-11-05")
    mask_2024 = (target_times >= "2024-10-20") & (target_times <= "2024-11-05")

    if mask_2025.sum() > 200:
        test_mask = mask_2025
        split_time = pd.Timestamp("2025-10-15")
        test_label = "October-November 2025 (Diwali Oct 20)"
    elif mask_2024.sum() > 200:
        test_mask = mask_2024
        split_time = pd.Timestamp("2024-10-20")
        test_label = "October-November 2024 (Diwali Oct 31)"
    else:
        # Fallback to last 15% chronologically
        split_time = dataset["target_time"].quantile(0.85)
        test_mask = dataset["target_time"] >= split_time
        test_label = f"Chronological split after {split_time.strftime('%Y-%m-%d')}"

    # Training set is strictly before split time
    train_mask = dataset["target_time"] < split_time

    X_train = dataset.loc[train_mask, FEATURE_COLS]
    y_train = dataset.loc[train_mask, "y_true"]

    X_test = dataset.loc[test_mask, FEATURE_COLS]
    y_test = dataset.loc[test_mask, "y_true"]
    baseline_test = dataset.loc[test_mask, "y_baseline_24h"]

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)} ({test_label})")

    # 1. Train Mean Regressor
    print("Training LightGBM Mean Regressor...")
    mean_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=100,
        learning_rate=0.08,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    mean_model.fit(X_train, y_train)

    # 2. Train Quantile 0.1 (Lower bound)
    print("Training LightGBM Quantile 0.1 (Lower Bound)...")
    q10_model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.1,
        n_estimators=80,
        learning_rate=0.08,
        num_leaves=31,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    q10_model.fit(X_train, y_train)

    # 3. Train Quantile 0.9 (Upper bound)
    print("Training LightGBM Quantile 0.9 (Upper Bound)...")
    q90_model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.9,
        n_estimators=80,
        learning_rate=0.08,
        num_leaves=31,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    q90_model.fit(X_train, y_train)

    # Evaluation on Test Window
    preds_mean = mean_model.predict(X_test)
    preds_q10 = q10_model.predict(X_test)
    preds_q90 = q90_model.predict(X_test)

    # Clean predictions to non-negative
    preds_mean = np.maximum(0.0, preds_mean)
    preds_q10 = np.maximum(0.0, preds_q10)
    preds_q90 = np.maximum(preds_q10, preds_q90)

    # Metrics
    model_mae = float(mean_absolute_error(y_test, preds_mean))
    model_rmse = float(np.sqrt(mean_squared_error(y_test, preds_mean)))
    base_mae = float(mean_absolute_error(y_test, baseline_test))
    base_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_test)))

    mae_improvement = round(((base_mae - model_mae) / base_mae) * 100, 2)
    rmse_improvement = round(((base_rmse - model_rmse) / base_rmse) * 100, 2)

    coverage_80 = round(float(np.mean((y_test >= preds_q10) & (y_test <= preds_q90)) * 100), 2)

    metrics = {
        "model": "LightGBM Multi-Horizon Quantile Regressor",
        "test_window": test_label,
        "test_samples": int(len(X_test)),
        "train_samples": int(len(X_train)),
        "model_mae": round(model_mae, 2),
        "model_rmse": round(model_rmse, 2),
        "baseline_mae": round(base_mae, 2),
        "baseline_rmse": round(base_rmse, 2),
        "mae_improvement_pct": mae_improvement,
        "rmse_improvement_pct": rmse_improvement,
        "prediction_interval_coverage_pct": coverage_80,
        "feature_importances": {
            feat: int(imp)
            for feat, imp in zip(FEATURE_COLS, mean_model.feature_importances_)
        },
    }

    print("\n--- LightGBM vs 24h Persistence Baseline ---")
    print(f"Model MAE:    {model_mae:.2f} µg/m³  (Baseline: {base_mae:.2f} µg/m³, {mae_improvement}% better)")
    print(f"Model RMSE:   {model_rmse:.2f} µg/m³ (Baseline: {base_rmse:.2f} µg/m³, {rmse_improvement}% better)")
    print(f"80% Interval Coverage: {coverage_80}%")

    # Save metrics JSON
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to {metrics_path}")

    # Save models
    joblib.dump(mean_model, MODELS_DIR / "lgbm_mean.joblib")
    joblib.dump(q10_model, MODELS_DIR / "lgbm_q10.joblib")
    joblib.dump(q90_model, MODELS_DIR / "lgbm_q90.joblib")
    print(f"Saved LightGBM models to {MODELS_DIR}")

    return metrics


def load_models() -> Tuple[Any, Any, Any]:
    """
    Loads trained LightGBM models from disk.
    If not found, triggers training first.
    """
    mean_file = MODELS_DIR / "lgbm_mean.joblib"
    q10_file = MODELS_DIR / "lgbm_q10.joblib"
    q90_file = MODELS_DIR / "lgbm_q90.joblib"

    if not (mean_file.exists() and q10_file.exists() and q90_file.exists()):
        print("Models not found on disk, training LightGBM suite...")
        train_lightgbm_models()

    mean_model = joblib.load(mean_file)
    q10_model = joblib.load(q10_file)
    q90_model = joblib.load(q90_file)
    return mean_model, q10_model, q90_model


def generate_72h_point_forecasts(
    mean_model, q10_model, q90_model, demo_mode: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Generates 72-hour forecast (t+1 .. t+72) for each of the 3x3 grid points
    using the trained LightGBM models and future weather forecast.
    """
    from src.features import add_base_features, add_autoregressive_lags

    grid_path = DATA_DIR / "grid_3x3.parquet"
    if not grid_path.exists():
        from src.data_fetcher import fetch_3x3_grid_data
        fetch_3x3_grid_data(demo_mode=demo_mode)

    df_grid = pd.read_parquet(grid_path)
    df_grid["time"] = pd.to_datetime(df_grid["time"])

    point_forecasts = {}

    for pt_id, grp in df_grid.groupby("point_id"):
        grp = grp.sort_values("time").reset_index(drop=True)
        # Add base and lag features
        grp = add_base_features(grp)
        grp = add_autoregressive_lags(grp)

        # Identify latest observation with valid historical PM2.5 (origin time t)
        valid_history = grp.dropna(subset=["pm2_5_lag_1", "pm2_5_lag_24"]).copy()
        if len(valid_history) < 24:
            continue

        # Origin row is the last valid observation row
        origin_row = valid_history.iloc[-1]
        origin_idx = grp[grp["time"] == origin_row["time"]].index[0]

        # Horizons 1 to 72
        future_rows = []
        for h in range(1, 73):
            target_idx = origin_idx + h
            if target_idx < len(grp):
                target_row = grp.iloc[target_idx]
            else:
                # Extrapolate target row if beyond forecast range
                target_row = origin_row.copy()
                target_row["time"] = origin_row["time"] + pd.Timedelta(hours=h)

            rec = {
                "horizon_hours": h,
                "pm2_5_lag_1": origin_row["pm2_5_lag_1"],
                "pm2_5_lag_3": origin_row["pm2_5_lag_3"],
                "pm2_5_lag_6": origin_row["pm2_5_lag_6"],
                "pm2_5_lag_24": origin_row["pm2_5_lag_24"],
                "pm2_5_roll_mean_6": origin_row["pm2_5_roll_mean_6"],
                "pm2_5_roll_mean_24": origin_row["pm2_5_roll_mean_24"],
                # Target weather features at t+h
                "hour": target_row["hour"],
                "day_of_week": target_row["day_of_week"],
                "is_weekend": target_row["is_weekend"],
                "temperature_2m": target_row["temperature_2m"],
                "relative_humidity_2m": target_row["relative_humidity_2m"],
                "wind_speed_10m": target_row["wind_speed_10m"],
                "wind_direction_10m": target_row["wind_direction_10m"],
                "boundary_layer_height": target_row["boundary_layer_height"],
                "wind_from_farm_belt": target_row["wind_from_farm_belt"],
                "is_stubble_season": target_row["is_stubble_season"],
                "days_from_diwali": target_row["days_from_diwali"],
                "is_diwali_night": target_row["is_diwali_night"],
                # Metadata
                "target_time": target_row["time"],
                "lat": grp["lat"].iloc[0],
                "lon": grp["lon"].iloc[0],
            }
            future_rows.append(rec)

        df_future = pd.DataFrame(future_rows)
        X_future = df_future[FEATURE_COLS]

        df_future["pm2_5_pred"] = np.maximum(5.0, mean_model.predict(X_future))
        df_future["pm2_5_q10"] = np.maximum(4.0, q10_model.predict(X_future))
        df_future["pm2_5_q90"] = np.maximum(df_future["pm2_5_q10"], q90_model.predict(X_future))

        point_forecasts[pt_id] = df_future

    return point_forecasts


def generate_and_save_per_hex_forecast(demo_mode: bool = False) -> pd.DataFrame:
    """
    Interpolates the 3x3 point forecasts to all 224 H3 hexagon centres across Patiala
    using Inverse Distance Weighting (IDW).
    Saves the 72h forecast grid to data/forecast_grid.parquet.
    """
    forecast_grid_path = DATA_DIR / "forecast_grid.parquet"

    # In demo_mode, load if available
    if demo_mode and forecast_grid_path.exists():
        print("[Pehredar DEMO_MODE] Loading per-hex forecast from cache.")
        return pd.read_parquet(forecast_grid_path)

    mean_m, q10_m, q90_m = load_models()
    pt_forecasts = generate_72h_point_forecasts(mean_m, q10_m, q90_m, demo_mode=demo_mode)
    h3_grid = generate_patiala_h3_grid()

    # Get points coordinates
    pt_meta = []
    for pt_id, df_pt in pt_forecasts.items():
        pt_meta.append({
            "point_id": pt_id,
            "lat": df_pt["lat"].iloc[0],
            "lon": df_pt["lon"].iloc[0],
            "df": df_pt.set_index("horizon_hours"),
        })

    print(f"Interpolating 72h forecasts across {len(h3_grid)} H3 hexagon cells via IDW...")

    all_hex_rows = []

    for hex_cell in h3_grid:
        h_lat = hex_cell["lat"]
        h_lon = hex_cell["lon"]
        h_id = hex_cell["hex_id"]
        locality = hex_cell["locality"]

        # Compute IDW weights to all 3x3 points
        weights = []
        for pt in pt_meta:
            dy = (pt["lat"] - h_lat) * 111.0
            dx = (pt["lon"] - h_lon) * (111.0 * math.cos(math.radians(30.34)))
            dist_km = math.sqrt(dx * dx + dy * dy)
            # Inverse distance weighting (p=2) with small epsilon
            w = 1.0 / ((dist_km + 0.05) ** 2)
            weights.append(w)

        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]

        # Interpolate each horizon hour (1 to 72)
        for h in range(1, 73):
            interp_pm25 = 0.0
            interp_q10 = 0.0
            interp_q90 = 0.0
            target_time = None
            hour_val = 0
            wind_dir_val = 270.0
            wind_spd_val = 8.0

            for w, pt in zip(norm_weights, pt_meta):
                row_h = pt["df"].loc[h]
                interp_pm25 += w * row_h["pm2_5_pred"]
                interp_q10 += w * row_h["pm2_5_q10"]
                interp_q90 += w * row_h["pm2_5_q90"]
                if target_time is None:
                    target_time = row_h["target_time"]
                    hour_val = int(row_h["hour"])
                    wind_dir_val = float(row_h["wind_direction_10m"])
                    wind_spd_val = float(row_h["wind_speed_10m"])

            # Compute Indian National AQI and Safety Indices
            aqi, aqi_cat, aqi_color = calculate_indian_aqi(interp_pm25)
            safety_healthy = calculate_safety_index(aqi, "healthy")
            safety_vuln = calculate_safety_index(aqi, "vulnerable")
            safety_resp = calculate_safety_index(aqi, "respiratory")

            all_hex_rows.append({
                "hex_id": h_id,
                "locality": locality,
                "lat": h_lat,
                "lon": h_lon,
                "horizon_hours": h,
                "target_time": target_time,
                "hour": hour_val,
                "pm2_5": round(interp_pm25, 1),
                "pm2_5_q10": round(interp_q10, 1),
                "pm2_5_q90": round(interp_q90, 1),
                "aqi": aqi,
                "aqi_category": aqi_cat,
                "aqi_color": aqi_color,
                "safety_healthy": safety_healthy,
                "safety_vulnerable": safety_vuln,
                "safety_respiratory": safety_resp,
                "wind_direction_10m": wind_dir_val,
                "wind_speed_10m": wind_spd_val,
            })

    df_forecast_grid = pd.DataFrame(all_hex_rows)
    df_forecast_grid.to_parquet(forecast_grid_path, index=False)
    print(f"Saved complete 72h per-hex forecast ({len(df_forecast_grid)} records) to {forecast_grid_path}")
    return df_forecast_grid


if __name__ == "__main__":
    print("Testing predictor module...")
    metrics = train_lightgbm_models(demo_mode=False)
    print("Metrics:", metrics)
    df_fc = generate_and_save_per_hex_forecast(demo_mode=False)
    print("Per-hex forecast preview:", df_fc.head(3))
