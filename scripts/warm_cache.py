"""
Demo Warm-Up Script for Pehredar
Pre-geocodes key Patiala locations and pre-fetches OSRM routes for:
  1. Thapar University -> Baradari
  2. Thapar University -> Omaxe Mall
  3. Tripuri -> Leela Bhawan
  4. Model Town -> Patiala Bus Stand
across all modes: walk, bike, car.

Ensures instant, reliable offline/live demo experience without network latency.

Author: Team Rooted – Pehredar, RootAccess Hackathon
"""

import sys
from pathlib import Path

# Safe encoding for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.router import geocode_place, plan_routes, get_known_localities

TARGET_PAIRS = [
    ("Thapar University", "Baradari"),
    ("Thapar University", "Omaxe Mall"),
    ("Tripuri", "Leela Bhawan"),
    ("Model Town", "Patiala Bus Stand"),
]

MODES = ["walk", "bike", "car"]


def run_warmup():
    print("=" * 70)
    print("[PEHREDAR] CACHE WARM-UP SCRIPT FOR LIVE DEMO")
    print("=" * 70)

    # 1. Warm geocode cache for all known localities
    print("\n[Step 1/2] Pre-geocoding known localities & POIs...")
    localities = get_known_localities()
    for loc in localities:
        try:
            lat, lon, disp = geocode_place(loc)
            print(f"  [OK] {loc:<20} -> ({lat:.4f}, {lon:.4f}) : {disp}")
        except Exception as e:
            print(f"  [FAIL] {loc:<20} -> FAILED: {e}")

    # Ensure forecast grid exists for scoring
    grid_parquet = PROJECT_ROOT / "data" / "forecast_grid.parquet"
    if grid_parquet.exists():
        df_forecast = pd.read_parquet(grid_parquet)
    else:
        print("  [WARN] data/forecast_grid.parquet not found, empty scoring will be used.")
        df_forecast = pd.DataFrame()

    # 2. Warm route cache for target pairs
    print(f"\n[Step 2/2] Pre-fetching routes for {len(TARGET_PAIRS)} key demo pairs across {len(MODES)} modes...")
    results = []

    for origin, dest in TARGET_PAIRS:
        print(f"\n--- Route: {origin} -> {dest} ---")
        for mode in MODES:
            try:
                routes = plan_routes(
                    from_place=origin,
                    to_place=dest,
                    departure_offset_hours=0,
                    mode=mode,
                    health_profile="healthy",
                    df_forecast=df_forecast,
                )
                if routes:
                    best = routes[0]
                    status_label = "OSRM Live Route" if best.get("is_real_route", True) else "Approx. Offline Route"
                    print(
                        f"  [OK] [{mode.upper():<4}] {best['distance_km']} km | "
                        f"{best['duration_min']} min | Safety Score: {best['safety_score']}/10 | {status_label}"
                    )
                    results.append({
                        "Pair": f"{origin} -> {dest}",
                        "Mode": mode,
                        "Distance (km)": best["distance_km"],
                        "Duration (min)": best["duration_min"],
                        "Safety Score": f"{best['safety_score']}/10",
                        "Source": status_label,
                        "Status": "Success",
                    })
                else:
                    print(f"  [FAIL] [{mode.upper():<4}] No route returned")
                    results.append({"Pair": f"{origin} -> {dest}", "Mode": mode, "Status": "Empty"})
            except Exception as e:
                print(f"  [FAIL] [{mode.upper():<4}] Failed: {e}")
                results.append({"Pair": f"{origin} -> {dest}", "Mode": mode, "Status": f"Failed: {e}"})

    # Summary table
    print("\n" + "=" * 70)
    print("WARM-UP SUMMARY TABLE")
    print("=" * 70)
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

    success_count = sum(1 for r in results if r.get("Status") == "Success")
    total_count = len(results)
    print("\n" + "=" * 70)
    print(f"Cache warm-up completed: {success_count}/{total_count} routes cached and ready!")
    print("=" * 70)


if __name__ == "__main__":
    run_warmup()
