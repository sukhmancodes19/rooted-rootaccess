"""
Safest Route Planner for Pehredar
Uses the public OSRM server for routing, Nominatim for geocoding,
and the H3 forecast grid for per-hex air quality scoring along routes.

Resilient Geocoding:
  1. Always checks disk cache first (ANY mode, including DEMO_MODE).
  2. Offline gazetteer matching (case-insensitive & fuzzy) against src/grid.py
     localities + Thapar University + Omaxe Mall.
  3. Online Nominatim with 3+ query variants if not in gazetteer, caching every success.
  4. Raises GeocodeFallbackNeeded with known_localities list so UI can show a dropdown.

Resilient Routing:
  1. Cache-first in any mode.
  2. Calls OSRM public server.
  3. Falls back to straight-line route sampled every 100 m, clearly labelled
     "approximate route (offline)", still fully scored with Safety Index.

Author: Team Rooted – Pehredar, RootAccess Hackathon
"""

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h3
import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_BASE = "http://router.project-osrm.org/route/v1"

OSRM_PROFILES: Dict[str, str] = {
    "walk": "foot",
    "bike": "bike",
    "car": "driving",
}

CAR_EXPOSURE_FACTOR = 0.5

SAFETY_COLUMNS: Dict[str, str] = {
    "healthy": "safety_healthy",
    "vulnerable": "safety_vulnerable",
    "respiratory": "safety_respiratory",
}

_GEOCODE_CACHE_PATH = DATA_DIR / "geocode_cache.json"

# Extra POIs beyond the 12 base LOCALITIES
_EXTRA_POIS: Dict[str, Tuple[float, float]] = {
    "Omaxe Mall": (30.3382, 76.3967),           # Mall Road, Patiala
    "Thapar University": (30.3545, 76.3658),    # Thapar Institute Campus
    "Qila Mubarak": (30.3360, 76.3940),
    "NIS Patiala": (30.3470, 76.3975),
    "Rajindra Hospital": (30.3389, 76.3972),
}

# Fuzzy aliases mapping lowercase query strings to canonical gazetteer names
_FUZZY_ALIASES: Dict[str, str] = {
    # Baradari
    "baradari": "Baradari",
    "baradari gardens": "Baradari",
    "baradari garden": "Baradari",
    "bara dari": "Baradari",
    "baradari bagh": "Baradari",
    # Thapar
    "thapar": "Thapar University",
    "thapar university": "Thapar University",
    "thapar institute": "Thapar University",
    "tiet": "Thapar University",
    "thapar patiala": "Thapar University",
    "thapar campus": "Thapar University",
    # Bus Stand
    "bus stand": "Patiala Bus Stand",
    "bus stand patiala": "Patiala Bus Stand",
    "patiala bus stand": "Patiala Bus Stand",
    "bus stop": "Patiala Bus Stand",
    "old bus stand": "Patiala Bus Stand",
    # Leela Bhawan
    "leela bhawan": "Leela Bhawan",
    "leela bawan": "Leela Bhawan",
    "leela": "Leela Bhawan",
    "lila bhawan": "Leela Bhawan",
    # Model Town
    "model town": "Model Town",
    "model town patiala": "Model Town",
    # Tripuri
    "tripuri": "Tripuri",
    "tripuri town": "Tripuri",
    "tripari": "Tripuri",
    # Urban Estate
    "urban estate": "Urban Estate",
    "urban estate phase 1": "Urban Estate",
    "urban estate phase 2": "Urban Estate",
    "urban estate patiala": "Urban Estate",
    # Bhupindra Road
    "bhupindra road": "Bhupindra Road",
    "bhupindra": "Bhupindra Road",
    "bhupinder road": "Bhupindra Road",
    # Sanauri Adda
    "sanauri adda": "Sanauri Adda",
    "sanauri": "Sanauri Adda",
    "sanouri adda": "Sanauri Adda",
    # Rajpura Road
    "rajpura road": "Rajpura Road",
    "rajpura bypass": "Rajpura Road",
    # Gurbax Colony
    "gurbax colony": "Gurbax Colony",
    "gurbux colony": "Gurbax Colony",
    "gurbax": "Gurbax Colony",
    # Sirhind Road
    "sirhind road": "Sirhind Road",
    "sirhind": "Sirhind Road",
    # Omaxe Mall
    "omaxe": "Omaxe Mall",
    "omaxe mall": "Omaxe Mall",
    "omaxe mall patiala": "Omaxe Mall",
    "omaxe patiala": "Omaxe Mall",
    "mall of patiala": "Omaxe Mall",
    # Qila Mubarak
    "qila mubarak": "Qila Mubarak",
    "qila": "Qila Mubarak",
    "quila mubarak": "Qila Mubarak",
    # NIS
    "nis": "NIS Patiala",
    "nis patiala": "NIS Patiala",
    "national institute of sports": "NIS Patiala",
    # Rajindra Hospital
    "rajindra hospital": "Rajindra Hospital",
    "rajindra": "Rajindra Hospital",
}

_NOMINATIM_VARIANTS = [
    "{name}",
    "{name}, Patiala",
    "{name}, Patiala, Punjab, India",
    "{name}, Punjab, India",
]

_GAZETTEER_CACHE: Optional[Dict[str, Tuple[float, float]]] = None


def get_gazetteer() -> Dict[str, Tuple[float, float]]:
    """Returns the full offline gazetteer of Patiala localities and landmarks."""
    global _GAZETTEER_CACHE
    if _GAZETTEER_CACHE is not None:
        return _GAZETTEER_CACHE
    try:
        from src.grid import LOCALITIES
        combined = dict(LOCALITIES)
    except ImportError:
        combined = {}
    combined.update(_EXTRA_POIS)
    _GAZETTEER_CACHE = combined
    return _GAZETTEER_CACHE


def get_known_localities() -> List[str]:
    """Returns sorted list of known locality/landmark names for UI dropdowns."""
    gz = get_gazetteer()
    return sorted(gz.keys())


class GeocodeFallbackNeeded(Exception):
    """Raised when geocoding fails all tiers, providing known localities for a fallback dropdown."""
    def __init__(self, message: str, known_localities: Optional[List[str]] = None):
        super().__init__(message)
        self.known_localities = known_localities or get_known_localities()


# ---------------------------------------------------------------------------
# Geocode Cache Management
# ---------------------------------------------------------------------------

def _load_geocode_cache() -> Dict[str, Any]:
    if _GEOCODE_CACHE_PATH.exists():
        try:
            with open(_GEOCODE_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_geocode_cache(cache: Dict[str, Any]) -> None:
    try:
        with open(_GEOCODE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def _clean_query(q: str) -> str:
    """Normalize query text for fuzzy comparison."""
    q = q.lower().strip()
    q = re.sub(r"[,\.\-\/\\_\(\)]", " ", q)
    q = re.sub(r"\b(patiala|punjab|india)\b", "", q)
    return " ".join(q.split())


def _nominatim_query(query: str, timeout: int = 8) -> Optional[Tuple[float, float, str]]:
    """Requests Nominatim with User-Agent header; returns (lat, lon, display_name) or None."""
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "Pehredar-AirSense/1.0 (Hackathon Prototype)"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if data and len(data) > 0:
            top = data[0]
            lat = float(top["lat"])
            lon = float(top["lon"])
            # Validate within greater Patiala region (lat 30.1-30.6, lon 76.1-76.6)
            if 30.1 <= lat <= 30.6 and 76.1 <= lon <= 76.6:
                return lat, lon, top.get("display_name", query)
    except Exception:
        pass
    return None


def geocode_place(place_name: str) -> Tuple[float, float, str]:
    """
    Geocodes a place name reliably:
      1. Always checks disk cache first (ANY mode).
      2. Matches against offline gazetteer (case-insensitive & fuzzy aliases).
      3. Tries Nominatim with query variants, caching every success.
      4. If all fails, raises GeocodeFallbackNeeded.

    Returns:
      (lat, lon, display_name)
    """
    raw_key = place_name.lower().strip()
    clean_key = _clean_query(place_name)

    # 1. Disk cache check
    cache = _load_geocode_cache()
    for k in (raw_key, clean_key, place_name):
        if k in cache:
            entry = cache[k]
            return float(entry["lat"]), float(entry["lon"]), entry["display_name"]

    gz = get_gazetteer()

    # 2. Offline gazetteer exact match (case-insensitive)
    for name, (lat, lon) in gz.items():
        if name.lower() == raw_key or name.lower() == clean_key:
            display = f"{name}, Patiala"
            cache[raw_key] = {"lat": lat, "lon": lon, "display_name": display}
            _save_geocode_cache(cache)
            return lat, lon, display

    # 2b. Fuzzy alias match
    if raw_key in _FUZZY_ALIASES:
        canonical = _FUZZY_ALIASES[raw_key]
        if canonical in gz:
            lat, lon = gz[canonical]
            display = f"{canonical}, Patiala"
            cache[raw_key] = {"lat": lat, "lon": lon, "display_name": display}
            _save_geocode_cache(cache)
            return lat, lon, display

    if clean_key in _FUZZY_ALIASES:
        canonical = _FUZZY_ALIASES[clean_key]
        if canonical in gz:
            lat, lon = gz[canonical]
            display = f"{canonical}, Patiala"
            cache[raw_key] = {"lat": lat, "lon": lon, "display_name": display}
            _save_geocode_cache(cache)
            return lat, lon, display

    # 2c. Substring match against gazetteer or aliases
    for alias, canonical in _FUZZY_ALIASES.items():
        if alias in raw_key or raw_key in alias or alias in clean_key:
            if canonical in gz:
                lat, lon = gz[canonical]
                display = f"{canonical}, Patiala"
                cache[raw_key] = {"lat": lat, "lon": lon, "display_name": display}
                _save_geocode_cache(cache)
                return lat, lon, display

    for name, (lat, lon) in gz.items():
        n_clean = _clean_query(name)
        if n_clean in clean_key or clean_key in n_clean:
            display = f"{name}, Patiala"
            cache[raw_key] = {"lat": lat, "lon": lon, "display_name": display}
            _save_geocode_cache(cache)
            return lat, lon, display

    # 3. Online Nominatim with variants (cache every success)
    for variant in _NOMINATIM_VARIANTS:
        q_str = variant.format(name=place_name.strip())
        res = _nominatim_query(q_str)
        if res:
            lat, lon, disp = res
            cache[raw_key] = {"lat": lat, "lon": lon, "display_name": disp}
            _save_geocode_cache(cache)
            return lat, lon, disp
        time.sleep(0.15)

    # 4. Fallback needed
    raise GeocodeFallbackNeeded(
        f"Could not find coordinates for '{place_name}'.",
        known_localities=get_known_localities(),
    )


# ---------------------------------------------------------------------------
# Routing Cache & Fallback
# ---------------------------------------------------------------------------

def _route_cache_key(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float, mode: str) -> str:
    return hashlib.md5(f"{origin_lat:.4f},{origin_lon:.4f};{dest_lat:.4f},{dest_lon:.4f};{mode}".encode()).hexdigest()[:12]


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2.0) ** 2
    return r * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _build_straight_line_fallback(
    origin_lat: float, origin_lon: float,
    dest_lat: float, dest_lon: float,
    mode: str,
) -> Dict[str, Any]:
    """
    Builds a synthetic straight-line route sampled every ~100 m between origin and dest.
    Speed estimates (km/h): walk=5, bike=15, car=35.
    """
    total_dist_m = _haversine_m(origin_lat, origin_lon, dest_lat, dest_lon)
    speeds = {"walk": 5.0, "bike": 15.0, "car": 35.0}
    speed_kmh = speeds.get(mode, 5.0)
    duration_sec = (total_dist_m / 1000.0 / speed_kmh) * 3600.0

    n_samples = max(2, int(total_dist_m / 100.0) + 1)
    coords = []
    for i in range(n_samples):
        frac = i / (n_samples - 1)
        lat = origin_lat + frac * (dest_lat - origin_lat)
        lon = origin_lon + frac * (dest_lon - origin_lon)
        coords.append([lon, lat])  # GeoJSON [lon, lat]

    return {
        "geometry": {"type": "LineString", "coordinates": coords},
        "duration": duration_sec,
        "distance": total_dist_m,
        "_is_fallback": True,
        "_label": "approximate route (offline)",
    }


def fetch_osrm_routes(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    mode: str = "walk",
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Fetches routes:
      1. Check disk cache first (ANY mode).
      2. Call OSRM public server.
      3. Fall back to straight-line route sampled every 100m, labelled 'approximate route (offline)'.

    Returns:
      (routes_list, is_real_osrm: bool)
    """
    key = _route_cache_key(origin_lat, origin_lon, dest_lat, dest_lon, mode)
    cache_path = DATA_DIR / f"route_cache_{key}.json"

    # 1. Disk cache
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict) and "routes" in saved:
                return saved["routes"], saved.get("is_real", True)
            elif isinstance(saved, list):
                return saved, True
        except Exception:
            pass

    # 2. OSRM API
    profile = OSRM_PROFILES.get(mode, "foot")
    url = f"{OSRM_BASE}/{profile}/{origin_lon:.6f},{origin_lat:.6f};{dest_lon:.6f},{dest_lat:.6f}?alternatives=true&overview=full&geometries=geojson"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == "Ok" and data.get("routes"):
            routes = data["routes"]
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"routes": routes, "is_real": True}, f)
            return routes, True
    except Exception:
        pass

    # 3. Straight-line fallback
    fallback = _build_straight_line_fallback(origin_lat, origin_lon, dest_lat, dest_lon, mode)
    return [fallback], False


# ---------------------------------------------------------------------------
# Sampling & Scoring
# ---------------------------------------------------------------------------

def sample_route_points(
    coords: List[List[float]],
    interval_m: float = 100.0,
) -> List[Tuple[float, float]]:
    """Interpolates points every `interval_m` metres along GeoJSON [[lon, lat], ...]."""
    if not coords:
        return []
    sampled = [(coords[0][1], coords[0][0])]
    accum = 0.0

    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        seg_dist = _haversine_m(lat1, lon1, lat2, lon2)
        if seg_dist == 0:
            continue

        rem = interval_m - accum
        if accum + seg_dist >= interval_m:
            d = rem
            while d <= seg_dist:
                frac = d / seg_dist
                s_lat = lat1 + frac * (lat2 - lat1)
                s_lon = lon1 + frac * (lon2 - lon1)
                sampled.append((s_lat, s_lon))
                d += interval_m
            accum = seg_dist - (d - interval_m)
        else:
            accum += seg_dist

    last_lon, last_lat = coords[-1]
    if sampled[-1] != (last_lat, last_lon):
        sampled.append((last_lat, last_lon))
    return sampled


def get_hex_forecast_for_time(
    lat: float,
    lon: float,
    target_time: pd.Timestamp,
    df_forecast: pd.DataFrame,
) -> Dict[str, Any]:
    """Finds H3 cell at resolution 8 and returns matching forecast record."""
    try:
        hex_id = h3.latlng_to_cell(lat, lon, 8)
    except Exception:
        hex_id = None

    subset = df_forecast[df_forecast["hex_id"] == hex_id] if hex_id else pd.DataFrame()
    if subset.empty:
        subset = df_forecast.copy()

    if "target_time" in subset.columns:
        try:
            t = pd.to_datetime(target_time)
            if t.tzinfo is not None:
                t = t.tz_localize(None)
            times = pd.to_datetime(subset["target_time"])
            if times.dt.tz is not None:
                times = times.dt.tz_localize(None)
            diffs = (times - t).abs()
            row = subset.loc[diffs.idxmin()]
        except Exception:
            row = subset.iloc[0]
    else:
        now = pd.Timestamp.now()
        h_ahead = max(1, min(72, int((pd.to_datetime(target_time) - now).total_seconds() / 3600)))
        h_sub = subset[subset["horizon_hours"] == h_ahead]
        row = h_sub.iloc[0] if not h_sub.empty else subset.iloc[0]

    return {
        "hex_id": hex_id,
        "pm2_5": float(row.get("pm2_5", 50.0)),
        "aqi": int(row.get("aqi", 100)),
        "aqi_category": str(row.get("aqi_category", "Moderate")),
        "aqi_color": str(row.get("aqi_color", "#fbc02d")),
        "safety_healthy": float(row.get("safety_healthy", 6.0)),
        "safety_vulnerable": float(row.get("safety_vulnerable", 5.0)),
        "safety_respiratory": float(row.get("safety_respiratory", 4.0)),
        "locality": str(row.get("locality", "Patiala")),
    }


def score_route(
    route_coords: List[List[float]],
    duration_sec: float,
    departure_time: pd.Timestamp,
    df_forecast: pd.DataFrame,
    health_profile: str = "healthy",
    mode: str = "walk",
) -> Dict[str, Any]:
    safety_col = SAFETY_COLUMNS.get(health_profile, "safety_healthy")
    is_car = mode == "car"

    sampled_points = sample_route_points(route_coords, interval_m=100.0)
    if not sampled_points:
        return {
            "safety_score": 5.0,
            "worst_locality": "Unknown",
            "worst_aqi": 100,
            "worst_pm25": 50.0,
            "worst_safety": 5.0,
            "points": [],
        }

    total_dist_m = 0.0
    seg_dists = [0.0]
    for i in range(1, len(sampled_points)):
        d = _haversine_m(*sampled_points[i - 1], *sampled_points[i])
        total_dist_m += d
        seg_dists.append(total_dist_m)

    scored_points = []
    worst_safety = 10.0
    worst_loc = "Unknown"
    worst_aqi = 0
    worst_pm25 = 0.0

    for i, (lat, lon) in enumerate(sampled_points):
        frac = seg_dists[i] / total_dist_m if total_dist_m > 0 else 0.0
        elapsed_sec = frac * duration_sec
        p_time = departure_time + pd.Timedelta(seconds=elapsed_sec)

        fc = get_hex_forecast_for_time(lat, lon, p_time, df_forecast)
        raw_safety = fc.get(safety_col, 5.0)

        # Car halving exposure factor
        if is_car:
            loss = 10.0 - raw_safety
            adj_safety = min(10.0, 10.0 - loss * CAR_EXPOSURE_FACTOR)
        else:
            adj_safety = raw_safety

        scored_points.append({
            "lat": lat,
            "lon": lon,
            "safety": adj_safety,
            "aqi": fc["aqi"],
            "pm2_5": fc["pm2_5"],
            "locality": fc["locality"],
            "elapsed_min": round(elapsed_sec / 60, 1),
        })

        if adj_safety < worst_safety:
            worst_safety = adj_safety
            worst_loc = fc["locality"]
            worst_aqi = fc["aqi"]
            worst_pm25 = fc["pm2_5"]

    avg_safety = float(np.mean([p["safety"] for p in scored_points]))
    return {
        "safety_score": round(avg_safety, 1),
        "worst_locality": worst_loc,
        "worst_aqi": worst_aqi,
        "worst_pm25": round(worst_pm25, 1),
        "worst_safety": round(worst_safety, 1),
        "points": scored_points,
    }


def plan_routes(
    from_place: str,
    to_place: str,
    departure_offset_hours: float,
    mode: str,
    health_profile: str,
    df_forecast: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Main entry point for route planning.
    Geocodes origin and destination, fetches up to 3 routes (falling back to straight line if offline),
    scores with Safety Index, and returns routes sorted by safety.
    """
    departure_time = pd.Timestamp.now() + pd.Timedelta(hours=departure_offset_hours)

    origin_lat, origin_lon, origin_name = geocode_place(from_place)
    dest_lat, dest_lon, dest_name = geocode_place(to_place)

    osrm_routes, is_real = fetch_osrm_routes(origin_lat, origin_lon, dest_lat, dest_lon, mode)

    results = []
    for idx, r in enumerate(osrm_routes):
        dur_sec = float(r.get("duration", 0))
        dist_m = float(r.get("distance", 0))
        coords = r.get("geometry", {}).get("coordinates", [])
        route_label = r.get("_label", "")

        score_res = score_route(
            route_coords=coords,
            duration_sec=dur_sec,
            departure_time=departure_time,
            df_forecast=df_forecast,
            health_profile=health_profile,
            mode=mode,
        )

        results.append({
            "route_idx": idx + 1,
            "safety_score": score_res["safety_score"],
            "duration_min": round(dur_sec / 60.0, 1),
            "distance_km": round(dist_m / 1000.0, 2),
            "worst_locality": score_res["worst_locality"],
            "worst_aqi": score_res["worst_aqi"],
            "worst_pm25": score_res["worst_pm25"],
            "worst_safety": score_res["worst_safety"],
            "geometry_coords": coords,
            "scored_points": score_res["points"],
            "origin_latlon": (origin_lat, origin_lon),
            "dest_latlon": (dest_lat, dest_lon),
            "origin_name": origin_name,
            "dest_name": dest_name,
            "is_real_route": is_real,
            "route_label": route_label if not is_real else "",
        })

    results.sort(key=lambda x: x["safety_score"], reverse=True)
    for rank, r in enumerate(results):
        r["rank"] = rank + 1
    return results


if __name__ == "__main__":
    for p in ["Thapar University", "Omaxe Mall", "baradari gardens", "bus stand", "Model Town"]:
        lat, lon, d = geocode_place(p)
        print(f"OK: '{p}' -> ({lat:.4f}, {lon:.4f}) : {d}")
