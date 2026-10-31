"""
Safest Route Planner for Pehredar
Uses the public OSRM server for routing, Nominatim for geocoding,
and the H3 forecast grid for per-hex air quality scoring along routes.

Geocoding is resilient in any mode (DEMO or live):
  1. Check disk cache first (always)
  2. Match against offline gazetteer (LOCALITIES from grid.py + extra POIs)
  3. Try Nominatim with query variants (even in DEMO_MODE, it's free)
  4. Return fallback coords and signal caller to show dropdown

Routing is resilient:
  1. Check disk cache first
  2. Try OSRM public server
  3. Fall back to a straight-line route sampled every 100 m, clearly labelled

Author: Team Rooted – Pehredar, RootAccess Hackathon
"""

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h3
import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Nominatim search endpoint (free, no key required)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# OSRM public server (free, no key required)
OSRM_BASE = "http://router.project-osrm.org/route/v1"

# OSRM mode mapping
OSRM_PROFILES: Dict[str, str] = {
    "walk": "foot",
    "bike": "bike",
    "car": "driving",
}

# Car exposure factor (car counts half the exposure of walk/bike)
CAR_EXPOSURE_FACTOR = 0.5

# Safety column mapping by health profile
SAFETY_COLUMNS: Dict[str, str] = {
    "healthy": "safety_healthy",
    "vulnerable": "safety_vulnerable",
    "respiratory": "safety_respiratory",
}

_GEOCODE_CACHE_PATH = DATA_DIR / "geocode_cache.json"

# ---------------------------------------------------------------------------
# Offline gazetteer — seeded from LOCALITIES in grid.py plus extra POIs.
# Keys are canonical names; aliases map onto canonical names for fuzzy match.
# ---------------------------------------------------------------------------

# Imported lazily to avoid circular imports at module load time
_GAZETTEER: Optional[Dict[str, Tuple[float, float]]] = None

# Extra POIs beyond the 12 LOCALITIES (coords from Nominatim, cached manually)
_EXTRA_POIS: Dict[str, Tuple[float, float]] = {
    "Omaxe Mall": (30.3497, 76.3964),          # Omaxe Mall, Patiala
    "Thapar Institute": (30.3551, 76.3648),     # alias for Thapar University
    "TIET": (30.3551, 76.3648),                 # alias
    "NIS Patiala": (30.3470, 76.3975),          # National Institute of Sports
    "Qila Mubarak": (30.3360, 76.3940),         # Qila Mubarak palace
    "Punjabi University": (30.2338, 76.4129),   # outskirts
    "Rajindra Hospital": (30.3389, 76.3972),
    "Old Sabzi Mandi": (30.3420, 76.4060),
}

# Fuzzy alias table: lower-case search term → canonical gazetteer name
_ALIASES: Dict[str, str] = {
    # Baradari
    "baradari": "Baradari",
    "baradari gardens": "Baradari",
    "baradari garden": "Baradari",
    "bara dari": "Baradari",
    # Thapar
    "thapar": "Thapar University",
    "thapar university": "Thapar University",
    "thapar institute": "Thapar University",
    "tiet": "Thapar University",
    "tiet patiala": "Thapar University",
    # Bus Stand
    "bus stand": "Patiala Bus Stand",
    "patiala bus stand": "Patiala Bus Stand",
    "old bus stand": "Patiala Bus Stand",
    # Leela Bhawan
    "leela bhawan": "Leela Bhawan",
    "leela bawan": "Leela Bhawan",
    "leela": "Leela Bhawan",
    # Model Town
    "model town": "Model Town",
    "model town patiala": "Model Town",
    # Tripuri
    "tripuri": "Tripuri",
    "tripuri patiala": "Tripuri",
    # Urban Estate
    "urban estate": "Urban Estate",
    "urban estate patiala": "Urban Estate",
    # Bhupindra Road
    "bhupindra road": "Bhupindra Road",
    "bhupindra": "Bhupindra Road",
    # Sanauri Adda
    "sanauri adda": "Sanauri Adda",
    "sanauri": "Sanauri Adda",
    # Rajpura Road
    "rajpura road": "Rajpura Road",
    "rajpura": "Rajpura Road",
    # Gurbax Colony
    "gurbax colony": "Gurbax Colony",
    "gurbax": "Gurbax Colony",
    # Sirhind Road
    "sirhind road": "Sirhind Road",
    "sirhind": "Sirhind Road",
    # Omaxe Mall
    "omaxe": "Omaxe Mall",
    "omaxe mall": "Omaxe Mall",
    "omaxe patiala": "Omaxe Mall",
    # NIS
    "nis": "NIS Patiala",
    "nis patiala": "NIS Patiala",
    "national institute of sports": "NIS Patiala",
    # Qila Mubarak
    "qila mubarak": "Qila Mubarak",
    "qila": "Qila Mubarak",
    "fort": "Qila Mubarak",
    # Rajindra Hospital
    "rajindra hospital": "Rajindra Hospital",
    "rajindra": "Rajindra Hospital",
    "govt hospital": "Rajindra Hospital",
}

# Nominatim query variants tried for each unresolved place
_NOMINATIM_VARIANTS = [
    "{name}",
    "{name}, Patiala",
    "{name}, Patiala, Punjab, India",
    "{name}, Punjab, India",
]


def _build_gazetteer() -> Dict[str, Tuple[float, float]]:
    """Builds the combined offline gazetteer (LOCALITIES + extra POIs), lazily."""
    global _GAZETTEER
    if _GAZETTEER is not None:
        return _GAZETTEER
    try:
        from src.grid import LOCALITIES
        combined = dict(LOCALITIES)
    except ImportError:
        combined = {}
    combined.update(_EXTRA_POIS)
    _GAZETTEER = combined
    return _GAZETTEER


def get_known_localities() -> List[str]:
    """Returns sorted list of all known locality names (for UI dropdowns)."""
    gz = _build_gazetteer()
    return sorted(gz.keys())


# ---------------------------------------------------------------------------
# Geocode cache helpers
# ---------------------------------------------------------------------------

def _load_geocode_cache() -> Dict[str, Any]:
    """Loads geocode cache from disk."""
    if _GEOCODE_CACHE_PATH.exists():
        try:
            with open(_GEOCODE_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_geocode_cache(cache: Dict[str, Any]) -> None:
    """Persists geocode cache to disk."""
    try:
        with open(_GEOCODE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def _nominatim_lookup(query: str, timeout: int = 8) -> Optional[Tuple[float, float, str]]:
    """
    Makes a single Nominatim request.
    Returns (lat, lon, display_name) or None on failure/no results.
    """
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "Pehredar/1.0 (RootAccess Hackathon; contact=team@pehredar.in)"},
            timeout=timeout,
        )
        resp.raise_for_status()
        results = resp.json()
        if results:
            top = results[0]
            return float(top["lat"]), float(top["lon"]), top.get("display_name", query)
    except Exception:
        pass
    return None


def geocode_place(place_name: str) -> Tuple[float, float, str]:
    """
    Resolves a place name to (lat, lon, display_name) using a 3-tier strategy:

    1. Disk cache — always checked first regardless of DEMO_MODE.
    2. Offline gazetteer — LOCALITIES + extra POIs, fuzzy alias matching.
    3. Nominatim — tried with up to 4 query variants; result cached on success.

    Raises:
        GeocodeFallbackNeeded: Custom exception with `known_localities` attribute
            when all tiers fail, so the caller can show a dropdown instead of
            an error page.
    """
    cache_key = place_name.lower().strip()
    cache = _load_geocode_cache()

    # --- Tier 1: disk cache ---
    if cache_key in cache:
        c = cache[cache_key]
        return float(c["lat"]), float(c["lon"]), c["display_name"]

    # --- Tier 2: offline gazetteer (exact alias match, then substring match) ---
    gz = _build_gazetteer()
    # Exact alias lookup (case-insensitive)
    canonical = _ALIASES.get(cache_key)
    if canonical and canonical in gz:
        lat, lon = gz[canonical]
        display_name = f"{canonical}, Patiala"
        cache[cache_key] = {"lat": lat, "lon": lon, "display_name": display_name}
        _save_geocode_cache(cache)
        return lat, lon, display_name

    # Direct name match in gazetteer (case-insensitive)
    for gz_name, (lat, lon) in gz.items():
        if gz_name.lower() == cache_key:
            display_name = f"{gz_name}, Patiala"
            cache[cache_key] = {"lat": lat, "lon": lon, "display_name": display_name}
            _save_geocode_cache(cache)
            return lat, lon, display_name

    # Substring match: the query contains the canonical name or vice-versa
    for gz_name, (lat, lon) in gz.items():
        if gz_name.lower() in cache_key or cache_key in gz_name.lower():
            display_name = f"{gz_name}, Patiala"
            cache[cache_key] = {"lat": lat, "lon": lon, "display_name": display_name}
            _save_geocode_cache(cache)
            return lat, lon, display_name

    # Alias substring match
    for alias, canonical in _ALIASES.items():
        if alias in cache_key or cache_key in alias:
            if canonical in gz:
                lat, lon = gz[canonical]
                display_name = f"{canonical}, Patiala"
                cache[cache_key] = {"lat": lat, "lon": lon, "display_name": display_name}
                _save_geocode_cache(cache)
                return lat, lon, display_name

    # --- Tier 3: Nominatim (free, tried even in DEMO_MODE) ---
    bare_name = place_name.strip()
    for variant_template in _NOMINATIM_VARIANTS:
        query = variant_template.format(name=bare_name)
        result = _nominatim_lookup(query)
        if result:
            lat, lon, display_name = result
            cache[cache_key] = {"lat": lat, "lon": lon, "display_name": display_name}
            _save_geocode_cache(cache)
            return lat, lon, display_name
        time.sleep(0.3)  # Nominatim rate-limit courtesy delay

    # --- All tiers failed ---
    exc = GeocodeFallbackNeeded(
        f"Could not find '{place_name}'. Please pick a locality from the dropdown.",
        known_localities=get_known_localities(),
    )
    raise exc


class GeocodeFallbackNeeded(Exception):
    """
    Raised when geocoding fails all tiers.
    Carries `known_localities` so the UI can show a dropdown.
    """
    def __init__(self, message: str, known_localities: List[str]):
        super().__init__(message)
        self.known_localities = known_localities


# ---------------------------------------------------------------------------
# Route cache helpers
# ---------------------------------------------------------------------------

def _route_cache_path(
    origin_lat: float, origin_lon: float,
    dest_lat: float, dest_lon: float,
    mode: str,
) -> Path:
    """Returns a unique cache file path for an OSRM route query."""
    key = f"{origin_lat:.5f},{origin_lon:.5f},{dest_lat:.5f},{dest_lon:.5f},{mode}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return DATA_DIR / f"route_cache_{h}.json"


def _straight_line_route(
    origin_lat: float, origin_lon: float,
    dest_lat: float, dest_lon: float,
    mode: str,
) -> List[Dict[str, Any]]:
    """
    Constructs a synthetic straight-line route when OSRM is unavailable.
    Samples 20 intermediate points, estimates duration from typical speeds,
    and tags the route as approximate.

    Speed estimates (km/h): walk=5, bike=15, car=40.
    """
    speeds = {"walk": 5.0, "bike": 15.0, "car": 40.0}
    speed_kmh = speeds.get(mode, 5.0)

    dist_m = _haversine_m(origin_lat, origin_lon, dest_lat, dest_lon)
    duration_sec = (dist_m / 1000.0 / speed_kmh) * 3600.0

    # Sample 20 evenly-spaced intermediate points
    n_pts = max(2, min(20, int(dist_m / 100) + 2))
    coords = []
    for i in range(n_pts):
        frac = i / (n_pts - 1)
        lat = origin_lat + frac * (dest_lat - origin_lat)
        lon = origin_lon + frac * (dest_lon - origin_lon)
        coords.append([lon, lat])  # GeoJSON [lon, lat]

    return [{
        "geometry": {"type": "LineString", "coordinates": coords},
        "duration": duration_sec,
        "distance": dist_m,
        "_pehredar_label": "approximate route (offline)",
    }]


def fetch_osrm_routes(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    mode: str = "walk",
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Fetches up to 3 routes. Returns (routes, is_real) where is_real=False
    means we fell back to a straight-line approximation.

    Strategy:
      1. Disk cache (always)
      2. OSRM public server
      3. Straight-line fallback (always succeeds)
    """
    cache_path = _route_cache_path(origin_lat, origin_lon, dest_lat, dest_lon, mode)

    # --- Cache first ---
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data.get("routes", []), data.get("is_real", True)
            # Legacy format (list of routes)
            return data, True
        except Exception:
            pass

    # --- OSRM ---
    profile = OSRM_PROFILES.get(mode, "foot")
    url = (
        f"{OSRM_BASE}/{profile}/"
        f"{origin_lon:.6f},{origin_lat:.6f};"
        f"{dest_lon:.6f},{dest_lat:.6f}"
        f"?alternatives=true&overview=full&geometries=geojson"
    )

    try:
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == "Ok" and data.get("routes"):
            routes = data["routes"]
            _cache_routes(cache_path, routes, is_real=True)
            return routes, True
    except Exception:
        pass

    # --- Straight-line fallback ---
    routes = _straight_line_route(origin_lat, origin_lon, dest_lat, dest_lon, mode)
    # Do NOT cache straight-line; try OSRM again next time
    return routes, False


def _cache_routes(cache_path: Path, routes: List[Dict], is_real: bool) -> None:
    """Persists routes to disk cache."""
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"routes": routes, "is_real": is_real}, f)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in metres between two lat/lon points."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def sample_route_points(
    coords: List[List[float]],
    interval_m: float = 100.0,
) -> List[Tuple[float, float]]:
    """
    Interpolates sample points every `interval_m` metres along a GeoJSON polyline.

    Args:
        coords: List of [lon, lat] pairs from GeoJSON geometry.
        interval_m: Sampling interval in metres (default 100 m).

    Returns:
        List of (lat, lon) tuples sampled along the route.
    """
    if not coords:
        return []

    sampled: List[Tuple[float, float]] = []
    sampled.append((coords[0][1], coords[0][0]))
    accumulated = 0.0

    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        seg_dist = _haversine_m(lat1, lon1, lat2, lon2)
        if seg_dist == 0:
            continue

        remaining_to_next = interval_m - accumulated

        if accumulated + seg_dist >= interval_m:
            d = remaining_to_next
            while d <= seg_dist:
                frac = d / seg_dist
                s_lat = lat1 + frac * (lat2 - lat1)
                s_lon = lon1 + frac * (lon2 - lon1)
                sampled.append((s_lat, s_lon))
                d += interval_m
            accumulated = seg_dist - (d - interval_m)
        else:
            accumulated += seg_dist

    last_lon, last_lat = coords[-1]
    if sampled[-1] != (last_lat, last_lon):
        sampled.append((last_lat, last_lon))

    return sampled


# ---------------------------------------------------------------------------
# Forecast lookup
# ---------------------------------------------------------------------------

def get_hex_forecast_for_time(
    lat: float,
    lon: float,
    target_time: pd.Timestamp,
    df_forecast: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Given a lat/lon and a target timestamp, finds the H3 cell (res 8),
    then returns the forecast row in df_forecast closest to target_time.

    Falls back to citywide average if the hex is not in the forecast grid.
    """
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
            closest_idx = diffs.idxmin()
            row = subset.loc[closest_idx]
        except Exception:
            row = subset.iloc[0]
    else:
        now = pd.Timestamp.now()
        hours_ahead = max(1, int((pd.to_datetime(target_time) - now).total_seconds() / 3600))
        hours_ahead = min(72, max(1, hours_ahead))
        h_sub = subset[subset["horizon_hours"] == hours_ahead]
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


# ---------------------------------------------------------------------------
# Route scoring
# ---------------------------------------------------------------------------

def score_route(
    route_coords: List[List[float]],
    duration_sec: float,
    departure_time: pd.Timestamp,
    df_forecast: pd.DataFrame,
    health_profile: str = "healthy",
    mode: str = "walk",
) -> Dict[str, Any]:
    """
    Computes a time-weighted average Safety Index along the sampled route.

    Car counts half the exposure of walk/bike.
    """
    safety_col = SAFETY_COLUMNS.get(health_profile, "safety_healthy")
    is_car = mode == "car"

    sampled_points = sample_route_points(route_coords, interval_m=100.0)
    n = len(sampled_points)
    if n == 0:
        return {
            "safety_score": 5.0,
            "worst_locality": "Unknown",
            "worst_aqi": 150,
            "worst_pm25": 75.0,
            "worst_safety": 5.0,
            "points": [],
        }

    total_dist_m = 0.0
    seg_dists = [0.0]
    for i in range(1, n):
        d = _haversine_m(*sampled_points[i - 1], *sampled_points[i])
        total_dist_m += d
        seg_dists.append(total_dist_m)

    scored_points = []
    worst_safety = 10.0
    worst_locality = "Unknown"
    worst_aqi = 0
    worst_pm25 = 0.0

    for i, (lat, lon) in enumerate(sampled_points):
        frac = seg_dists[i] / total_dist_m if total_dist_m > 0 else 0.0
        elapsed_sec = frac * duration_sec
        point_time = departure_time + pd.Timedelta(seconds=elapsed_sec)

        fc = get_hex_forecast_for_time(lat, lon, point_time, df_forecast)
        raw_safety = fc.get(safety_col, 5.0)

        if is_car:
            safety_loss = 10.0 - raw_safety
            adjusted_safety = min(10.0, 10.0 - safety_loss * CAR_EXPOSURE_FACTOR)
        else:
            adjusted_safety = raw_safety

        scored_points.append({
            "lat": lat,
            "lon": lon,
            "safety": adjusted_safety,
            "aqi": fc["aqi"],
            "pm2_5": fc["pm2_5"],
            "locality": fc["locality"],
            "elapsed_min": round(elapsed_sec / 60, 1),
        })

        if adjusted_safety < worst_safety:
            worst_safety = adjusted_safety
            worst_locality = fc["locality"]
            worst_aqi = fc["aqi"]
            worst_pm25 = fc["pm2_5"]

    avg_safety = float(np.mean([p["safety"] for p in scored_points]))

    return {
        "safety_score": round(avg_safety, 1),
        "worst_locality": worst_locality,
        "worst_aqi": worst_aqi,
        "worst_pm25": round(worst_pm25, 1),
        "worst_safety": round(worst_safety, 1),
        "points": scored_points,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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

    Geocodes both places (resilient: cache → gazetteer → Nominatim),
    fetches up to 3 OSRM routes (falling back to straight-line), scores
    each, and returns them sorted safest-first.

    Each result dict contains:
        route_idx, rank, safety_score, duration_min, distance_km,
        worst_locality, worst_aqi, worst_pm25, worst_safety,
        geometry_coords, scored_points,
        origin_latlon, dest_latlon, origin_name, dest_name,
        is_real_route (bool — False means straight-line fallback)

    Raises:
        GeocodeFallbackNeeded: If geocoding fails completely.
    """
    departure_time = pd.Timestamp.now() + pd.Timedelta(hours=departure_offset_hours)

    origin_lat, origin_lon, origin_name = geocode_place(from_place)
    dest_lat, dest_lon, dest_name = geocode_place(to_place)

    osrm_routes, is_real = fetch_osrm_routes(origin_lat, origin_lon, dest_lat, dest_lon, mode)

    results = []
    for idx, route in enumerate(osrm_routes):
        duration_sec = float(route.get("duration", 0))
        distance_m = float(route.get("distance", 0))
        coords = route.get("geometry", {}).get("coordinates", [])
        route_label = route.get("_pehredar_label", "")

        score_info = score_route(
            route_coords=coords,
            duration_sec=duration_sec,
            departure_time=departure_time,
            df_forecast=df_forecast,
            health_profile=health_profile,
            mode=mode,
        )

        results.append({
            "route_idx": idx + 1,
            "safety_score": score_info["safety_score"],
            "duration_min": round(duration_sec / 60, 1),
            "distance_km": round(distance_m / 1000, 2),
            "worst_locality": score_info["worst_locality"],
            "worst_aqi": score_info["worst_aqi"],
            "worst_pm25": score_info["worst_pm25"],
            "worst_safety": score_info["worst_safety"],
            "geometry_coords": coords,
            "scored_points": score_info["points"],
            "origin_latlon": (origin_lat, origin_lon),
            "dest_latlon": (dest_lat, dest_lon),
            "origin_name": origin_name,
            "dest_name": dest_name,
            "is_real_route": is_real,
            "route_label": route_label,
        })

    results.sort(key=lambda r: r["safety_score"], reverse=True)
    for rank, r in enumerate(results):
        r["rank"] = rank + 1

    return results


if __name__ == "__main__":
    print("Testing geocoder...")
    for name in ["Baradari", "bus stand", "Baradari Gardens", "thapar", "Omaxe Mall"]:
        try:
            lat, lon, display = geocode_place(name)
            print(f"  ✓ '{name}' → {lat:.4f}, {lon:.4f} ({display[:50]})")
        except GeocodeFallbackNeeded as e:
            print(f"  ✗ '{name}' → fallback needed: {e}")
        except Exception as e:
            print(f"  ✗ '{name}' → {e}")
