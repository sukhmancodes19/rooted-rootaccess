"""
H3 Hexagonal Grid Module for Pehredar
Generates resolution 8 H3 hexagons (~0.7 km2) covering Patiala, Punjab
Bounding box: lat 30.29 - 30.40, lon 76.32 - 76.47
"""

import h3
import math
from typing import Dict, List, Tuple, Any

# Fixed list of 12 landmark localities across Patiala
LOCALITIES: Dict[str, Tuple[float, float]] = {
    "Tripuri": (30.3580, 76.3980),
    "Leela Bhawan": (30.3340, 76.3860),
    "Model Town": (30.3280, 76.3760),
    "Urban Estate": (30.3600, 76.4350),
    "Bhupindra Road": (30.3420, 76.3820),
    "Thapar University": (30.3550, 76.3650),
    "Sanauri Adda": (30.3160, 76.4020),
    "Rajpura Road": (30.3580, 76.4520),
    "Baradari": (30.3400, 76.3910),
    "Gurbax Colony": (30.3320, 76.4080),
    "Sirhind Road": (30.3700, 76.3900),
    "Patiala Bus Stand": (30.3390, 76.4050),
}

# Patiala Bounding Box coordinates
BBOX_LAT_MIN = 30.29
BBOX_LAT_MAX = 30.40
BBOX_LON_MIN = 76.32
BBOX_LON_MAX = 76.47


def get_nearest_locality(lat: float, lon: float) -> Tuple[str, float]:
    """
    Finds the closest locality from the fixed list of 12 landmarks.
    Returns (locality_name, distance_km).
    """
    best_loc = ""
    best_dist = float("inf")

    # Approx distance calculation (1 deg lat ~ 111 km, 1 deg lon ~ 96 km at lat 30.34)
    for name, (loc_lat, loc_lon) in LOCALITIES.items():
        dy = (loc_lat - lat) * 111.0
        dx = (loc_lon - lon) * (111.0 * math.cos(math.radians(30.34)))
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < best_dist:
            best_dist = dist
            best_loc = name

    return best_loc, round(best_dist, 2)


def generate_patiala_h3_grid(resolution: int = 8) -> List[Dict[str, Any]]:
    """
    Generates H3 hexagons at resolution 8 covering Patiala bounding box.
    Each hexagon is enriched with:
      - hex_id (H3 index)
      - lat, lon (centroid coordinates)
      - locality (nearest of the 12 fixed localities)
      - distance_to_locality_km
      - boundary (list of [lat, lon] tuples for polygon rendering)
      - geojson_geometry (GeoJSON Polygon dict)
    """
    # Create bounding polygon
    poly = h3.LatLngPoly([
        (BBOX_LAT_MIN, BBOX_LON_MIN),
        (BBOX_LAT_MAX, BBOX_LON_MIN),
        (BBOX_LAT_MAX, BBOX_LON_MAX),
        (BBOX_LAT_MIN, BBOX_LON_MAX),
    ])

    cell_ids = list(h3.polygon_to_cells(poly, res=resolution))

    grid = []
    for cell in sorted(cell_ids):
        lat, lon = h3.cell_to_latlng(cell)
        locality, dist_km = get_nearest_locality(lat, lon)
        boundary_coords = h3.cell_to_boundary(cell)  # tuple of (lat, lon)
        
        # Format for Folium and GeoJSON
        polygon_latlng = [[p[0], p[1]] for p in boundary_coords]
        # GeoJSON uses [lon, lat] and must be closed
        geojson_coords = [[p[1], p[0]] for p in boundary_coords]
        geojson_coords.append(geojson_coords[0])

        grid.append({
            "hex_id": cell,
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "locality": locality,
            "display_name": f"{locality} ({cell[-4:]})",
            "distance_to_locality_km": dist_km,
            "boundary": polygon_latlng,
            "geojson": {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [geojson_coords],
                },
                "properties": {
                    "hex_id": cell,
                    "locality": locality,
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                },
            },
        })

    return grid


if __name__ == "__main__":
    grid = generate_patiala_h3_grid()
    print(f"Generated {len(grid)} H3 cells for Patiala.")
    print("Sample cell:", grid[0]["display_name"], grid[0]["lat"], grid[0]["lon"])
