"""
AQI and Environmental Safety Index Module for Pehredar
Implements the Indian National Air Quality Index (NAQI) breakpoints from PM2.5 (ug/m3)
and computes demographic-specific Safety Indices (1 to 10 scale).
"""

from typing import Dict, Tuple, Union
import numpy as np

# Indian National AQI Breakpoints for PM2.5 (ug/m3 -> AQI)
# Format: (C_low, C_high, I_low, I_high, category, hex_color)
NAQI_BREAKPOINTS = [
    (0.0, 30.0, 0, 50, "Good", "#2e7d32"),
    (30.1, 60.0, 51, 100, "Satisfactory", "#7cb342"),
    (60.1, 90.0, 101, 200, "Moderate", "#fbc02d"),
    (90.1, 120.0, 201, 300, "Poor", "#f57c00"),
    (120.1, 250.0, 301, 400, "Very Poor", "#d32f2f"),
    (250.1, 500.0, 401, 500, "Severe", "#4a148c"),
]

# Vulnerability Factor Multipliers
DEMOGRAPHIC_FACTORS: Dict[str, float] = {
    "healthy": 1.0,       # Healthy Adult
    "vulnerable": 1.25,   # Children / Elderly / Pregnant
    "respiratory": 1.5,   # Asthma / Heart / Lung Condition
}

DEMOGRAPHIC_LABELS: Dict[str, str] = {
    "healthy": "Healthy Adult (factor 1.0)",
    "vulnerable": "Children / Elderly / Pregnant (factor 1.25)",
    "respiratory": "Asthma / Heart / Lung Condition (factor 1.5)",
}


def calculate_indian_aqi(pm25: float) -> Tuple[int, str, str]:
    """
    Computes Indian National AQI from PM2.5 (ug/m3) using official linear piecewise formula.
    Returns: (aqi_value, category, hex_color)
    """
    pm25 = max(0.0, float(pm25))

    for c_low, c_high, i_low, i_high, category, color in NAQI_BREAKPOINTS:
        if c_low <= pm25 <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
            return int(round(aqi)), category, color

    if pm25 > 500.0:
        return 500, "Severe (Extreme)", "#311b92"

    return 50, "Satisfactory", "#7cb342"


def calculate_safety_index(
    aqi: float,
    demographic: str = "healthy",
    custom_factor: float = None,
) -> float:
    """
    Computes Safety Index (1 to 10, higher = safer):
      Safety Index = clip(10 - (AQI * factor) / 50, 1, 10)
    
    Factors:
      - healthy: 1.0 (Healthy Adult)
      - vulnerable: 1.25 (Children / Elderly / Pregnant)
      - respiratory: 1.5 (Asthma / Heart or Lung Condition)
    """
    if custom_factor is not None:
        factor = custom_factor
    else:
        factor = DEMOGRAPHIC_FACTORS.get(demographic.lower(), 1.0)

    val = 10.0 - (float(aqi) * factor) / 50.0
    clipped = float(np.clip(val, 1.0, 10.0))
    return round(clipped, 1)


def get_safety_summary(aqi: int) -> Dict[str, Any]:
    """
    Generates Safety Index across all 3 demographic profiles along with
    actionable health protocols.
    """
    indices = {
        "healthy": calculate_safety_index(aqi, "healthy"),
        "vulnerable": calculate_safety_index(aqi, "vulnerable"),
        "respiratory": calculate_safety_index(aqi, "respiratory"),
    }

    if aqi <= 50:
        advisory = "Clean air. Ideal for all outdoor exercises and outdoor school activities."
    elif aqi <= 100:
        advisory = "Air quality is acceptable. Extremely sensitive persons should monitor minor discomfort."
    elif aqi <= 200:
        advisory = "Breathing discomfort to asthmatics and sensitive children. Limit intense outdoor cardio."
    elif aqi <= 300:
        advisory = "Breathing discomfort to most people on prolonged exposure. Wear N95 outdoors."
    elif aqi <= 400:
        advisory = "Respiratory illness on prolonged exposure. Keep windows shut, run HEPA filtration."
    else:
        advisory = "Health emergency. Severe impact even on healthy adults. Avoid all outdoor exposure."

    return {
        "aqi": aqi,
        "indices": indices,
        "advisory": advisory,
    }


if __name__ == "__main__":
    test_pm25 = [15.0, 45.0, 75.0, 110.0, 180.0, 320.0]
    for p in test_pm25:
        aqi, cat, col = calculate_indian_aqi(p)
        s_h = calculate_safety_index(aqi, "healthy")
        s_v = calculate_safety_index(aqi, "vulnerable")
        s_r = calculate_safety_index(aqi, "respiratory")
        print(f"PM2.5: {p:5.1f} -> AQI: {aqi:3d} ({cat:12s}) | Safety: Healthy={s_h:.1f}, Vuln={s_v:.1f}, Resp={s_r:.1f}")
