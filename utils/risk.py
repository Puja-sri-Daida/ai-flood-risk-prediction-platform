import numpy as np

def classify_risk(prob):
    if prob < 0.25:
        return "LOW", "🟢", "Normal monitoring is sufficient."
    if prob < 0.50:
        return "MODERATE", "🟡", "Monitor rainfall and local advisories."
    if prob < 0.75:
        return "HIGH", "🟠", "Prepare for possible flooding and monitor official alerts."
    return "CRITICAL", "🔴", "High-risk conditions detected. Follow official emergency guidance."

def local_reasoning(row):
    reasons = []
    if row["rainfall_24h_mm"] >= 100: reasons.append("heavy 24-hour rainfall")
    if row["rainfall_3day_mm"] >= 250: reasons.append("high cumulative 3-day rainfall")
    if row["humidity_pct"] >= 85: reasons.append("high humidity")
    if row["elevation_m"] <= 30: reasons.append("low elevation")
    if row["river_distance_km"] <= 2: reasons.append("close proximity to a water body")
    if row["soil_moisture_pct"] >= 75: reasons.append("high soil moisture")
    if row["historical_flood_index"] >= .65: reasons.append("high historical flood-risk index")
    return reasons or ["no single dominant risk factor"]

def score_for_map(lat, lon, base_prob, center_lat, center_lon):
    d = ((lat-center_lat)**2 + (lon-center_lon)**2)**0.5
    return float(np.clip(base_prob * np.exp(-d*2.8) + 0.10*(1-np.exp(-d*2.8)), 0.02, 0.98))
