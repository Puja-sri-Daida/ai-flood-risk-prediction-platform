import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

print("Loading flood dataset...")

g = gpd.read_file("global_flood_records.gpkg")

print("Dataset loaded!")
print("Total records:", len(g))

# Convert FloodImpactIndex to numeric
g["FloodImpactIndex"] = pd.to_numeric(
    g["FloodImpactIndex"],
    errors="coerce"
)

# Ask user for location
place = input("\nEnter your city / area: ")

# -----------------------------
# STEP 1: FIND LOCATION
# -----------------------------

geo_url = "https://geocoding-api.open-meteo.com/v1/search"

geo_params = {
    "name": place,
    "count": 1,
    "language": "en",
    "format": "json"
}

response = requests.get(
    geo_url,
    params=geo_params
)

data = response.json()

if "results" not in data:
    print("Location not found.")
    exit()

location = data["results"][0]

latitude = location["latitude"]
longitude = location["longitude"]

print()
print("LOCATION FOUND")
print("Area:", location.get("name"))
print("Country:", location.get("country"))
print("Latitude:", latitude)
print("Longitude:", longitude)

# -----------------------------
# STEP 2: CREATE LOCATION POINT
# -----------------------------

point = gpd.GeoSeries(
    [Point(longitude, latitude)],
    crs="EPSG:4326"
)

# -----------------------------
# STEP 3: FIND UTM ZONE
# -----------------------------

utm_zone = int((longitude + 180) / 6) + 1

if latitude >= 0:
    epsg = 32600 + utm_zone
else:
    epsg = 32700 + utm_zone

# Convert coordinates to metric system
floods = g.to_crs(epsg=epsg)
point = point.to_crs(epsg=epsg)

# -----------------------------
# STEP 4: CALCULATE DISTANCE
# -----------------------------

g["distance_km"] = (
    floods.geometry.distance(point.iloc[0]) / 1000
)

# -----------------------------
# STEP 5: FIND NEARBY FLOODS
# -----------------------------

nearby = g[g["distance_km"] <= 50]

print()
print("HISTORICAL FLOOD ANALYSIS")
print("-------------------------")

print(
    "Flood records within 50 km:",
    len(nearby)
)

if len(nearby) > 0:

    average_impact = nearby["FloodImpactIndex"].mean()
    maximum_impact = nearby["FloodImpactIndex"].max()

    print(
        "Average Flood Impact:",
        round(average_impact, 2)
    )

    print(
        "Maximum Flood Impact:",
        round(maximum_impact, 2)
    )

else:

    average_impact = 0

    print("No historical flood records found within 50 km.")

print()
print("Historical Flood Index:", round(average_impact, 2))
