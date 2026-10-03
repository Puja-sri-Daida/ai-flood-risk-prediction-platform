import math
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


# =========================================================
# 1. FIND LOCATION
# =========================================================

def get_location(place):

    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": place,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    if "results" not in data or not data["results"]:
        raise ValueError("Location not found.")

    result = data["results"][0]

    return {
        "name": result["name"],
        "country": result.get("country", ""),
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "elevation": result.get("elevation")
    }


# =========================================================
# 2. GET WEATHER + ENVIRONMENT DATA
# =========================================================

def get_environmental_data(latitude, longitude):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "soil_moisture_0_to_1cm"
        ],

        "hourly": "precipitation",

        "past_days": 3,
        "forecast_days": 1,

        "timezone": "auto"
    }

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    current = data["current"]

    temperature = current["temperature_2m"]
    humidity = current["relative_humidity_2m"]

    soil_fraction = current["soil_moisture_0_to_1cm"]

    if soil_fraction is None:
        soil_moisture_pct = 0
    else:
        soil_moisture_pct = soil_fraction * 100

    # -----------------------------------------
    # RAINFALL
    # Use only completed hours up to current time
    # Do NOT include future forecast hours
    # -----------------------------------------

    hourly_times = data["hourly"]["time"]
    rainfall_values = data["hourly"]["precipitation"]

    current_time = current["time"]

    completed_rainfall = []

    for time_value, rainfall_value in zip(
        hourly_times,
        rainfall_values
    ):
        if time_value <= current_time:

            if rainfall_value is not None:
                completed_rainfall.append(rainfall_value)

    rainfall_24h = sum(completed_rainfall[-24:])
    rainfall_3day = sum(completed_rainfall[-72:])

    elevation = data.get("elevation")

    return {
        "rainfall_24h_mm": rainfall_24h,
        "rainfall_3day_mm": rainfall_3day,
        "humidity_pct": humidity,
        "temperature_c": temperature,
        "elevation_m": elevation,
        "soil_moisture_pct": soil_moisture_pct
    }

# =========================================================
# 3. CALCULATE SLOPE
# =========================================================

def get_slope(latitude, longitude):

    url = "https://api.open-meteo.com/v1/elevation"

    offset = 0.001

    latitudes = [
        latitude,
        latitude + offset,
        latitude - offset
    ]

    longitudes = [
        longitude,
        longitude,
        longitude
    ]

    params = {
        "latitude": ",".join(
            map(str, latitudes)
        ),
        "longitude": ",".join(
            map(str, longitudes)
        )
    }

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    elevations = data["elevation"]

    north = elevations[1]
    south = elevations[2]

    horizontal_distance = 222

    elevation_difference = north - south

    slope_deg = math.degrees(
        math.atan(
            abs(elevation_difference)
            / horizontal_distance
        )
    )

    return slope_deg


# =========================================================
# 4. FIND UTM ZONE
# =========================================================

def get_utm_epsg(latitude, longitude):

    utm_zone = int(
        (longitude + 180) / 6
    ) + 1

    if latitude >= 0:
        return 32600 + utm_zone

    return 32700 + utm_zone


# =========================================================
# 5. FIND NEAREST RIVER
# =========================================================

def get_river_distance(latitude, longitude):

    river_file = (
        "hydrorivers/"
        "HydroRIVERS_v10_as_shp/"
        "HydroRIVERS_v10_as.shp"
    )

    print()
    print("Loading nearby river data...")

    # Small geographic area around user
    buffer = 0.5

    bbox = (
        longitude - buffer,
        latitude - buffer,
        longitude + buffer,
        latitude + buffer
    )

    rivers = gpd.read_file(
        river_file,
        bbox=bbox
    )

    if rivers.empty:

        print("No rivers found in nearby area.")

        return None

    print(
        "Nearby river features:",
        len(rivers)
    )

    location = gpd.GeoDataFrame(
        {"name": ["User Location"]},
        geometry=[
            Point(longitude, latitude)
        ],
        crs="EPSG:4326"
    )

    epsg = get_utm_epsg(
        latitude,
        longitude
    )

    rivers_projected = rivers.to_crs(
        epsg=epsg
    )

    location_projected = location.to_crs(
        epsg=epsg
    )

    distances = rivers_projected.geometry.distance(
        location_projected.geometry.iloc[0]
    )

    distances = distances.dropna()

    if distances.empty:
        return None

    nearest_distance_m = distances.min()

    nearest_distance_km = (
        nearest_distance_m / 1000
    )

    return nearest_distance_km


# =========================================================
# 6. HISTORICAL FLOOD INDEX
# =========================================================

def get_historical_flood_index(
    latitude,
    longitude
):

    print()
    print("Loading historical flood data...")

    flood_file = "global_flood_records.gpkg"

    floods = gpd.read_file(
        flood_file
    )

    floods[
        "FloodImpactIndex"
    ] = pd.to_numeric(
        floods["FloodImpactIndex"],
        errors="coerce"
    )

    floods = floods[
        floods.geometry.notna()
    ]

    floods = floods[
        floods.geometry.is_valid
    ]

    location = gpd.GeoSeries(
        [
            Point(
                longitude,
                latitude
            )
        ],
        crs="EPSG:4326"
    )

    epsg = get_utm_epsg(
        latitude,
        longitude
    )

    floods_projected = floods.to_crs(
        epsg=epsg
    )

    location_projected = location.to_crs(
        epsg=epsg
    )

    distances = floods_projected.geometry.distance(
        location_projected.iloc[0]
    )

    floods["distance_km"] = (
        distances / 1000
    )

    nearby = floods[
        floods["distance_km"] <= 50
    ]

    print(
        "Historical floods within 50 km:",
        len(nearby)
    )

    if nearby.empty:

        return 0.0

    historical_index = (
        nearby["FloodImpactIndex"]
        .mean()
    )

    historical_index = historical_index / 8.49
    historical_index = min(max(historical_index, 0.0), 1.0)

    return historical_index


# =========================================================
# 7. MAIN PROGRAM
# =========================================================

place = input(
    "Enter your area/city: "
)

try:

    # -----------------------------------------------------
    # LOCATION
    # -----------------------------------------------------

    location = get_location(place)

    latitude = location["latitude"]
    longitude = location["longitude"]

    print()
    print("===================================")
    print("LOCATION FOUND")
    print("===================================")

    print(
        "Area:",
        location["name"]
    )

    print(
        "Country:",
        location["country"]
    )

    print(
        "Latitude:",
        latitude
    )

    print(
        "Longitude:",
        longitude
    )


    # -----------------------------------------------------
    # WEATHER
    # -----------------------------------------------------

    environment = get_environmental_data(
        latitude,
        longitude
    )


    # -----------------------------------------------------
    # SLOPE
    # -----------------------------------------------------

    slope = get_slope(
        latitude,
        longitude
    )


    # -----------------------------------------------------
    # RIVER DISTANCE
    # -----------------------------------------------------

    river_distance = get_river_distance(
        latitude,
        longitude
    )

    if river_distance is None:
        river_distance = 999.0


    # -----------------------------------------------------
    # HISTORICAL FLOOD
    # -----------------------------------------------------

    historical_index = (
        get_historical_flood_index(
            latitude,
            longitude
        )
    )


    # -----------------------------------------------------
    # FINAL 9 MODEL FEATURES
    # -----------------------------------------------------

    features = {

        "rainfall_24h_mm":
            environment["rainfall_24h_mm"],

        "rainfall_3day_mm":
            environment["rainfall_3day_mm"],

        "humidity_pct":
            environment["humidity_pct"],

        "temperature_c":
            environment["temperature_c"],

        "elevation_m":
            environment["elevation_m"],

        "slope_deg":
            slope,

        "river_distance_km":
            river_distance,

        "soil_moisture_pct":
            environment["soil_moisture_pct"],

        "historical_flood_index":
            historical_index
    }


    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    print()
    print("===================================")
    print("AUTOMATIC MODEL INPUTS")
    print("===================================")

    for feature, value in features.items():

        if isinstance(value, float):
            print(
                feature,
                ":",
                round(value, 2)
            )
        else:
            print(
                feature,
                ":",
                value
            )


    print()
    print("===================================")
    print("ALL 9 FEATURES COLLECTED")
    print("===================================")
    # =========================================================
    # 7. RUN FLOOD PREDICTION
    # =========================================================

    import joblib
    import pandas as pd

    print()
    print("=" * 45)
    print("RUNNING FLOOD PREDICTION...")
    print("=" * 45)

    model_data = joblib.load("models/flood_model.joblib")

    model = model_data["model"]
    model_features = model_data["features"]

    input_values = [
        features[name]
        for name in model_features
    ]

    input_data = pd.DataFrame(
        [input_values],
        columns=model_features
    )

    prediction = model.predict(input_data)[0]

    print()
    print("=" * 45)
    print("FLOOD PREDICTION RESULT")
    print("=" * 45)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]

        for class_value, probability in zip(
            model.classes_,
            probabilities
        ):
            print(
                "Class",
                class_value,
                ":",
                round(probability * 100, 2),
                "%"
            )

    print()

    if prediction == 1:
        print("FLOOD RISK: HIGH")
    else:
        print("FLOOD RISK: LOW")
except Exception as error:

    print()
    print("===================================")
    print("ERROR")
    print("===================================")

    print(type(error).__name__)
    print(error)
