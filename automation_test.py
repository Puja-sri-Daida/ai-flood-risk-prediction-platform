import requests


# -------------------------------------------------
# 1. FIND LOCATION
# -------------------------------------------------

def get_location(place):
    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": place,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()

    if "results" not in data or not data["results"]:
        raise ValueError("Location not found.")

    result = data["results"][0]

    return {
        "name": result["name"],
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "country": result.get("country", "")
    }


# -------------------------------------------------
# 2. GET ENVIRONMENTAL DATA
# -------------------------------------------------

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

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()

    # Current weather
    current = data["current"]

    temperature = current["temperature_2m"]
    humidity = current["relative_humidity_2m"]

    # Soil moisture is returned as fraction
    soil_moisture_fraction = current["soil_moisture_0_to_1cm"]
    soil_moisture_pct = soil_moisture_fraction * 100

    # Rainfall
    rainfall_values = data["hourly"]["precipitation"]

    rainfall_24h = sum(rainfall_values[-24:])
    rainfall_3day = sum(rainfall_values[-72:])

    # Elevation
    elevation = data.get("elevation")

    return {
        "rainfall_24h_mm": rainfall_24h,
        "rainfall_3day_mm": rainfall_3day,
        "humidity_pct": humidity,
        "temperature_c": temperature,
        "elevation_m": elevation,
        "soil_moisture_pct": soil_moisture_pct
    }


# -------------------------------------------------
# 3. MAIN PROGRAM
# -------------------------------------------------

place = input("Enter your area/city: ")

try:

    location = get_location(place)

    print("\n📍 LOCATION FOUND")
    print("Area:", location["name"])
    print("Country:", location["country"])
    print("Latitude:", location["latitude"])
    print("Longitude:", location["longitude"])

    environment = get_environmental_data(
        location["latitude"],
        location["longitude"]
    )

    print("\n🌦️ AUTOMATIC ENVIRONMENT DATA")

    for feature, value in environment.items():
        print(feature, ":", value)

except Exception as error:

    print("\n❌ ERROR:")
    print(error)
