import requests
import math

latitude = 17.38405
longitude = 78.45636

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
    "latitude": ",".join(map(str, latitudes)),
    "longitude": ",".join(map(str, longitudes))
}

response = requests.get(url, params=params)

print("Status:", response.status_code)

data = response.json()

elevations = data["elevation"]

center = elevations[0]
north = elevations[1]
south = elevations[2]

horizontal_distance = 222

elevation_difference = north - south

slope_percent = (elevation_difference / horizontal_distance) * 100

slope_deg = math.degrees(
    math.atan(abs(elevation_difference) / horizontal_distance)
)

print()
print("CENTER ELEVATION:", center, "m")
print("NORTH ELEVATION:", north, "m")
print("SOUTH ELEVATION:", south, "m")
print("SLOPE:", round(slope_deg, 2), "degrees")
