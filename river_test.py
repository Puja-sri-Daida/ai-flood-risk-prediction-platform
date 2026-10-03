import requests
import math

latitude = 17.38405
longitude = 78.45636

# Search rivers within 20 km
radius = 5000

query = f"""
[out:json][timeout:30];
way(around:{radius},{latitude},{longitude})["waterway"="river"];
out geom;
"""

url = "https://overpass.private.coffee/api/interpreter"

headers = {
    "User-Agent": "AI-Disaster-Flood-Platform/1.0",
    "Referer": "https://www.openstreetmap.org/"
}

response = requests.post(
    url,
    data={"data": query},
    headers=headers,
    timeout=90
)

print("Status:", response.status_code)

if response.status_code != 200:
    print("Server response:")
    print(response.text[:1000])
    raise SystemExit

data = response.json()

rivers = data.get("elements", [])

print("Rivers found:", len(rivers))


def haversine(lat1, lon1, lat2, lon2):

    R = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c


nearest_distance = None
nearest_river = None


for river in rivers:

    river_name = river.get(
        "tags", {}
    ).get(
        "name",
        "Unnamed River"
    )

    geometry = river.get(
        "geometry",
        []
    )

    for point in geometry:

        river_lat = point["lat"]
        river_lon = point["lon"]

        distance = haversine(
            latitude,
            longitude,
            river_lat,
            river_lon
        )

        if (
            nearest_distance is None
            or distance < nearest_distance
        ):

            nearest_distance = distance
            nearest_river = river_name


print()

if nearest_distance is not None:

    print(
        "NEAREST RIVER:",
        nearest_river
    )

    print(
        "RIVER DISTANCE:",
        round(nearest_distance, 3),
        "km"
    )

else:

    print(
        "No river found within 20 km."
    )
