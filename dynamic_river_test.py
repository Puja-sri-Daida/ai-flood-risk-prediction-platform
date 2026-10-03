import geopandas as gpd
from shapely.geometry import Point

# Test location only
# Final app lo these coordinates API nunchi automatic ga vastayi
latitude = 17.38405
longitude = 78.45636

river_file = "hydrorivers/HydroRIVERS_v10_as_shp/HydroRIVERS_v10_as.shp"

print("Loading river dataset...")

rivers = gpd.read_file(river_file)

print("River dataset loaded.")
print("Total river features:", len(rivers))

# Create user location
location = gpd.GeoDataFrame(
    {"name": ["User Location"]},
    geometry=[Point(longitude, latitude)],
    crs="EPSG:4326"
)

# Convert both to a metric CRS
rivers_projected = rivers.to_crs("EPSG:3857")
location_projected = location.to_crs("EPSG:3857")

# Calculate distance to every river
distances = rivers_projected.geometry.distance(
    location_projected.geometry.iloc[0]
)

nearest_index = distances.idxmin()

nearest_distance_m = distances.loc[nearest_index]
nearest_distance_km = nearest_distance_m / 1000

print()
print("NEAREST RIVER FOUND")
print("-------------------")
print("Distance:", round(nearest_distance_km, 3), "km")
