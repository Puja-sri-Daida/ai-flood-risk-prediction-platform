import geopandas as gpd

river_file = "hydrorivers/HydroRIVERS_v10_as_shp/HydroRIVERS_v10_as.shp"

print("Loading river dataset...")

rivers = gpd.read_file(river_file)

print()
print("DATASET LOADED SUCCESSFULLY")
print("Number of river features:", len(rivers))
print("CRS:", rivers.crs)

print()
print("Available columns:")
print(rivers.columns.tolist())
