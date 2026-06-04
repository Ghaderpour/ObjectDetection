# -----------------------------------------------------------------------------
# Source: Ebrahim Ghaderpour, et al.: A YOLO-Based Workflow for Detecting 
#         and Mapping Archaeological Stone Cairns in Satellite Imagery: 
#         A Case Study fromWestern Ennedi, Chad
#
# This script is part of the methodology presented in the above publication.
# Users are kindly requested to cite the paper when using or modifying this code.
#
# This script downloads the highest-resolution Google Satellite imagery,
# mosaics the tiles, clips the result to the input boundary shapefile,
# and saves the final RGB raster as a GeoTIFF.
# -----------------------------------------------------------------------------

import requests
import mercantile
import geopandas as gpd
import numpy as np
from PIL import Image
from io import BytesIO
from osgeo import gdal, osr

# -----------------------------------------------------------------
# Settings
shapefile = r"D:\CHAD\Border\AOI.shp"
output_tif = r"D:\CHAD\GoogleSatClipped\google_mosaic.tif"
clipped_tif = r"D:\CHAD\GoogleSatClipped\GoogleSatChadAOI.tif"
zoom = 19  # 19–21 for max resolution

tile_url = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"

# -----------------------------------------------------------------
# Load AOI
gdf = gpd.read_file(shapefile).to_crs(epsg=4326)
minx, miny, maxx, maxy = gdf.total_bounds

# -----------------------------------------------------------------
# Get tiles
tiles = list(mercantile.tiles(minx, miny, maxx, maxy, zoom))

tile_size = 256
min_tx = min(t.x for t in tiles)
min_ty = min(t.y for t in tiles)
max_tx = max(t.x for t in tiles)
max_ty = max(t.y for t in tiles)

width = (max_tx - min_tx + 1) * tile_size
height = (max_ty - min_ty + 1) * tile_size

# -----------------------------------------------------------------
# Create mosaic array
mosaic = np.zeros((height, width, 3), dtype=np.uint8)

for tile in tiles:
    url = tile_url.format(x=tile.x, y=tile.y, z=zoom)
    r = requests.get(url)

    if r.status_code != 200:
        continue

    img = Image.open(BytesIO(r.content)).convert("RGB")
    img_array = np.array(img)

    x_offset = (tile.x - min_tx) * tile_size
    y_offset = (tile.y - min_ty) * tile_size

    mosaic[y_offset:y_offset+tile_size, x_offset:x_offset+tile_size] = img_array

print("Tiles stitched into mosaic")

# -----------------------------------------------------------------
# Create GeoTiff
driver = gdal.GetDriverByName("GTiff")
out_ds = driver.Create(output_tif, width, height, 3, gdal.GDT_Byte)

if out_ds is None:
    raise RuntimeError("Failed to create output GeoTIFF")

# -----------------------------------------------------------------
# Georeference 
top_left = mercantile.bounds(min_tx, min_ty, zoom)
bottom_right = mercantile.bounds(max_tx, max_ty, zoom)

xmin = top_left.west
ymax = top_left.north
xmax = bottom_right.east
ymin = bottom_right.south

geotransform = [xmin, (xmax - xmin) / width, 0, ymax, 0, -(ymax - ymin) / height]

out_ds.SetGeoTransform(geotransform)

srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
out_ds.SetProjection(srs.ExportToWkt())

# -----------------------------------------------------------------
# Write RGB bands
out_ds.GetRasterBand(1).WriteArray(mosaic[:, :, 0])
out_ds.GetRasterBand(2).WriteArray(mosaic[:, :, 1])
out_ds.GetRasterBand(3).WriteArray(mosaic[:, :, 2])

out_ds.FlushCache()
out_ds = None

print("GeoTIFF created:", output_tif)

# -----------------------------------------------------------------
# Clip with shapefile 
gdal.Warp(
    clipped_tif,
    output_tif,
    cutlineDSName = shapefile,
    dstSRS="EPSG:32634",  # adjust zone for Chad
    cropToCutline=True,
    dstNodata=-9999
)

print("Clipped GeoTIFF:", clipped_tif)



