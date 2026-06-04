# -----------------------------------------------------------------------------
# Source: Ebrahim Ghaderpour, et al.: A YOLO-Based Workflow for Detecting 
#         and Mapping Archaeological Stone Cairns in Satellite Imagery: 
#         A Case Study fromWestern Ennedi, Chad
#
# This script is part of the methodology presented in the above publication.
# Users are kindly requested to cite the paper when using or modifying this code.
#
# This script converts YOLO-predicted bounding boxes for circular stone cairns 
# into circles.
# -----------------------------------------------------------------------------

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

# -----------------------------------------------------------------------------
# Input: YOLO predicted boxes (.shp) and Output: YOLO detected circles (.shp)
input_shp = r"D:\CHAD\GoogleSatClipped\PredictedBoxes\PredictedBoxes.shp"
output_shp = r"D:\CHAD\GoogleSatClipped\PredictedCircles\PredictedCircles.shp"

# -----------------------------------------------------------------------------
# Load shapefile
gdf = gpd.read_file(input_shp)

# Ensure projected CRS (meters)
if not gdf.crs:
    raise ValueError("CRS is missing. Assign EPSG:32634 first.")

# -----------------------------------------------------------------------------
# Extract box centers and sizes 
centers = []
sizes = []

for geom in gdf.geometry:
    minx, miny, maxx, maxy = geom.bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    
    width = maxx - minx
    height = maxy - miny

    centers.append([cx, cy])
    sizes.append([width, height])

centers = np.array(centers)
sizes = np.array(sizes)

# -----------------------------------------------------------------------------
# Cluster boxes (same stone cairns)
# eps in meters (important!)
eps = 5   # adjust based on stone cairns spacing

clustering = DBSCAN(eps=eps, min_samples=1).fit(centers)
labels = clustering.labels_

# -----------------------------------------------------------------------------
# Create circles
circles = []
ids = []
radii = []

for label in set(labels):
    cluster_centers = centers[labels == label]
    cluster_sizes = sizes[labels == label]

    # Center of the circles (stone cairns)
    cx = np.mean(cluster_centers[:, 0])
    cy = np.mean(cluster_centers[:, 1])

    # Estimate radius from boxes
    avg_w = np.mean(cluster_sizes[:, 0])
    avg_h = np.mean(cluster_sizes[:, 1])

    radius = (avg_w + avg_h) / 4  # average radius

    circle = Point(cx, cy).buffer(radius)

    circles.append(circle)
    ids.append(int(label))
    radii.append(radius)

# -----------------------------------------------------------------------------
# Save output
out_gdf = gpd.GeoDataFrame({"id": ids, "radius_m": radii, "geometry": circles}, crs=gdf.crs)

out_gdf.to_file(output_shp)

print("Circles shapefile saved to:", output_shp)