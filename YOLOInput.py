# -----------------------------------------------------------------------------
# Source: Ebrahim Ghaderpour, et al.: A YOLO-Based Workflow for Detecting 
#         and Mapping Archaeological Stone Cairns in Satellite Imagery: 
#         A Case Study fromWestern Ennedi, Chad
#
# This script is part of the methodology presented in the above publication.
# Users are kindly requested to cite the paper when using or modifying this code.
#
# This script prepares the training and validation tiles for YOLO training
# This is a step before calling YOLOv8. 
# -----------------------------------------------------------------------------
import os
import cv2
import numpy as np
import geopandas as gpd
from osgeo import gdal
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pyproj import CRS

# =========================
# INPUTS
# =========================
raster_path = r"D:\CHAD\GoogleSatClipped\GoogleSatTrain285mm.tif"
shapefile_path = r"D:\CHAD\GoogleSatClipped\TrainCircles\TrainSet.shp"
output_dir = r"D:\CHAD\GoogleSatClipped\YoloTrainSet"

tile_size = 320
stride = 80
PIXEL_SIZE = 0.28492  # meters per pixel

MIN_DIAMETER_M = 1    # Minimum diameter of stone cairns in meter (1 m)
MAX_DIAMETER_M = 100  # Minimum diameter of stone cairns in meter (12 m usually)
MIN_RADIUS_PX = (MIN_DIAMETER_M / 2) / PIXEL_SIZE
MAX_RADIUS_PX = (MAX_DIAMETER_M / 2) / PIXEL_SIZE

# ------------------------------------------------
# Load raster
ds = gdal.Open(raster_path)
gt = ds.GetGeoTransform()
proj = CRS.from_wkt(ds.GetProjection())

B = ds.GetRasterBand(1).ReadAsArray()
G = ds.GetRasterBand(2).ReadAsArray()
R = ds.GetRasterBand(3).ReadAsArray()
img = np.dstack((B, G, R))
h, w, _ = img.shape

# ------------------------------------------------
# Load shapefile and reproject
gdf = gpd.read_file(shapefile_path)
if gdf.crs != proj:
    gdf = gdf.to_crs(proj)

# ------------------------------------------------
# Pixel function (y-fixed)
def map_to_pixel(x, y, gt):
    px = (x - gt[0]) / gt[1]
    py = (gt[3] - y) / abs(gt[5])  # correct north-up
    return px, py


# ------------------------------------------------
# Covert polygons: centers and radii
objects = []
for geom in gdf.geometry:
    print (geom)
    geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for g in geoms:
        if not g.is_valid:
            continue
        cx, cy = g.centroid.x, g.centroid.y
        px, py = map_to_pixel(cx, cy, gt)
        area = g.area
        radius = np.sqrt(area / np.pi) / PIXEL_SIZE
        if radius < MIN_RADIUS_PX or radius > MAX_RADIUS_PX:
            continue
        if 0 <= px < w and 0 <= py < h:
            objects.append((px, py, radius))
print("Valid objects:", len(objects))

# ------------------------------------------------
# Output folders
img_dir = os.path.join(output_dir, "images")
lbl_dir = os.path.join(output_dir, "labels")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(lbl_dir, exist_ok=True)


# ------------------------------------------------
# Tiles and Labels
tile_ids = []
tile_id = 0

for y in tqdm(range(0, h, stride)):
    for x in range(0, w, stride):
        tile = img[y:y+tile_size, x:x+tile_size]
        if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
            continue
        labels = []
        for (px, py, r) in objects:
            # include objects overlapping tile (not just centers)
            if not ((px - r > x) and (px + r < x + tile_size) and (py - r > y) and (py + r < y + tile_size)):
                continue
            # shift to tile coords
            tx = px - x
            ty = py - y
            # YOLO normalized coordinates
            xc = np.clip(tx / tile_size, 0.0, 1.0)
            yc = np.clip(ty / tile_size, 0.0, 1.0)
            w_box = min((2 * r) / tile_size, 1.0)
            h_box = min((2 * r) / tile_size, 1.0)
            if w_box <= 0 or h_box <= 0:
                continue
            labels.append((xc, yc, w_box, h_box))
        # save image
        img_name = f"{tile_id}.jpg"
        cv2.imwrite(os.path.join(img_dir, img_name), tile)
        # save labels
        lbl_name = f"{tile_id}.txt"
        with open(os.path.join(lbl_dir, lbl_name), "w") as f:
            for (xc, yc, w_box, h_box) in labels:
                f.write(f"0 {xc} {yc} {w_box} {h_box}\n")
        tile_ids.append(tile_id)
        tile_id += 1
print("Tiles created:", tile_id)
print(len(labels))
# ------------------------------------------------
# Train and Validation Random Split
train_ids, val_ids = train_test_split(tile_ids, test_size=0.2, random_state=42)

def move_files(ids, split):
    os.makedirs(os.path.join(output_dir, f"images/{split}"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, f"labels/{split}"), exist_ok=True)
    for i in ids:
        os.rename(os.path.join(img_dir, f"{i}.jpg"),
                  os.path.join(output_dir, f"images/{split}/{i}.jpg"))
        os.rename(os.path.join(lbl_dir, f"{i}.txt"),
                  os.path.join(output_dir, f"labels/{split}/{i}.txt"))

move_files(train_ids, "train")
move_files(val_ids, "val")

print("Dataset ready for YOLO Training!")