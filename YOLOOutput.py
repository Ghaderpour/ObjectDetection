# -----------------------------------------------------------------------------
# Source: Ebrahim Ghaderpour, et al.: A YOLO-Based Workflow for Detecting 
#         and Mapping Archaeological Stone Cairns in Satellite Imagery: 
#         A Case Study fromWestern Ennedi, Chad
#
# This script is part of the methodology presented in the above publication.
# Users are kindly requested to cite the paper when using or modifying this code.
#
# This script uses the best-trained YOLO model (best.pt) to run inference
# on a GeoTIFF and predict bounding boxes for stone cairns (graves).
# -----------------------------------------------------------------------------
from ultralytics import YOLO
import geopandas as gpd
from shapely.geometry import box as shapely_box
from osgeo import gdal
import numpy as np

# -----------------------------------------------------------------------------
# Load model (Detection!)
model = YOLO(r"D:\CHAD\runs-optimal\detect\grave_detection_clean\weights\best.pt")
raster_path = "D:\CHAD\GoogleSatClipped\GoogleSatTest285mm.tif"

ds = gdal.Open(raster_path)
gt = ds.GetGeoTransform()
proj = ds.GetProjection()

# -----------------------------------------------------------------------------
# Read RGB image  
B = ds.GetRasterBand(1).ReadAsArray()
G = ds.GetRasterBand(2).ReadAsArray()
R = ds.GetRasterBand(3).ReadAsArray()
img = np.dstack((B, G, R))

h, w, _ = img.shape

# -----------------------------------------------------------------------------
# Convert pixels to map coordinates
def pixel_to_map(px, py, gt):
    X = gt[0] + px * gt[1] + py * gt[2]
    Y = gt[3] + px * gt[4] + py * gt[5]
    return (X, Y)

# -----------------------------------------------------------------------------
# Tile prediction
tile_size = 320
stride = 80  

polygons = []

for y in range(0, h, stride):
    print(y,h)
    for x in range(0, w, stride):
        tile = img[y:y+tile_size, x:x+tile_size]
        if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
            continue

        results = model.predict(tile, conf=0.41, iou=0.5, verbose=False)

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes.data.cpu().numpy()[:, :4]

            for box in boxes:
                x1, y1, x2, y2 = box
                # shift back to full image coordinates
                x1 += x
                x2 += x
                y1 += y
                y2 += y
                # convert to map coordinates
                X1, Y1 = pixel_to_map(x1, y1, gt)
                X2, Y2 = pixel_to_map(x2, y2, gt)
                # create polygon (note Y flip!)
                poly = shapely_box(X1, Y2, X2, Y1)
                if poly.is_valid:
                    polygons.append(poly)
print("Total detections:", len(polygons))

# -----------------------------------------------------------------------------
# Save shapefile
if polygons:
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=proj)
    gdf.to_file(r"D:\CHAD\GoogleSatClipped\PredictedBoxes\PredictedBoxes.shp")
    print("Saved", len(polygons), "stone cairns")
else:
    print("No detections")
    
