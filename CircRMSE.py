# -----------------------------------------------------------------------------
# Source: Ebrahim Ghaderpour, et al.: A YOLO-Based Workflow for Detecting 
#         and Mapping Archaeological Stone Cairns in Satellite Imagery: 
#         A Case Study fromWestern Ennedi, Chad
#
# This script is part of the methodology presented in the above publication.
# Users are kindly requested to cite the paper when using or modifying this code.
#
# This script evaluates the performance of the predicted circles.
# It computes the RMSE between true and predicted circles, along with
# other metrics, such as F1-score, precision, and recall.
# -----------------------------------------------------------------------------
import geopandas as gpd
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# File paths
gt_path = r"D:\CHAD\GoogleSatClipped\TestCircles\Testset.shp"
pred_path = r"D:\CHAD\GoogleSatClipped\PredictedCircles\PredictedCircles.shp"


# -----------------------------------------------------------------------------
# Load data
gdf_gt = gpd.read_file(gt_path)
gdf_pred = gpd.read_file(pred_path)

# -----------------------------------------------------------------------------
# Ensure same CRS (very important!)
if gdf_gt.crs != gdf_pred.crs:
    gdf_pred = gdf_pred.to_crs(gdf_gt.crs)

gdf_gt = gdf_gt.to_crs(epsg=32634)
gdf_pred = gdf_pred.to_crs(epsg=32634)

# -----------------------------------------------------------------------------
# Function: Compute radius 
def compute_radius(polygon):
    area = polygon.area
    return np.sqrt(area / np.pi)

# Add radius columns
gdf_gt["radius"] = gdf_gt.geometry.apply(compute_radius)
gdf_pred["radius"] = gdf_pred.geometry.apply(compute_radius)

# -----------------------------------------------------------------------------
# Spatial join (cairns overlaps)
# Find all overlapping polygon pairs
joined = gpd.sjoin(gdf_gt, gdf_pred, how="inner", predicate="intersects")
print(joined.columns)
# -----------------------------------------------------------------------------
# Match best overlaps
matched_pairs = []

for idx, row in joined.iterrows():
    
    gt_radius = row["radius_left"]
    pred_radius = row["radius_right"]   

    gt_geom = row.geometry
    pred_geom = gdf_pred.loc[row["index_right"]].geometry

    intersection_area = gt_geom.intersection(pred_geom).area

    matched_pairs.append({
        "gt_index": idx,
        "pred_index": row["index_right"],
        "gt_radius": gt_radius,
        "pred_radius": pred_radius,
        "intersection_area": intersection_area
    })

df_matches = pd.DataFrame(matched_pairs)


# -----------------------------------------------------------------------------
# Keep best match per grount-truth
df_best = df_matches.loc[
    df_matches.groupby("gt_index")["intersection_area"].idxmax()
]

# -----------------------------------------------------------------------------
# Calculate the evaluation metrics 
errors = df_best["pred_radius"] - df_best["gt_radius"]
MaxPredRadius= np.round(max(df_best["pred_radius"]),2)
MinPredRadius= np.round(min(df_best["pred_radius"]),2)
MaxGTRadius= np.round(max(df_best["gt_radius"]),2)
MinGTRadius= np.round(min(df_best["gt_radius"]),2)
print(f"Maximum predicted radius (m): {MaxPredRadius}")
print(f"Minimum predicted radius (m): {MinPredRadius}")
print(f"Maximum ground-truth radius (m): {MaxGTRadius}")
print(f"Minimum ground-truth radius (m): {MinGTRadius}")

rmse = np.sqrt(np.mean(errors**2))
print(f"Number of overlapping circles: {len(df_best)}")
print(f"Number of all predictions: {len(gdf_pred)}")
print(f"Number of all truth: {len(gdf_gt)}")
print(f"FN: {len(gdf_gt)-len(df_best)}")
print(f"FP: {len(gdf_pred)-len(df_best)}")
print(f"TP: {len(df_best)}")
print(f"Recall: {100*len(df_best)/len(gdf_gt)}")
print(f"Precision: {100*len(df_best)/len(gdf_pred)}")
print(f"F1-score: {200*((len(df_best)/len(gdf_gt))*len(df_best)/len(gdf_pred))/((len(df_best)/len(gdf_gt))+len(df_best)/len(gdf_pred))}")
print(f"RMSE of radii: {rmse:.4f}")