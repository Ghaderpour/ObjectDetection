# -----------------------------------------------------------------------------
# Source: Ebrahim Ghaderpour, et al.: A YOLO-Based Workflow for Detecting 
#         and Mapping Archaeological Stone Cairns in Satellite Imagery: 
#         A Case Study fromWestern Ennedi, Chad
#
# This script is part of the methodology presented in the above publication.
# Users are kindly requested to cite the paper when using or modifying this code.
#
# This script loads training and validation tiles defined in data.yaml
# and uses YOLOv8 to train the model.
# -----------------------------------------------------------------------------
from ultralytics import YOLO
import os

# -----------------------------------------------------------------------------
# Paths (use raw strings)
DATA_PATH = r"D:\CHAD\GoogleSatClipped\data.yaml"
MODEL_PATH = r"D:\CHAD\GoogleSatClipped\yolov8n.pt"

# -----------------------------------------------------------------------------
# Check paths exist
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"data.yaml not found at {DATA_PATH}")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"YOLOv8 pretrained model not found at {MODEL_PATH}")

# -----------------------------------------------------------------------------
# Load model (strictly local)
model = YOLO(MODEL_PATH)

# -----------------------------------------------------------------------------
# Training configuration
RUN_NAME = "grave_detection_cpu"

model.train(
    data = DATA_PATH,
    epochs = 500,       # allow enough room
    patience = 12,      # stops automatically (EARLY STOPPING)
    imgsz = 320,
    batch = 4,
    device = "cpu",
    workers = 0,
    name = "grave_detection_clean",
    exist_ok = True
)

# -----------------------------------------------------------------------------
# Resolve results directory (robust way)
results_dir = os.path.abspath(os.path.join("runs", "Detection", RUN_NAME))
print("Full path to results:", results_dir)