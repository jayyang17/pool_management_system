import torch
import uvicorn
import cv2
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from src.utils.utils import read_yaml
from src.constants import CONFIG_FILE_PATH
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load config
config = read_yaml(CONFIG_FILE_PATH)

# Paths
MODEL_PATH = Path(config["MODEL_OUTPUT_DIR"]) / "model.pt"
OUTPUT_DIR = Path(config["INFERENCE_OUTPUT_DIR"]).resolve()

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load YOLO model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"🚨 Trained model not found at {MODEL_PATH}")

logger.info(f"✅ Loading YOLO model from {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

# Initialize FastAPI app
app = FastAPI(title="YOLOv8 Inference API", version="1.0")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file, runs YOLO inference, and returns predictions as JSON.
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run YOLO inference
        logger.info(f"🖼 Running inference on uploaded image: {file.filename}")
        results = model(img)

        # Process results
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box
                conf = round(box.conf[0].item(), 2)  # Confidence score
                cls = int(box.cls[0].item())  # Class index
                detections.append({"class": cls, "confidence": conf, "bbox": [x1, y1, x2, y2]})

        return {"filename": file.filename, "detections": detections}

    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "YOLOv8 Inference API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
