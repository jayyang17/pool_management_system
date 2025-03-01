import os
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
import tempfile
from src.scripts.inference import load_model, read_image, run_inference_on_video
import cv2
import numpy as np
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# FastAPI app setup
app = FastAPI(title="YOLOv8 Inference API", version="1.0")

# Enable CORS for frontend access
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Model path
MODEL_PATH = Path(os.getenv("MODEL_PATH", "final_model/train/weights/best.pt")).resolve()
logger.info(f"Using YOLO model from: {MODEL_PATH}")

# Load YOLO model at startup
try:
    model = load_model(str(MODEL_PATH))
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    model = None  # Prevent crashes if model isn't found

@app.get("/", tags=["root"])
async def index():
    """Redirects to API docs."""
    return RedirectResponse(url="/docs")

@app.post("/predict/")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    conf_threshold: float = 0.5 
):
    """
    Accepts an image file, runs YOLO inference, and returns an HTML page with the processed image.
    
    Query Parameters:
    - `conf_threshold` (float, default=0.5): Minimum confidence score to display detections.

    Returns:
    - An HTML page displaying the processed image and detected objects.
    """
    if model is None:
        return {"error": "🚨 Model is not loaded. Check logs for issues."}

    try:
        # Read and process image
        image_bytes = await file.read()
        processed_img = read_image(model, image_bytes, conf_threshold=conf_threshold)

        # Encode the processed image as JPEG
        _, encoded_image = cv2.imencode('.jpg', processed_img)

        return Response(content=encoded_image.tobytes(), media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return {"error": str(e)}

@app.post("/predict/video/")
async def predict_video(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.5, description="Confidence threshold for detections")
):
    """
    Accepts a video file, runs YOLO inference frame-by-frame,
    and returns the processed video as a response.

    Query Parameters:
    - `conf_threshold` (float, default=0.5): Minimum confidence to display detections.
    
    Returns:
    - A video (mp4) with bounding boxes drawn on each frame.
    """
    if model is None:
        return {"error": "Model is not loaded. Check logs for issues."}
    
    try:
        # Save the uploaded video to a temporary file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input_path = temp_input.name
        temp_input.close()
        with open(temp_input_path, "wb") as f:
            f.write(await file.read())
        logger.info(f"Video saved to temp file: {temp_input_path}")

        # Create a temporary file for the processed output video
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_output_path = temp_output.name
        temp_output.close()

        # Run inference on video: this function writes processed frames to temp_output_path.
        run_inference_on_video(model, temp_input_path, temp_output_path, conf_threshold=conf_threshold, imgsz=640)

        # Read the processed video and return it as the response
        with open(temp_output_path, "rb") as processed_video:
            video_bytes = processed_video.read()

        return Response(content=video_bytes, media_type="video/mp4")

    except Exception as e:
        logger.error(f"Video inference failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
