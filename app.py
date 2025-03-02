import os
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import tempfile
from src.scripts.inference import load_model, read_image, run_inference_on_video
import cv2
import numpy as np
from ultralytics.utils.downloads import download
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

# Download model from S3 and load YOLO model
S3_MODEL = "https://poolmanagementsystem.s3.ap-southeast-1.amazonaws.com/final_model/train/weights/best.pt"
download_path = download(S3_MODEL, dir="weights/")
MODEL_PATH = "weights/best.pt"
logger.info(f"Using YOLO model from: {MODEL_PATH}")
print(MODEL_PATH)
model = load_model(MODEL_PATH)

@app.get("/", tags=["root"])
async def index():
    """Redirects to API docs."""
    return RedirectResponse(url="/docs")

@app.post("/predict/image/")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    conf_threshold: float = 0.5 
):
    """
    Accepts an image file, runs YOLO inference, and returns the processed image.
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
    
    This endpoint assumes that run_inference_on_video (unchanged) writes the processed video to "processed_video.mp4".
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
        
        # Run inference on video.
        # We are not modifying run_inference_on_video; it is expected to write output to "processed_video.mp4"
        run_inference_on_video(model, temp_input_path, conf_threshold=conf_threshold, imgsz=640)
        
        # Define the output file path (as written by run_inference_on_video)
        output_video_path = "processed_video.mp4"
        
        # Read the processed video and return it as the response
        with open(output_video_path, "rb") as processed_video:
            video_bytes = processed_video.read()
        
        # Cleanup temporary files
        os.remove(temp_input_path)
        logger.info(f"Deleted temporary input file: {temp_input_path}")
        os.remove(output_video_path)
        logger.info(f"Deleted output video file: {output_video_path}")
        
        return Response(content=video_bytes, media_type="video/mp4")
    
    except Exception as e:
        logger.error(f"Video inference failed: {e}")
        return {"error": str(e)}

@app.on_event("shutdown")
def cleanup_model():
    """Deletes the downloaded model when the FastAPI app shuts down."""
    try:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            logger.info(f"Deleted model file: {MODEL_PATH}")
        
        # Remove the weights directory if empty
        weights_dir = Path("weights/")
        if weights_dir.exists() and not any(weights_dir.iterdir()):
            os.rmdir(weights_dir)
            logger.info(f"Deleted empty weights directory: {weights_dir}")
    
    except Exception as e:
        logger.error(f"Failed to delete model file: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
