import cv2
import time
import numpy as np
from ultralytics import YOLO

import cv2
import numpy as np
from ultralytics import YOLO

def load_model(model_path):
    """
    Loads the YOLO model from the given model path.
    """
    model = YOLO(model_path, task="detect")
    return model

def read_image(model, image_bytes, imgsz=640, conf_threshold=0.5 ):
    """
    Reads an image from bytes, runs YOLO inference, and returns the processed image.
    Confidence score defaults at 0.5

    Returns:
        np.ndarray: Processed image with bounding boxes drawn.
    """
    # Convert bytes to OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    labels = model.names

    if img is None:
        raise ValueError("❌ Image decoding failed.")

    # Save original dimensions
    original_h, original_w = img.shape[:2]

    # Resize to model input size
    resized_img = cv2.resize(img, (imgsz, imgsz))

    # Run inference
    results = model(resized_img, verbose=False)
    detections = results[0].boxes

    for detection in detections:
        xyxy = detection.xyxy.cpu().numpy().squeeze()
        if xyxy.ndim == 0 or xyxy.shape[0] < 4:
            continue

        xmin, ymin, xmax, ymax = map(int, xyxy[:4])
        conf = detection.conf.item()
        cls = int(detection.cls.item())

        if conf < conf_threshold:
            continue  

        label = f"Class {cls} ({conf:.2f})"

        # Correctly scale bounding boxes back to original image size
        scale_x = original_w / imgsz
        scale_y = original_h / imgsz
        xmin = int(xmin * scale_x)
        ymin = int(ymin * scale_y)
        xmax = int(xmax * scale_x)
        ymax = int(ymax * scale_y)

        # Draw bounding box
        color = (0, 255, 255) if cls == 0 else (0, 255, 100)
        label = f"{labels.get(cls, str(cls))}: {conf:.2f}"

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img, label, (xmin, max(ymin - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



    return img

def run_inference_on_video(model_path, video_path, imgsz=640, conf_threshold=0.5, display=True):
    """
    Runs YOLO inference on a video file.
    """
    # Load model and its label names
    model = load_model(model_path)
    labels = model.names

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        original_h, original_w = frame.shape[:2]
        resized_frame = cv2.resize(frame, (imgsz, imgsz))
        results = model(resized_frame, verbose=False)
        detections = results[0].boxes

        for detection in detections:
            xyxy = detection.xyxy.cpu().numpy().squeeze()
            if xyxy.ndim == 0 or xyxy.shape[0] < 4:
                continue
            xmin, ymin, xmax, ymax = map(int, xyxy[:4])
            conf = detection.conf.item()
            cls = int(detection.cls.item())

            if conf > conf_threshold:
                scale_x = original_w / imgsz
                scale_y = original_h / imgsz
                xmin = int(xmin * scale_x)
                ymin = int(ymin * scale_y)
                xmax = int(xmax * scale_x)
                ymax = int(ymax * scale_y)
                
                color = (0, 255, 255) if cls == 0 else (0, 255, 100)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f"{labels.get(cls, str(cls))}: {conf:.2f}"
                cv2.putText(frame, label, (xmin, max(ymin - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if display:
            cv2.imshow("YOLO Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage for video inference:
    model_path = r"C:\Users\User\Python\pool_management_system\final_model\train\weights\best.pt"
    video_path = r"C:\Users\User\Python\pool_management_system\test_videos\clip4.mp4"
    run_inference_on_video(model_path, video_path)
