import cv2
import time
import numpy as np
from ultralytics import YOLO

def run_inference_on_video(model_path, video_path, imgsz=640, conf_threshold=0.5, display=True):
    """
    Runs YOLO inference on a video file.
    
    Parameters:
      model_path (str): Path to the YOLO model (.pt file).
      video_path (str): Path to the video file.
      imgsz (int): Input size for inference (e.g., 640).
      conf_threshold (float): Confidence threshold for displaying detections.
      display (bool): Whether to display the results.
    """
    # Load the pretrained YOLO model (assumes it was trained on imgsz x imgsz images)
    model = YOLO(model_path, task="detect")
    labels = model.names

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Save original dimensions to scale detection coordinates later
        original_h, original_w = frame.shape[:2]
        # Resize frame to imgsz x imgsz (same as training preprocessing)
        resized_frame = cv2.resize(frame, (imgsz, imgsz))
        
        # Run inference on the resized frame
        results = model(resized_frame, verbose=False)
        detections = results[0].boxes

        # Loop through detections and draw bounding boxes if confidence is high enough
        for detection in detections:
            xyxy = detection.xyxy.cpu().numpy().squeeze()
            if xyxy.ndim == 0 or xyxy.shape[0] < 4:
                continue
            xmin, ymin, xmax, ymax = map(int, xyxy[:4])
            conf = detection.conf.item()
            cls = int(detection.cls.item())

            if conf > conf_threshold:
                # Scale coordinates back to the original frame size
                scale_x = original_w / imgsz
                scale_y = original_h / imgsz
                xmin = int(xmin * scale_x)
                ymin = int(ymin * scale_y)
                xmax = int(xmax * scale_x)
                ymax = int(ymax * scale_y)
                
                # Draw bounding box and label
                color = (0, 255, 255) if cls == 0 else (0, 255, 100)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f"{labels[cls]}: {conf:.2f}"
                cv2.putText(frame, label, (xmin, max(ymin - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Calculate FPS and overlay it on the frame
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
    # Change these paths as needed.
    model_path = r"C:\Users\User\Python\pool_management_system\model\train\weights\best.pt"
    video_path = r"C:\Users\User\Python\pool_management_system\test_videos\clip4.mp4"
    
    run_inference_on_video(model_path, video_path)
