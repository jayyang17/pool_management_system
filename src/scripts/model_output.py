import cv2
import time
import numpy as np
from ultralytics import YOLO

def run_inference_on_video(model_path, video_path, inference_width=640, inference_height=640, conf_threshold=0.5, display=True):
    """
    Runs YOLO inference on a video file.
    
    Parameters:
      model_path (str): Path to the YOLO model (.pt file).
      video_path (str): Path to the video file.
      inference_width (int): Width to resize frame for faster inference.
      inference_height (int): Height to resize frame for faster inference.
      conf_threshold (float): Confidence threshold for displaying detections.
      display (bool): Whether to display the output window.
    """
    # Load YOLO model
    model = YOLO(model_path, task="detect")
    labels = model.names  # Get class names

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

        original_h, original_w = frame.shape[:2]
        # Resize frame for faster inference
        resized_frame = cv2.resize(frame, (inference_width, inference_height))
        
        # Run inference on the resized frame
        results = model(resized_frame, verbose=False)
        detections = results[0].boxes  # Get bounding boxes from the first result

        # Loop over detections and draw bounding boxes if confidence > threshold
        for detection in detections:
            # Get bounding box coordinates on the resized image
            xyxy = detection.xyxy.cpu().numpy().squeeze()
            if xyxy.ndim == 0 or xyxy.shape[0] < 4:
                continue
            xmin, ymin, xmax, ymax = map(int, xyxy[:4])
            conf = detection.conf.item()
            cls = int(detection.cls.item())

            if conf > conf_threshold:
                # Scale coordinates back to the original frame size.
                scale_x = original_w / inference_width
                scale_y = original_h / inference_height
                xmin = int(xmin * scale_x)
                ymin = int(ymin * scale_y)
                xmax = int(xmax * scale_x)
                ymax = int(ymax * scale_y)

                # Use green for class 0 and red for class 1
                color = (0, 255, 0) if cls == 0 else (250, 128, 114)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f"{labels[cls]}: {conf:.2f}"
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

        # Calculate FPS and display on frame
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

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
    video_path = r"C:\Users\User\Python\pool_management_system\test_videos\videoplayback.mp4"
    
    run_inference_on_video(model_path, video_path)
