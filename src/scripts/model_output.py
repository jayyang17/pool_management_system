import cv2
import time
import numpy as np
from ultralytics import YOLO

# Configuration: change these paths to point to your model and video file.
model_path = r"C:\Users\User\Python\pool_management_system\model\train\weights\best.pt"
video_path = r"C:\Users\User\Python\pool_management_system\test_videos\videoplayback.mp4"

# Load YOLO model
model = YOLO(model_path, task="detect")
labels = model.names  # Get class names

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Optionally, you can record the output by uncommenting the next block.
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# out = cv2.VideoWriter("output_video.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Run inference on the current frame
    results = model(frame, verbose=False)
    detections = results[0].boxes  # Get bounding boxes from the first result

    # Loop over detections and draw bounding boxes if confidence > 0.5
    for detection in detections:
        # Get bounding box coordinates and convert to int
        xyxy = detection.xyxy.cpu().numpy().squeeze()
        # In case there's only one detection, ensure we have 1D array with at least 4 elements.
        if xyxy.ndim == 0 or xyxy.shape[0] < 4:
            continue
        xmin, ymin, xmax, ymax = map(int, xyxy[:4])
        conf = detection.conf.item()
        cls = int(detection.cls.item())

        if conf > 0.5:
            # Draw the bounding box and label on the frame.
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"{labels[cls]}: {conf:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    # Display the inference results.
    cv2.imshow("YOLO Inference", frame)
    # Uncomment the next line if you set up recording.
    # out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# If recording, release the recorder:
# out.release()
cv2.destroyAllWindows()
