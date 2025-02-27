
import cv2
import torch
import numpy as np
# Parse user inputs
model_path = r"C:\Users\User\Python\pool_management_system\model\train\weights\best.pt"
video_path = r"C:\Users\User\Python\pool_management_system\test_videos\videoplayback.mp4"



# Load your trained model
import torch
from ultralytics.nn.tasks import DetectionModel  # Import the DetectionModel class

# Use add_safe_globals to allow the DetectionModel during loading.
with torch.serialization.add_safe_globals([DetectionModel]):
    model = torch.load(model_path, weights_only=True)  # Load with weights_only=True

model.eval()  # Set the model to evaluation mode

# Open your video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Prepare to write the output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("output_video.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame:
    # Convert BGR to RGB and resize to the input dimensions expected by your model (e.g., 640x640)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (640, 640))
    input_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).float() / 255.0  # Normalize to [0,1]
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)

    # Postprocessing (this section is model-specific; adjust based on your output format)
    # Here, assume outputs is a list/array of detections per image, where each detection is
    # [x1, y1, x2, y2, confidence, class]
    # Also assume coordinates are relative to the resized frame dimensions.
    # You may need to rescale coordinates to match the original frame size.
    for det in outputs[0]:  # Loop over detections for the first (and only) image in the batch
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5:  # Confidence threshold
            # Scale coordinates back to original frame dimensions if necessary
            x_scale = frame_width / 640
            y_scale = frame_height / 640
            x1 = int(x1 * x_scale)
            y1 = int(y1 * y_scale)
            x2 = int(x2 * x_scale)
            y2 = int(y2 * y_scale)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {int(cls)}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to the output video and display it
    out.write(frame)
    cv2.imshow("Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
