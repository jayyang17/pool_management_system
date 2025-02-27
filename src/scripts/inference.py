import torch
import cv2
import numpy as np
import onnxruntime as ort


# Load your PyTorch model
model = torch.load("models/model.pt")
model.eval()

# Create a dummy input tensor with the expected input size (e.g., 1x3x640x640)
dummy_input = torch.randn(1, 3, 640, 640)

# Export the model to ONNX
torch.onnx.export(model, dummy_input, "models/model.onnx",
                  opset_version=11,
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})


# Load the ONNX model using ONNX Runtime
session = ort.InferenceSession("models/model.onnx")
input_name = session.get_inputs()[0].name

# Define video paths
video_path = "data/input_video.mp4"  # Replace with your input video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Prepare to write the output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("data/output_video.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing:
    # Convert frame from BGR to RGB, resize, normalize, and convert to numpy array in CHW format.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (640, 640))
    input_tensor = resized_frame.astype(np.float32) / 255.0  # Normalize to [0,1]
    input_tensor = np.transpose(input_tensor, (2, 0, 1))       # Convert from HWC to CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)        # Add batch dimension

    # Run inference using the ONNX model
    outputs = session.run(None, {input_name: input_tensor})
    
    # Assume outputs[0] contains detections for the batch in the form:
    # [[x1, y1, x2, y2, confidence, class], ...]
    detections = outputs[0]

    # Postprocessing: Draw detections on the original frame.
    # Adjust coordinates from the resized 640x640 image back to the original frame dimensions.
    for det in detections[0]:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5:  # Confidence threshold; adjust as needed
            x_scale = frame_width / 640
            y_scale = frame_height / 640
            x1 = int(x1 * x_scale)
            y1 = int(y1 * y_scale)
            x2 = int(x2 * x_scale)
            y2 = int(y2 * y_scale)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {int(cls)}: {conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to the output video and display it
    out.write(frame)
    cv2.imshow("ONNX Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
