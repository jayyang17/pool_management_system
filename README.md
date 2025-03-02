# AI-Powered Resort Pool Management System

## Overview  
This project provides an end-to-end AI-powered system for monitoring resort pool occupancy by analyzing video feeds. It detects pool chairs and determines their occupancy status in real time. The system is built on YOLOv8 and served via a FastAPI backend, enabling users to simply upload a video file for processing.

## Technical Stack  
- **Model**: YOLOv8 fine-tuned on real-world images and synthetic images (created through Gemini)  
- **Backend**: FastAPI  
- **Inference**: Manual execution (deployment automation with Docker, GitHub Actions, and AWS ECR/EC2 is deferred)  
- **Libraries**:  
  - `ultralytics` (for YOLOv8)  
  - `fastapi` (for serving the model)  
  - `opencv-python` (for video processing)

## Training Process  
This repository supports an end-to-end training pipeline designed to optimize the model for accurate pool chair detection and occupancy analysis.

### 1. Data Annotation  
- Annotate raw video frames using [LabelStudio](https://github.com/heartexlabs/label-studio) with a YOLO-compatible format.

### 2. Preprocessing  
- Resize all annotated images to 640x640 pixels to standardize input dimensions.

### 3. Data Augmentation & Balancing  
- Generate augmented images using techniques such as rotation, flipping, and scaling.
- Balance the dataset by combining augmented images with the resized originals for robust training.

### 4. Model Training  
- The entire training pipeline is encapsulated in `main.py`.
- Running the training script executes the preprocessing, augmentation, and training steps sequentially.

#### Running the Training Pipeline  
To initiate the complete training process, execute:
```bash
python main.py --train
