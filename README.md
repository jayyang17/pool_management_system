# AI-Powered Resort Pool Management System

## Overview  
This project provides an end-to-end AI-powered system for monitoring resort pool occupancy by analyzing video feeds. It detects pool chairs and determines their occupancy status in real time. The system is built on YOLOv8 and served via a FastAPI backend, enabling users to simply upload a video file for processing.

https://colab.research.google.com/github/username/resort-pool-management/blob/main/notebooks/colab_notebook.ipynb

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
- Annotate raw video frames using LabelStudio with a YOLO-compatible format.

### 2. Preprocessing  
- Resize all annotated images to 640x640 pixels to standardize input dimensions.

### 3. Data Augmentation & Balancing  
- Generate augmented images using techniques such as rotation, flipping, and scaling.
- Balance the dataset by combining augmented images with the resized originals for robust training.

### 4. Model Training  
- The entire training pipeline is encapsulated in `main.py`.
- Running the training script executes the preprocessing, augmentation, and training steps sequentially.

#### Running the Training Pipeline  
To initiate training with preprocessing and ingestion:
```bash
python main.py --ingest --preprocess --train
```
#### Uploading Trained Model to S3 (Optional)  
If you want to sync the trained model to an S3 bucket, add the `--upload` flag:

```bash
python main.py --upload
```
This command triggers the workflow—from data annotation, preprocessing, and augmentation to model training.  
Training parameters can be adjusted directly in `main.py` to suit your dataset and experimental requirements.

## Inference  
The system includes an inference API to process video inputs and return real-time occupancy analytics.

### 1. FastAPI Server Setup  
Start the FastAPI server by running:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at **[http://localhost:8000](http://localhost:8000).**

## Video Upload & Processing  
Users can upload a video file through the provided interface.  
The API processes the video by:  
- Extracting frames  
- Applying the YOLOv8 model for detection  
- Evaluating occupancy based on the overlap of detected persons and chairs  

---

## Limitations & Future Work  
While the current implementation effectively supports both training and real-time video inference, the following enhancements are recommended for future iterations:

### **Potential Enhancements**  
- **Enhanced Deployment**: Automate deployment using **Docker, GitHub Actions, and AWS (ECR/EC2)** for better scalability and maintenance.  
- **Real-Time Optimization**: Optimize the video processing pipeline for **faster inference and real-time performance**.  
- **Advanced Analytics & Dashboard Integration**: Develop a **dashboard for visualizing real-time occupancy analytics and historical trends**.  
- **Occupancy Metrics Extraction**: Implement tracking of:  
  - Number of pool chairs detected  
  - Count of occupied chairs  
  - Overall occupancy rate  
- **Data Logging for Analytics**: Store logs for future data analytics and trend analysis.  
- **Refined Detection Logic**: Improve the overlap detection algorithm to further boost accuracy.  
