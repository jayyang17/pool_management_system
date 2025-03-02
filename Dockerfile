# Use a lightweight Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Install system dependencies (required for OpenCV & video processing)
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev ffmpeg

# Prevent Qt plugin issues with OpenCV
ENV QT_QPA_PLATFORM=offscreen

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set PYTHONPATH so FastAPI can find `src.scripts`
ENV PYTHONPATH="/app"

# Expose FastAPI's default port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
