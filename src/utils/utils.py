import yaml
import os
import random
import shutil
from pathlib import Path
import zipfile

def unzip_file(zip_path, output_dir):
    """
    Unzips a given ZIP file to a specified output directory.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    os.makedirs(output_dir, exist_ok=True)  # Create the output dir if it doesn't exist

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Successfully extracted {zip_path} to {output_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to unzip {zip_path}: {e}")


def read_yaml(file_path):
    """Reads and returns data from a YAML file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if data is not None else {} 
    except Exception as e:
        raise RuntimeError(f"Error reading YAML file {file_path}: {e}")


def write_yaml(data, file_path):
    """Writes dictionary data to a YAML file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)
    except Exception as e:
        raise RuntimeError(f"Error writing YAML file {file_path}: {e}")


def save_model(source_train_dir="/content/runs/detect/train",
               model_weights="best.pt",
               output_dir="/content/model"):
    """
    Saves trained YOLO model weights and training results.
    """

    output_dir = Path(output_dir)
    source_train_dir = Path(source_train_dir)

    # Ensure source directory exists
    if not source_train_dir.exists():
        raise FileNotFoundError(f"Source train directory not found: {source_train_dir}")

    # Create the model output directory if it doesn’t exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy the best model weights
    best_model_path = source_train_dir / "weights" / model_weights
    if best_model_path.exists():
        shutil.copy(best_model_path, output_dir / "model.pt")
    else:
        raise FileNotFoundError(f"Model weights not found: {best_model_path}")

    # Copy entire training folder
    train_copy_path = output_dir / "train"
    if train_copy_path.exists():
        shutil.rmtree(train_copy_path)  # Remove if it already exists
    shutil.copytree(source_train_dir, train_copy_path)

    print(f"Model files saved successfully at: {output_dir}")


def zip_model(model_dir="/content/model", zip_path="/content/model.zip"):
    """
    Zips the saved YOLO model and training results.
    """

    model_dir = Path(model_dir)
    zip_path = Path(zip_path)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Create the zip file
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add model.pt
        model_file = model_dir / "model.pt"
        if model_file.exists():
            zipf.write(model_file, arcname="model.pt")
        else:
            print(f"Warning: {model_file} not found, skipping model.pt in zip.")

        # Add train directory recursively
        train_copy_path = model_dir / "train"
        for file in train_copy_path.rglob("*"):
            zipf.write(file, arcname=str(file.relative_to(model_dir)))

    print(f"Model zipped successfully at: {zip_path}")

def get_latest_yolo_run(runs_dir="runs/detect"):
    """
    Finds the latest YOLO training directory inside 'runs/detect'.
    """
    runs_path = Path(runs_dir)

    if not runs_path.exists():
        print(f"ERROR: No YOLO training directory found in {runs_dir}")
        return None

    # Find all YOLO runs
    yolo_runs = [d for d in runs_path.iterdir() if d.is_dir() and d.name.startswith("train")]

    if not yolo_runs:
        print(f"ERROR: No YOLO training runs found in {runs_dir}")
        return None

    # Sort by modification time (latest first)
    latest_run = max(yolo_runs, key=lambda d: d.stat().st_mtime)
    print(f"Detected latest YOLO training run: {latest_run}")

    return latest_run

