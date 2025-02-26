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

def split_train_val(data_path, train_percent=0.8, output_dir="data"):
    """
    Splits image and annotation files into train and validation sets.
    """
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Directory not found: {data_path}")
    if not (0.01 <= train_percent <= 0.99):
        raise ValueError(f"Invalid train_percent: {train_percent}. Must be between 0.01 and 0.99")

    # Define input dataset paths
    input_image_path = os.path.join(data_path, "images")
    input_label_path = os.path.join(data_path, "labels")

    # Check if input directories exist
    if not os.path.isdir(input_image_path):
        raise FileNotFoundError(f"Image directory not found: {input_image_path}")
    if not os.path.isdir(input_label_path):
        raise FileNotFoundError(f"Label directory not found: {input_label_path}")

    # Define paths to train/val images and labels
    train_img_path = os.path.join(output_dir, "train/images")
    train_txt_path = os.path.join(output_dir, "train/labels")
    val_img_path = os.path.join(output_dir, "validation/images")
    val_txt_path = os.path.join(output_dir, "validation/labels")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
        os.makedirs(dir_path, exist_ok=True)

    # Get list of image and annotation files
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    img_file_list = [f for f in Path(input_image_path).rglob("*") if f.suffix.lower() in img_extensions]
    txt_file_list = [f for f in Path(input_label_path).rglob("*") if f.suffix.lower() == ".txt"]

    print(f"Found {len(img_file_list)} images and {len(txt_file_list)} labels.")

    # Determine number of files for train and validation
    train_num = int(len(img_file_list) * train_percent)
    val_num = len(img_file_list) - train_num
    print(f"Splitting data: {train_num} for training, {val_num} for validation.")

    # Shuffle and split files
    random.seed(42) 
    random.shuffle(img_file_list)
    train_files = img_file_list[:train_num]
    val_files = img_file_list[train_num:]

    # Copy files to train/val directories
    for img_path, target_dir in zip([train_files, val_files], [train_img_path, val_img_path]):
        for img in img_path:
            img_fn = img.name
            base_fn = img.stem
            txt_fn = base_fn + ".txt"
            txt_path = os.path.join(input_label_path, txt_fn)

            if not os.path.exists(txt_path):
                print(f"Label file not found for image {img_fn}. Skipping.")
                continue

            label_target_dir = train_txt_path if target_dir == train_img_path else val_txt_path
            try:
                shutil.copy(img, os.path.join(target_dir, img_fn))  # Copy image
                shutil.copy(txt_path, os.path.join(label_target_dir, txt_fn))  # Copy label
            except Exception as e:
                print(f"Error copying file {img_fn}: {e}")

    print("Data split complete! Check the 'train/' and 'validation/' folders.")


def create_yolo_data_yaml(
        data_path,
        classes_txt_path,  
        output_yaml_path,
        train_path="train/images",
        val_path="validation/images"
        ): 
    
    """Creates a YOLO data.yaml file."""
    try:
        with open(classes_txt_path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        
        data = {
            'path': data_path, 
            'train': train_path,  
            'val': val_path, 
            'nc': len(classes),
            'names': classes
        }

        write_yaml(data, output_yaml_path)
        print(f"YAML file created at {output_yaml_path}")

    except FileNotFoundError:
        print(f"File not found: {classes_txt_path}")
    except Exception as e:
        print(f"Error: {e}")
