import yaml
import os
import random
import shutil
from pathlib import Path

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

def create_data_yaml(data_path, train_path, val_path, classes_txt_path, output_yaml_path):
    """Creates a YOLO data.yaml file."""
    try:
        with open(classes_txt_path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        
        data = {
            'path': data_path,  # Example: 'data'
            'train': train_path,  # Example: 'data/train/images'
            'val': val_path,  # Example: 'data/validation/images'
            'nc': len(classes),
            'names': classes
        }

        write_yaml(data, output_yaml_path)
        print(f"YAML file created at {output_yaml_path}")

    except FileNotFoundError:
        print(f"File not found: {classes_txt_path}")
    except Exception as e:
        print(f"Error: {e}")


def split_train_val(data_path, train_percent=0.8, output_dir="data"):
    """
    Splits image and annotation files into train and validation sets.
    """

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Directory not found: {data_path}")

    if not (0.01 <= train_percent <= 0.99):
        raise ValueError(f"Invalid train_percent: {train_percent}. Must be between 0.01 and 0.99")

    val_percent = 1 - train_percent

    # Define input dataset paths
    input_image_path = os.path.join(data_path, "images")
    input_label_path = os.path.join(data_path, "labels")

    # Define paths to train/val images and labels
    train_img_path = os.path.join(output_dir, "train/images")
    train_txt_path = os.path.join(output_dir, "train/labels")
    val_img_path = os.path.join(output_dir, "validation/images")
    val_txt_path = os.path.join(output_dir, "validation/labels")

    # Create output directories
    for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
        os.makedirs(dir_path, exist_ok=True)

    # Get list of image and annotation files
    img_file_list = list(Path(input_image_path).rglob("*"))
    txt_file_list = list(Path(input_label_path).rglob("*"))

    print(f"Found {len(img_file_list)} images and {len(txt_file_list)} labels.")

    # Determine number of files for train and validation
    train_num = int(len(img_file_list) * train_percent)
    val_num = len(img_file_list) - train_num

    print(f"Splitting data: {train_num} for training, {val_num} for validation.")

    # Shuffle and split files
    random.shuffle(img_file_list)
    train_files = img_file_list[:train_num]
    val_files = img_file_list[train_num:]

    for img_path, target_dir in zip([train_files, val_files], [train_img_path, val_img_path]):
        for img in img_path:
            img_fn = img.name
            base_fn = img.stem
            txt_fn = base_fn + ".txt"
            txt_path = os.path.join(input_label_path, txt_fn)
            label_target_dir = train_txt_path if target_dir == train_img_path else val_txt_path

            shutil.copy(img, os.path.join(target_dir, img_fn))  # Copy image
            if os.path.exists(txt_path):  # Copy label if it exists
                shutil.copy(txt_path, os.path.join(label_target_dir, txt_fn))

    print("Data split complete! Check the 'train/' and 'validation/' folders.")

