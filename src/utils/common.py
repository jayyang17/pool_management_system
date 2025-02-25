import yaml
from src.utils.logging import logging
import datetime
from pathlib import Path

def write_yaml(data, file_path):
    """Writes dictionary data to a YAML file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)
    except Exception as e:
        raise RuntimeError(f"Error writing YAML file {file_path}: {e}")


def read_yaml(file_path):
    """Reads and returns data from a YAML file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if data is not None else {} 
    except Exception as e:
        raise RuntimeError(f"Error reading YAML file {file_path}: {e}")


def create_data_yaml(data_path, train_path, val_path, classes_txt_path, output_yaml_path):
    """Creates a YOLO data.yaml file."""
    try:
        with open(classes_txt_path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        
        data = {
            'path': data_path, #'data',
            'train': train_path, #'data/train/images',
            'val': val_path, #'data/validation/images',
            'nc': len(classes),
            'names': classes
        }

        write_yaml(data, output_yaml_path)
        logging.info(f"YAML file created at {output_yaml_path}")

    except FileNotFoundError:
        logging.error(f"File not found: {classes_txt_path}")
    except Exception as e:
        logging.error(f"Error: {e}")
