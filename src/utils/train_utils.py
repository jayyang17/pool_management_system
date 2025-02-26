from src.utils.logging import logging
from src.utils.common_utils import write_yaml


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
