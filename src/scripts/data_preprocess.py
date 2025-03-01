from src.utils.utils import  * 
from src.utils.preprocess_utils import process_dataset, augment_images, balance_dataset, split_train_val, create_yolo_data_yaml
import os
from src.constants import CONFIG_FILE_PATH

def preprocess_data():
    # config
    config = read_yaml(CONFIG_FILE_PATH)

    # PATH CONFIGURATIONS
    ORIGINAL_IMG_DIR  = config["ORIGINAL_IMG_DIR"]    # Images from Label Studio
    YOLO_LABELS_DIR   = config["YOLO_LABELS_DIR"]     # YOLO labels from Label Studio

    # Resized data paths (before augmentation)
    RESIZED_IMG_DIR   = config["RESIZED_IMG_DIR"]     # Resized images
    RESIZED_LABELS_DIR = config["RESIZED_LABELS_DIR"]    # Updated YOLO labels

    # Augmented data paths
    AUGMENTED_IMG_DIR   = config["AUGMENTED_IMG_DIR"]     # Augmented images
    AUGMENTED_LABELS_DIR = config["AUGMENTED_LABELS_DIR"]   # Augmented YOLO labels

    # Target training directories
    TRAINING_IMG_DIR   = config["TRAINING_IMG_DIR"]
    TRAINING_LABELS_DIR = config["TRAINING_LABELS_DIR"]

    # Other paths
    CLASSES_TXT_PATH  = config["CLASSES_TXT_PATH"]
    DATA_YAML_PATH    = config["MODEL"]["DATA_YAML_PATH"]


    # data path for model training
    # Determine the base path dynamically, to avoid changing the env
    if os.path.exists('/content'):
        # Set the dataset path for Colab
        DATA_PATH = '/content/data'
    else:
        # For local
        DATA_PATH = os.path.join(os.getcwd(), 'data')

    # Data splitting ratios
    DATA_SPLIT = config["DATA_SPLIT"]
    IMG_BALANCE_RATIO = DATA_SPLIT["img_balance_ratio"]
    TRAIN_RATIO       = DATA_SPLIT["train_ratio"]

    print("begin processing images")
    process_dataset(ORIGINAL_IMG_DIR, RESIZED_IMG_DIR, RESIZED_LABELS_DIR, YOLO_LABELS_DIR)

    print("augmenting images")
    augment_images(RESIZED_IMG_DIR, RESIZED_LABELS_DIR, AUGMENTED_IMG_DIR, AUGMENTED_LABELS_DIR)

    print("balancing resized images and augmented images volumes")
    balance_dataset(
        RESIZED_IMG_DIR,
        RESIZED_LABELS_DIR,
        AUGMENTED_IMG_DIR,
        AUGMENTED_LABELS_DIR,
        TRAINING_IMG_DIR,
        TRAINING_LABELS_DIR,
        ratio_original=IMG_BALANCE_RATIO,
        total_samples=None)
    
    print("splitting training data")
    split_train_val(TRAINING_IMG_DIR, TRAINING_LABELS_DIR, TRAIN_RATIO)

    print("create yaml file for training")
    create_yolo_data_yaml(DATA_PATH, CLASSES_TXT_PATH, DATA_YAML_PATH)

if __name__ == "__main__":
    preprocess_data()