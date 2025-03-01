import logging
from src.utils.utils import *
from src.constants import CONFIG_FILE_PATH
from src.utils.cloud_utils import  sync_folder_to_s3
from src.scripts.data_ingestion import main as ingest_data
from src.scripts.data_preprocess import preprocess_data
from src.scripts.model_trainer import train_yolo

# Load config
config = read_yaml(CONFIG_FILE_PATH)

# S3 Configuration
S3_CONFIG = config["S3"]
bucket_name = S3_CONFIG["BUCKET_NAME"]
s3_data = S3_CONFIG["S3_DATA"]

# Model paths
s3_model_path = S3_CONFIG["S3_MODEL_PATH"]
local_model_path = config["MODEL_OUTPUT_DIR"]

def main():
    print("Starting the Pipeline")

    print("Ingesting data")
    ingest_data()

    print("Transforming and Preprocessing data")
    preprocess_data()

    print("Training the model")
    train_yolo()

    print("Uploading trained model to S3")
    sync_folder_to_s3(local_model_path, s3_model_path)

    print("Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
