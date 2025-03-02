import argparse
from src.utils.utils import read_yaml
from src.constants import CONFIG_FILE_PATH
from src.utils.cloud_utils import sync_folder_to_s3
from src.scripts.data_ingestion import main as ingest_data
from src.scripts.data_preprocess import preprocess_data
from src.scripts.model_trainer import train_yolo

# Load config
config = read_yaml(CONFIG_FILE_PATH)

# S3 Configuration
S3_CONFIG = config["S3"]
bucket_name = S3_CONFIG["BUCKET_NAME"]
s3_data = S3_CONFIG["S3_DATA"]
s3_model_path = S3_CONFIG["S3_MODEL_PATH"]
local_model_path = config["MODEL_OUTPUT_DIR"]

def main(ingest, preprocess, train, upload):
    print("Starting the Pipeline")

    if ingest:
        print("Ingesting data")
        ingest_data()

    if preprocess:
        print("Transforming and Preprocessing data")
        preprocess_data()

    if train:
        print("Training the model")
        train_yolo()

    if upload:
        print("Uploading trained model to S3")
        sync_folder_to_s3(local_model_path, s3_model_path)

    print("Pipeline execution completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline for training YOLOv8 with optional steps.")
    
    parser.add_argument("--ingest", action="store_true", help="Run data ingestion step")
    parser.add_argument("--preprocess", action="store_true", help="Run data preprocessing step")
    parser.add_argument("--train", action="store_true", help="Run model training step")
    parser.add_argument("--upload", action="store_true", help="Upload trained model to S3")

    args = parser.parse_args()

    main(args.ingest, args.preprocess, args.train, args.upload)
