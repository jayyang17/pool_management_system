import os
from src.utils.cloud_utils import download_from_s3
from src.utils.utils import read_yaml, unzip_file
from src.constants import CONFIG_FILE_PATH
# config
config = read_yaml(CONFIG_FILE_PATH)

# Define S3 parameters
S3_CONFIG = config["S3"]
bucket_name = S3_CONFIG["BUCKET_NAME"]
s3_data = S3_CONFIG["S3_DATA"]
s3_output_path = config["RAW_DATA_PATH"]
unzip_output_path = config["ORIGINAL_PATH"]
# 

def main():
    # download data from s3
    download_from_s3(bucket_name, s3_data, s3_output_path)

    # unzip to local
    unzip_file(s3_output_path,unzip_output_path)

if __name__ == "__main__":
    main()

