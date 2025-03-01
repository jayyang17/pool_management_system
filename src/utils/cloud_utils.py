import boto3
import os

def download_from_s3(bucket_name, s3_path, local_path):
    """
    Download a file from S3 to a local path.
    """
    s3 = boto3.client("s3")
    
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket_name, s3_path, local_path)
        print(f"downloaded to {local_path}")
    except Exception as e:
        print(f"Failed to download {s3_path} from S3: {e}")
        raise

# upload folder using aws cli
def sync_folder_to_s3(folder, aws_bucket_url):
    command = f"aws s3 sync {folder} {aws_bucket_url}"
    os.system(command)

