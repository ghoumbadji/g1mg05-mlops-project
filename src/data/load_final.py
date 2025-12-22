"""
Upload cleaned data into S3.
"""

import os
from src.utils.s3_utils import upload_file_to_s3


def load_to_s3_final(bucket_name, local_path, s3_key):
    """Load cleaned data to S3."""
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File {local_path} does not exist.")
    print(f"Uploading cleaned file to {bucket_name}/{s3_key}...")
    upload_file_to_s3(local_path, bucket_name, s3_key)
    print("Data pipeline finished successfully.")
