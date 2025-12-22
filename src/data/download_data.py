"""
Download a sample of the Amazon Polarity dataset from Hugging Face,
save it locally as a parquet file, and upload it to an S3 bucket.
"""

from src.utils.s3_utils import upload_file_to_s3
from datasets import load_dataset


def download_and_upload_raw(bucket_name, s3_key, local_path):
    """Download data from Hugging Face and upload it into S3."""
    # 1. Download and Sample
    print("Downloading dataset...")
    dataset = load_dataset("amazon_polarity", split="train[:20000]")
    # 2. Convert to Pandas DataFrame
    df = dataset.to_pandas()
    print(f"Data downloaded successfully. Shape: {df.shape}")
    # 3. Save locally
    df.to_parquet(local_path, index=False)
    print(f"Local file saved: {local_path}")
    # 4. Upload to S3
    print(f"Uploading to S3: {bucket_name}/{s3_key}...")
    upload_file_to_s3(local_path, bucket_name, s3_key)
    print("Upload successful.")
