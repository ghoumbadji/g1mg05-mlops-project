"""Put all of the model pipeline steps together."""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from src.utils.s3_utils import download_file_from_s3
from . import train_model
from . import evaluate_model


# Configuration
BUCKET_NAME = os.getenv("BUCKET_NAME")
DATA_KEY = "data/processed/amazon_polarity_cleaned.parquet"
LOCAL_DATA_FILE = "amazon_polarity_cleaned.parquet"

# Artifact paths in S3 (to add or to retrieve)
MODEL_S3_KEY = "models/sentiment_model.keras"
TOKENIZER_S3_KEY = "models/tokenizer.pickle"
METRICS_S3_KEY = "models/evaluation_results.json"


def load_artifacts():
    """Download and load artifacts"""
    # 1. Download artifacts
    local_model = "downloaded_model.keras"
    local_tokenizer = "downloaded_tokenizer.pickle"
    download_file_from_s3(BUCKET_NAME, MODEL_S3_KEY, local_model)
    download_file_from_s3(BUCKET_NAME, TOKENIZER_S3_KEY, local_tokenizer)
    # 2. Load artifacts
    model = tf.keras.models.load_model(local_model)
    with open(local_tokenizer, "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer


def load_and_split_data():
    """Download data and split it. Return training data."""
    if not os.path.exists(LOCAL_DATA_FILE):
        download_file_from_s3(BUCKET_NAME, DATA_KEY, LOCAL_DATA_FILE)
    df = pd.read_parquet(LOCAL_DATA_FILE)
    x = df["content"]
    y = df["label"]
    return train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)


def run_model_pipeline():
    """Run the full model pipeline."""
    print("Step 1: Training")
    # 1. Load data
    x_train, x_test, y_train, y_test = load_and_split_data()
    # 2. Train model
    tokenizer, model = train_model.train(x_train, y_train)
    # 3. Save artifacts to S3
    train_model.save_and_upload_models(
        model, tokenizer, BUCKET_NAME, MODEL_S3_KEY, TOKENIZER_S3_KEY
    )
    print("Training complete and artifacts uploaded.")
    print("Step 2: Evaluation")
    # 4. Prepare test data for evaluation
    x_test_pad, y_test = evaluate_model.prepare_test_data(
        x_test, y_test, tokenizer
    )
    # 5. Evaluate the model
    results, report_dict = evaluate_model.evaluate(model, x_test_pad, y_test)
    # 6. Save metrics
    evaluate_model.save_and_upload_metrics(
        results, report_dict, BUCKET_NAME, METRICS_S3_KEY
    )
    print("Evaluation complete and metrics uploaded.")


if __name__ == "__main__":
    run_model_pipeline()
