"""Load model, tokenizer and metrics."""

import os
import pickle
import json
import tensorflow as tf
from src.utils.s3_utils import download_file_from_s3


# Configuration
BUCKET_NAME = os.getenv("BUCKET_NAME")
# S3 keys defined in the pipeline
MODEL_S3_KEY = "models/sentiment_model.keras"
TOKENIZER_S3_KEY = "models/tokenizer.pickle"
METRICS_S3_KEY = "models/evaluation_results.json"


class ModelLoader:
    """Class for artifact loading."""
    _instance = None
    model = None
    tokenizer = None
    metrics = None

    @classmethod
    def get_instance(cls):
        """Singleton pattern to retrieve the loaded instance."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load_artifacts()
        return cls._instance

    def load_artifacts(self):
        """Download and upload model, tokenizer, and KPIs from S3."""
        print("Loading artifacts from S3...")
        # Temporary local paths
        local_model = "temp_model.keras"
        local_tokenizer = "temp_tokenizer.pickle"
        local_metrics = "temp_metrics.json"
        try:
            # 1. Download via your S3 utilities
            download_file_from_s3(BUCKET_NAME, MODEL_S3_KEY, local_model)
            download_file_from_s3(BUCKET_NAME, TOKENIZER_S3_KEY, local_tokenizer)
            download_file_from_s3(BUCKET_NAME, METRICS_S3_KEY, local_metrics)
            # 2. Loading into memory
            self.model = tf.keras.models.load_model(local_model)
            with open(local_tokenizer, "rb") as handle:
                self.tokenizer = pickle.load(handle)
            with open(local_metrics, "r") as f:
                self.metrics = json.load(f)
            print("Model and artifacts loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None 
