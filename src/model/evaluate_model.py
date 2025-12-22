"""
Evaluate the LSTM model.
"""

import json
from tensorflow import keras
from sklearn.metrics import classification_report
from src.utils.s3_utils import upload_file_to_s3


def prepare_test_data(x_test, y_test, tokenizer):
    """Prepare test data for evaluation."""
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    x_test_pad = keras.utils.pad_sequences(
        x_test_seq, padding="post", maxlen=128
    )
    return x_test_pad, y_test


def evaluate(model, x_test_pad, y_test):
    """Evaluate the model and output metrics."""
    results = model.evaluate(x_test_pad, y_test, verbose=0)
    y_pred_prob = model.predict(x_test_pad)
    y_pred = (y_pred_prob > 0.5).astype(int)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    return results, report_dict


def save_and_upload_metrics(results, report_dict, bucket_name, metrics_s3_key):
    """Upload evaluation metrics into S3."""
    metrics_data = {
        "global_score": {
            "loss": float(results[0]),
            "accuracy": float(results[1]),
            "precision": float(results[2]),
            "recall": float(results[3]),
        },
        "classification_report": report_dict,
    }
    local_metrics_file = "metrics.json"
    with open(local_metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    upload_file_to_s3(local_metrics_file, bucket_name, metrics_s3_key)
