"""
Build and train a binary classification model using BiLSTMs.
"""

import pickle
import tensorflow as tf
from tensorflow import keras
from src.utils.s3_utils import upload_file_to_s3


def create_lstm_model(vocab_size):
    """Defines the LSTM architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 32),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=["accuracy", "precision", "recall"]
    )
    return model


def train(x_train, y_train):
    """Main function."""
    # 1. Tokenization
    max_vocab = 10000
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(x_train)  # Fit only on training data
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_train_pad = keras.utils.pad_sequences(
        x_train_seq, padding="post", maxlen=128
    )
    # 2. Model Training
    print("Starting training...")
    model = create_lstm_model(max_vocab)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    model.fit(
        x_train_pad, y_train,
        epochs=20,
        validation_split=0.1,
        batch_size=32,
        callbacks=[early_stop]
    )
    return tokenizer, model


def save_and_upload_models(model, tokenizer, bucket_name, model_s3_key,
                           tokenizer_s3_key):
    """Save and upload model into S3."""
    # 3. Save Artifacts and Upload to S3
    print("Saving artifacts...")
    # Save & Upload Model
    local_model_path = "model_temp.keras"
    model.save(local_model_path)
    upload_file_to_s3(local_model_path, bucket_name, model_s3_key)
    # Save & Upload Tokenizer
    local_tok_path = "tokenizer_temp.pickle"
    with open(local_tok_path, "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    upload_file_to_s3(local_tok_path, bucket_name, tokenizer_s3_key)
    print("Training finished and artifacts uploaded to S3.")
