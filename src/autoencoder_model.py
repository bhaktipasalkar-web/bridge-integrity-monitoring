import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# DEEP AUTOENCODER
# BRIDGE STRUCTURAL HEALTH MONITORING
# ==========================================

INPUT_FILE = "results/pca_features.csv"
MODEL_FOLDER = "models"
RESULTS_FOLDER = "results"

EPOCHS = 50
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.20

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

print("=" * 60)
print("DEEP AUTOENCODER TRAINING")
print("=" * 60)

# --------------------------------------------------
# Load PCA features
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)

pc_columns = [
    c for c in df.columns
    if c.startswith("PC")
]

X = df[pc_columns].values.astype(np.float32)

print("Samples:", X.shape[0])
print("Input features:", X.shape[1])
print("Features:", pc_columns)

# --------------------------------------------------
# Autoencoder Architecture
# --------------------------------------------------
input_dim = X.shape[1]

inputs = Input(shape=(input_dim,), name="input_layer")

# Encoder
encoded = Dense(8, activation="relu", name="encoder_8")(inputs)
encoded = Dense(4, activation="relu", name="encoder_4")(encoded)
bottleneck = Dense(2, activation="relu", name="bottleneck")(encoded)

# Decoder
decoded = Dense(4, activation="relu", name="decoder_4")(bottleneck)
decoded = Dense(8, activation="relu", name="decoder_8")(decoded)
outputs = Dense(
    input_dim,
    activation="linear",
    name="reconstruction"
)(decoded)

autoencoder = Model(
    inputs,
    outputs,
    name="Bridge_Deep_Autoencoder"
)

# --------------------------------------------------
# Compile
# --------------------------------------------------
autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

print()
print("MODEL ARCHITECTURE:")
autoencoder.summary()

# --------------------------------------------------
# Early stopping
# --------------------------------------------------
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

# --------------------------------------------------
# Train Autoencoder
# --------------------------------------------------
print()
print("=" * 60)
print("TRAINING STARTED")
print("=" * 60)

history = autoencoder.fit(
    X,
    X,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    shuffle=True,
    callbacks=[early_stopping],
    verbose=1
)

# --------------------------------------------------
# Reconstruction
# --------------------------------------------------
print()
print("Calculating reconstruction errors...")

X_reconstructed = autoencoder.predict(
    X,
    batch_size=BATCH_SIZE,
    verbose=0
)

reconstruction_error = np.mean(
    np.square(X - X_reconstructed),
    axis=1
)

# --------------------------------------------------
# Threshold
# --------------------------------------------------
threshold = np.percentile(
    reconstruction_error,
    95
)

# --------------------------------------------------
# Results
# --------------------------------------------------
results_df = pd.DataFrame({
    "ts": df["ts"],
    "reconstruction_error": reconstruction_error
})

results_df["anomaly"] = (
    results_df["reconstruction_error"] > threshold
).astype(int)

results_file = os.path.join(
    RESULTS_FOLDER,
    "autoencoder_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

# --------------------------------------------------
# Save model
# --------------------------------------------------
model_file = os.path.join(
    MODEL_FOLDER,
    "bridge_autoencoder.keras"
)

autoencoder.save(model_file)

# --------------------------------------------------
# Save training history
# --------------------------------------------------
history_df = pd.DataFrame(history.history)

history_file = os.path.join(
    RESULTS_FOLDER,
    "training_history.csv"
)

history_df.to_csv(
    history_file,
    index=False
)

# --------------------------------------------------
# Save threshold
# --------------------------------------------------
threshold_file = os.path.join(
    MODEL_FOLDER,
    "threshold.joblib"
)

joblib.dump(
    {
        "threshold": threshold,
        "method": "95th_percentile"
    },
    threshold_file
)

# --------------------------------------------------
# Final statistics
# --------------------------------------------------
anomaly_count = results_df["anomaly"].sum()
anomaly_percentage = (
    anomaly_count / len(results_df)
) * 100

print()
print("=" * 60)
print("AUTOENCODER TRAINING COMPLETED")
print("=" * 60)

print(
    "Final Training Loss:",
    round(float(history.history["loss"][-1]), 8)
)

print(
    "Final Validation Loss:",
    round(float(history.history["val_loss"][-1]), 8)
)

print(
    "Reconstruction Error Mean:",
    round(float(reconstruction_error.mean()), 8)
)

print(
    "Reconstruction Error Max:",
    round(float(reconstruction_error.max()), 8)
)

print(
    "Anomaly Threshold:",
    round(float(threshold), 8)
)

print(
    "Anomalies Detected:",
    int(anomaly_count)
)

print(
    "Anomaly Percentage:",
    round(float(anomaly_percentage), 2),
    "%"
)

print()
print("MODEL:", model_file)
print("RESULTS:", results_file)
print("HISTORY:", history_file)
print("THRESHOLD:", threshold_file)

print()
print("=" * 60)
print("AUTOENCODER PIPELINE COMPLETED")
print("=" * 60)