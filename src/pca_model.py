import pandas as pd
import numpy as np
import os
import joblib

# ==========================================
# PCA MODEL FOR BRIDGE SENSOR DATA
# ==========================================

INPUT_FILE = "results/clean_sensor_data.csv"
MODEL_FOLDER = "models"

N_COMPONENTS = 13

# Create model folder
os.makedirs(MODEL_FOLDER, exist_ok=True)

print("=" * 60)
print("PCA MODEL TRAINING")
print("=" * 60)

# --------------------------------------------------
# Load cleaned dataset
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)

# Sensor data only
sensor_columns = [c for c in df.columns if c != "ts"]
X = df[sensor_columns].values.astype(float)

print("Rows:", X.shape[0])
print("Original features:", X.shape[1])

# --------------------------------------------------
# Standardization
# --------------------------------------------------
mean = X.mean(axis=0)
std = X.std(axis=0)

# Prevent division by zero
std[std == 0] = 1.0

X_scaled = (X - mean) / std

# --------------------------------------------------
# PCA using NumPy SVD
# --------------------------------------------------
U, S, Vt = np.linalg.svd(
    X_scaled,
    full_matrices=False
)

# Explained variance
explained_variance = (S ** 2) / (len(X_scaled) - 1)
explained_variance_ratio = (
    explained_variance / explained_variance.sum()
)

# Principal components
components = Vt[:N_COMPONENTS]

# Transform data
X_pca = X_scaled @ components.T

cumulative_variance = np.cumsum(
    explained_variance_ratio
)

# --------------------------------------------------
# Save PCA model information
# --------------------------------------------------
pca_model = {
    "mean": mean,
    "std": std,
    "components": components,
    "explained_variance": explained_variance,
    "explained_variance_ratio": explained_variance_ratio,
    "sensor_columns": sensor_columns,
    "n_components": N_COMPONENTS
}

model_path = os.path.join(
    MODEL_FOLDER,
    "pca_model.joblib"
)

joblib.dump(pca_model, model_path)

# --------------------------------------------------
# Save PCA transformed dataset
# --------------------------------------------------
pca_columns = [
    f"PC{i+1}" for i in range(N_COMPONENTS)
]

pca_df = pd.DataFrame(
    X_pca,
    columns=pca_columns
)

pca_df.insert(0, "ts", df["ts"])

pca_output = "results/pca_features.csv"

pca_df.to_csv(
    pca_output,
    index=False
)

# --------------------------------------------------
# Display results
# --------------------------------------------------
print()
print("PCA COMPONENTS:", N_COMPONENTS)
print(
    "VARIANCE RETAINED:",
    round(cumulative_variance[N_COMPONENTS - 1] * 100, 2),
    "%"
)

print("PCA MODEL:", model_path)
print("PCA DATA:", pca_output)
print("PCA DATA SHAPE:", pca_df.shape)

print()
print("Explained Variance:")
for i in range(N_COMPONENTS):
    print(
        f"PC{i+1}: "
        f"{explained_variance_ratio[i] * 100:.2f}% "
        f"(Cumulative: "
        f"{cumulative_variance[i] * 100:.2f}%)"
    )

print()
print("=" * 60)
print("PCA TRAINING COMPLETED")
print("=" * 60)