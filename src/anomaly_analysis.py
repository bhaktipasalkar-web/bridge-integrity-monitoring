import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# ANOMALY ANALYSIS
# BRIDGE STRUCTURAL HEALTH MONITORING
# ==========================================

INPUT_FILE = "results/autoencoder_results.csv"
OUTPUT_FOLDER = "results"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 60)
print("ANOMALY ANALYSIS")
print("=" * 60)

# Load results
df = pd.read_csv(INPUT_FILE)
df["ts"] = pd.to_datetime(df["ts"])

# Load threshold
threshold = df.loc[df["reconstruction_error"].idxmax(), "reconstruction_error"]

# The actual threshold used by the Autoencoder pipeline
# Recalculate from the same complete reconstruction-error dataset
threshold = df["reconstruction_error"].quantile(0.95)

df["anomaly"] = (
    df["reconstruction_error"] > threshold
).astype(int)

# --------------------------------------------------
# Overall statistics
# --------------------------------------------------

total_samples = len(df)
anomaly_count = int(df["anomaly"].sum())
normal_count = total_samples - anomaly_count
anomaly_percentage = anomaly_count / total_samples * 100

print("Total samples:", total_samples)
print("Normal samples:", normal_count)
print("Anomaly samples:", anomaly_count)
print("Anomaly percentage:", round(anomaly_percentage, 2), "%")
print("Threshold:", round(threshold, 6))

# --------------------------------------------------
# Plot 1: Complete reconstruction error
# --------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    df["ts"],
    df["reconstruction_error"],
    linewidth=0.8,
    label="Reconstruction Error"
)

plt.axhline(
    threshold,
    linestyle="--",
    linewidth=2,
    label=f"Anomaly Threshold ({threshold:.3f})"
)

plt.title("Bridge Reconstruction Error Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plot1 = os.path.join(
    OUTPUT_FOLDER,
    "reconstruction_error_overall.png"
)

plt.savefig(plot1, dpi=150)
plt.close()

# --------------------------------------------------
# Plot 2: Documented event period
# --------------------------------------------------

event_start = pd.Timestamp("2023-03-09 23:40:00")
event_end = pd.Timestamp("2023-03-10 00:10:00")

event_df = df[
    (df["ts"] >= event_start) &
    (df["ts"] <= event_end)
].copy()

plt.figure(figsize=(14, 6))

plt.plot(
    event_df["ts"],
    event_df["reconstruction_error"],
    linewidth=1.2,
    label="Reconstruction Error"
)

plt.axhline(
    threshold,
    linestyle="--",
    linewidth=2,
    label=f"Threshold ({threshold:.3f})"
)

plt.title(
    "Reconstruction Error Around Documented Bridge Event Period"
)

plt.xlabel("Timestamp")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plot2 = os.path.join(
    OUTPUT_FOLDER,
    "event_period_anomaly.png"
)

plt.savefig(plot2, dpi=150)
plt.close()

# --------------------------------------------------
# Plot 3: Normal vs anomaly counts
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    ["Normal", "Anomaly"],
    [normal_count, anomaly_count]
)

plt.title("Normal vs Anomaly Samples")
plt.ylabel("Number of Samples")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plot3 = os.path.join(
    OUTPUT_FOLDER,
    "normal_vs_anomaly.png"
)

plt.savefig(plot3, dpi=150)
plt.close()

# --------------------------------------------------
# Top anomalies
# --------------------------------------------------

top_anomalies = df.sort_values(
    "reconstruction_error",
    ascending=False
).head(20)

top_file = os.path.join(
    OUTPUT_FOLDER,
    "top_20_anomalies.csv"
)

top_anomalies.to_csv(
    top_file,
    index=False
)

# --------------------------------------------------
# Event-period summary
# --------------------------------------------------

event_anomalies = event_df[
    event_df["anomaly"] == 1
]

event_summary = pd.DataFrame({
    "metric": [
        "Event Period Samples",
        "Event Period Anomalies",
        "Event Period Anomaly Percentage",
        "Maximum Reconstruction Error",
        "Maximum Error Timestamp"
    ],
    "value": [
        len(event_df),
        len(event_anomalies),
        round(
            len(event_anomalies) / len(event_df) * 100,
            2
        ) if len(event_df) > 0 else 0,
        event_df["reconstruction_error"].max(),
        event_df.loc[
            event_df["reconstruction_error"].idxmax(),
            "ts"
        ] if len(event_df) > 0 else None
    ]
})

summary_file = os.path.join(
    OUTPUT_FOLDER,
    "event_summary.csv"
)

event_summary.to_csv(
    summary_file,
    index=False
)

# --------------------------------------------------
# Final output
# --------------------------------------------------

print()
print("=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)

print("Overall graph:", plot1)
print("Event graph:", plot2)
print("Normal/Anomaly graph:", plot3)
print("Top anomalies:", top_file)
print("Event summary:", summary_file)

print()
print("Event period samples:", len(event_df))
print("Event period anomalies:", len(event_anomalies))

if len(event_df) > 0:
    print(
        "Event period anomaly percentage:",
        round(
            len(event_anomalies) / len(event_df) * 100,
            2
        ),
        "%"
    )

    max_row = event_df.loc[
        event_df["reconstruction_error"].idxmax()
    ]

    print(
        "Maximum event-period error:",
        round(
            float(max_row["reconstruction_error"]),
            6
        )
    )

    print(
        "Maximum error timestamp:",
        max_row["ts"]
    )

print()
print("=" * 60)