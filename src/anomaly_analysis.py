import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# ANOMALY ANALYSIS + NOVELTY
# BRIDGE STRUCTURAL HEALTH MONITORING
# ==========================================

INPUT_FILE = "results/autoencoder_results.csv"
OUTPUT_FOLDER = "results"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 70)
print("BRIDGE ANOMALY ANALYSIS + ADAPTIVE SEVERITY DETECTION")
print("=" * 70)

# --------------------------------------------------
# Load Autoencoder Results
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)
df["ts"] = pd.to_datetime(df["ts"])

# --------------------------------------------------
# GLOBAL ANOMALY THRESHOLD
# --------------------------------------------------

threshold = df["reconstruction_error"].quantile(0.95)

df["global_threshold"] = threshold

df["global_anomaly"] = (
    df["reconstruction_error"] > threshold
).astype(int)

# --------------------------------------------------
# NOVELTY:
# ADAPTIVE ROLLING THRESHOLD
#
# Previous observations are used to calculate
# the local baseline.
#
# This avoids using the current observation
# to calculate its own threshold.
# --------------------------------------------------

WINDOW_SIZE = 501
MAD_MULTIPLIER = 3.0

rolling_median = (
    df["reconstruction_error"]
    .shift(1)
    .rolling(
        window=WINDOW_SIZE,
        min_periods=50
    )
    .median()
)

rolling_mad = (
    df["reconstruction_error"]
    .shift(1)
    .rolling(
        window=WINDOW_SIZE,
        min_periods=50
    )
    .apply(
        lambda x: (abs(x - x.median())).median(),
        raw=False
    )
)

# Robust adaptive threshold
adaptive_threshold = (
    rolling_median
    + MAD_MULTIPLIER * 1.4826 * rolling_mad
)

# The adaptive threshold should never become
# lower than the original global threshold.
df["adaptive_threshold"] = (
    adaptive_threshold
    .fillna(threshold)
    .clip(lower=threshold)
)

# --------------------------------------------------
# ADAPTIVE ANOMALY
# --------------------------------------------------

df["adaptive_anomaly"] = (
    df["reconstruction_error"]
    > df["adaptive_threshold"]
).astype(int)

# --------------------------------------------------
# SEVERITY CLASSIFICATION
#
# NORMAL   = below global threshold
# WARNING  = above threshold
# CRITICAL = more than 2x global threshold
# --------------------------------------------------

df["severity"] = "NORMAL"

df.loc[
    df["reconstruction_error"] > threshold,
    "severity"
] = "WARNING"

df.loc[
    df["reconstruction_error"] > (2 * threshold),
    "severity"
] = "CRITICAL"

# --------------------------------------------------
# OVERALL STATISTICS
# --------------------------------------------------

total_samples = len(df)

global_anomaly_count = int(
    df["global_anomaly"].sum()
)

adaptive_anomaly_count = int(
    df["adaptive_anomaly"].sum()
)

normal_count = total_samples - global_anomaly_count

warning_count = int(
    (df["severity"] == "WARNING").sum()
)

critical_count = int(
    (df["severity"] == "CRITICAL").sum()
)

global_anomaly_percentage = (
    global_anomaly_count / total_samples * 100
)

adaptive_anomaly_percentage = (
    adaptive_anomaly_count / total_samples * 100
)

print()
print("TOTAL SAMPLES:", total_samples)
print("GLOBAL THRESHOLD:", round(threshold, 6))

print()
print("GLOBAL ANOMALIES:", global_anomaly_count)
print(
    "GLOBAL ANOMALY PERCENTAGE:",
    round(global_anomaly_percentage, 2),
    "%"
)

print()
print("ADAPTIVE ANOMALIES:", adaptive_anomaly_count)
print(
    "ADAPTIVE ANOMALY PERCENTAGE:",
    round(adaptive_anomaly_percentage, 2),
    "%"
)

print()
print("NORMAL:", normal_count)
print("WARNING:", warning_count)
print("CRITICAL:", critical_count)

# --------------------------------------------------
# PLOT 1:
# GLOBAL + ADAPTIVE THRESHOLD
# --------------------------------------------------

plt.figure(figsize=(15, 6))

plt.plot(
    df["ts"],
    df["reconstruction_error"],
    linewidth=0.8,
    label="Reconstruction Error"
)

plt.plot(
    df["ts"],
    df["adaptive_threshold"],
    linewidth=1.2,
    label="Adaptive Threshold"
)

plt.axhline(
    threshold,
    linestyle="--",
    linewidth=2,
    label=f"Global Threshold ({threshold:.3f})"
)

plt.title(
    "Bridge Reconstruction Error with Adaptive Anomaly Threshold"
)

plt.xlabel("Timestamp")
plt.ylabel("Reconstruction Error")

plt.legend()
plt.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()

plot1 = os.path.join(
    OUTPUT_FOLDER,
    "adaptive_anomaly_detection.png"
)

plt.savefig(
    plot1,
    dpi=150
)

plt.close()

# --------------------------------------------------
# DOCUMENTED EVENT PERIOD
# --------------------------------------------------

event_start = pd.Timestamp(
    "2023-03-09 23:40:00"
)

event_end = pd.Timestamp(
    "2023-03-10 00:10:00"
)

event_df = df[
    (df["ts"] >= event_start) &
    (df["ts"] <= event_end)
].copy()

# --------------------------------------------------
# PLOT 2:
# EVENT PERIOD
# --------------------------------------------------

plt.figure(figsize=(15, 6))

plt.plot(
    event_df["ts"],
    event_df["reconstruction_error"],
    linewidth=1.2,
    label="Reconstruction Error"
)

plt.plot(
    event_df["ts"],
    event_df["adaptive_threshold"],
    linewidth=1.5,
    label="Adaptive Threshold"
)

plt.axhline(
    threshold,
    linestyle="--",
    linewidth=2,
    label=f"Global Threshold ({threshold:.3f})"
)

plt.title(
    "Adaptive Anomaly Detection Around Documented Bridge Event"
)

plt.xlabel("Timestamp")
plt.ylabel("Reconstruction Error")

plt.legend()
plt.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()

plot2 = os.path.join(
    OUTPUT_FOLDER,
    "adaptive_event_period.png"
)

plt.savefig(
    plot2,
    dpi=150
)

plt.close()

# --------------------------------------------------
# PLOT 3:
# SEVERITY DISTRIBUTION
# --------------------------------------------------

severity_counts = df[
    "severity"
].value_counts()

plt.figure(figsize=(8, 5))

severity_order = [
    "NORMAL",
    "WARNING",
    "CRITICAL"
]

values = [
    int(severity_counts.get(
        level,
        0
    ))
    for level in severity_order
]

plt.bar(
    severity_order,
    values
)

plt.title(
    "Bridge Anomaly Severity Distribution"
)

plt.xlabel("Severity Level")
plt.ylabel("Number of Samples")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plot3 = os.path.join(
    OUTPUT_FOLDER,
    "severity_distribution.png"
)

plt.savefig(
    plot3,
    dpi=150
)

plt.close()

# --------------------------------------------------
# TOP 20 ANOMALIES
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
# EVENT PERIOD SUMMARY
# --------------------------------------------------

event_global_anomalies = event_df[
    event_df["global_anomaly"] == 1
]

event_adaptive_anomalies = event_df[
    event_df["adaptive_anomaly"] == 1
]

event_warning = event_df[
    event_df["severity"] == "WARNING"
]

event_critical = event_df[
    event_df["severity"] == "CRITICAL"
]

if len(event_df) > 0:

    event_max_row = event_df.loc[
        event_df[
            "reconstruction_error"
        ].idxmax()
    ]

    event_summary = pd.DataFrame({

        "metric": [

            "Event Period Samples",

            "Global Threshold",

            "Global Event Anomalies",

            "Global Event Anomaly Percentage",

            "Adaptive Event Anomalies",

            "Adaptive Event Anomaly Percentage",

            "Warning Samples",

            "Critical Samples",

            "Maximum Reconstruction Error",

            "Maximum Error Timestamp"

        ],

        "value": [

            len(event_df),

            threshold,

            len(event_global_anomalies),

            round(
                len(event_global_anomalies)
                / len(event_df)
                * 100,
                2
            ),

            len(event_adaptive_anomalies),

            round(
                len(event_adaptive_anomalies)
                / len(event_df)
                * 100,
                2
            ),

            len(event_warning),

            len(event_critical),

            float(
                event_max_row[
                    "reconstruction_error"
                ]
            ),

            event_max_row["ts"]

        ]

    })

else:

    event_summary = pd.DataFrame({
        "metric": [],
        "value": []
    })

summary_file = os.path.join(
    OUTPUT_FOLDER,
    "adaptive_event_summary.csv"
)

event_summary.to_csv(
    summary_file,
    index=False
)

# --------------------------------------------------
# SAVE NOVELTY RESULTS
# --------------------------------------------------

novelty_file = os.path.join(
    OUTPUT_FOLDER,
    "novelty_anomaly_results.csv"
)

df.to_csv(
    novelty_file,
    index=False
)

# --------------------------------------------------
# FINAL OUTPUT
# --------------------------------------------------

print()
print("=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print()
print("Generated Files:")

print(
    "1. Adaptive graph:",
    plot1
)

print(
    "2. Event-period graph:",
    plot2
)

print(
    "3. Severity graph:",
    plot3
)

print(
    "4. Top anomalies:",
    top_file
)

print(
    "5. Event summary:",
    summary_file
)

print(
    "6. Novelty results:",
    novelty_file
)

print()
print("=" * 70)
print("EVENT PERIOD RESULTS")
print("=" * 70)

print(
    "Event samples:",
    len(event_df)
)

print(
    "Global anomalies:",
    len(event_global_anomalies)
)

print(
    "Adaptive anomalies:",
    len(event_adaptive_anomalies)
)

print(
    "Warning samples:",
    len(event_warning)
)

print(
    "Critical samples:",
    len(event_critical)
)

if len(event_df) > 0:

    print(
        "Maximum reconstruction error:",
        round(
            float(
                event_df[
                    "reconstruction_error"
                ].max()
            ),
            6
        )
    )

    print(
        "Maximum error timestamp:",
        event_df.loc[
            event_df[
                "reconstruction_error"
            ].idxmax(),
            "ts"
        ]
    )

print()
print("=" * 70)
print("NOVELTY IMPLEMENTED SUCCESSFULLY")
print("=" * 70)