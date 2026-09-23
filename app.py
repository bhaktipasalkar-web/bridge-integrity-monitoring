import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# BRIDGE STRUCTURAL INTEGRITY MONITORING SYSTEM
# PCA + DEEP AUTOENCODER
# ============================================================

st.set_page_config(
    page_title="Bridge Structural Health Monitor",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "results" / "clean_sensor_data.csv"
PCA_FILE = BASE_DIR / "results" / "pca_features.csv"
AE_RESULTS_FILE = BASE_DIR / "results" / "autoencoder_results.csv"
HISTORY_FILE = BASE_DIR / "results" / "training_history.csv"

PCA_MODEL_FILE = BASE_DIR / "models" / "pca_model.joblib"
AE_MODEL_FILE = BASE_DIR / "models" / "bridge_autoencoder.keras"
THRESHOLD_FILE = BASE_DIR / "models" / "threshold.joblib"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        color: #667085;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .status-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e4e7ec;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        min-height: 125px;
    }

    .status-title {
        color: #667085;
        font-size: 14px;
        font-weight: 600;
    }

    .status-value {
        font-size: 28px;
        font-weight: 800;
        margin-top: 8px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .info-box {
        padding: 18px;
        border-radius: 12px;
        background: white;
        border: 1px solid #e4e7ec;
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        color: #667085;
        font-size: 13px;
        padding: 25px 0px 10px 0px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    sensor_df = pd.read_csv(DATA_FILE)
    sensor_df["ts"] = pd.to_datetime(sensor_df["ts"])

    pca_df = pd.read_csv(PCA_FILE)
    pca_df["ts"] = pd.to_datetime(pca_df["ts"])

    ae_df = pd.read_csv(AE_RESULTS_FILE)
    ae_df["ts"] = pd.to_datetime(ae_df["ts"])

    history_df = pd.read_csv(HISTORY_FILE)

    return sensor_df, pca_df, ae_df, history_df


@st.cache_resource
def load_models():

    pca_model = joblib.load(PCA_MODEL_FILE)

    autoencoder = tf.keras.models.load_model(
        AE_MODEL_FILE,
        compile=False
    )

    threshold_data = joblib.load(THRESHOLD_FILE)

    threshold = float(threshold_data["threshold"])

    return pca_model, autoencoder, threshold


# ============================================================
# SAFE LOAD
# ============================================================

try:

    sensor_df, pca_df, ae_df, history_df = load_data()
    pca_model, autoencoder, threshold = load_models()

except Exception as e:

    st.error("Unable to load project files.")
    st.exception(e)
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌉 Bridge Monitor")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📊 Sensor Analysis",
            "🧩 PCA Analysis",
            "🤖 Autoencoder",
            "🚨 Anomaly Detection",
            "📈 Model Performance",
            "ℹ️ About Project"
        ]
    )

    st.markdown("---")

    st.markdown("### Dataset")

    st.write(
        f"**Samples:** {len(sensor_df):,}"
    )

    st.write(
        f"**Sensors:** {len([c for c in sensor_df.columns if c != 'ts'])}"
    )

    st.write(
        f"**PCA Components:** {pca_model['n_components']}"
    )

    st.markdown("---")

    st.caption(
        "PCA + Deep Autoencoder based "
        "Structural Health Monitoring System"
    )


# ============================================================
# COMMON METRICS
# ============================================================

total_samples = len(ae_df)

anomaly_count = int(
    ae_df["anomaly"].sum()
)

normal_count = (
    total_samples - anomaly_count
)

anomaly_percentage = (
    anomaly_count / total_samples * 100
)

max_error = float(
    ae_df["reconstruction_error"].max()
)

max_error_row = ae_df.loc[
    ae_df["reconstruction_error"].idxmax()
]

max_error_timestamp = max_error_row["ts"]


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">🌉 Bridge Structural Health Monitor</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "AI-powered structural anomaly detection using PCA and Deep Autoencoder"
        "</div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    event_start = pd.Timestamp("2023-03-09 23:40:00")
    event_end = pd.Timestamp("2023-03-10 00:10:00")

    event_df = ae_df[
        (ae_df["ts"] >= event_start) &
        (ae_df["ts"] <= event_end)
    ]

    event_anomalies = int(
        event_df["anomaly"].sum()
    )

    if event_anomalies > 0:
        health_status = "ANOMALY DETECTED"
    else:
        health_status = "NORMAL"

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-title">BRIDGE STATUS</div>
                <div class="status-value">{health_status}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-title">TOTAL SAMPLES</div>
                <div class="status-value">{total_samples:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-title">ANOMALIES</div>
                <div class="status-value">{anomaly_count:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-title">ANOMALY RATE</div>
                <div class="status-value">{anomaly_percentage:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # KEY FINDINGS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🔎 Key Findings</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Anomaly Threshold",
            f"{threshold:.4f}"
        )

    with col2:

        st.metric(
            "Maximum Reconstruction Error",
            f"{max_error:.3f}"
        )

    with col3:

        st.metric(
            "Event Period Anomalies",
            f"{event_anomalies}"
        )

    # --------------------------------------------------------
    # RECONSTRUCTION ERROR
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📈 Reconstruction Error Timeline</div>',
        unsafe_allow_html=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=ae_df["ts"],
            y=ae_df["reconstruction_error"],
            mode="lines",
            name="Reconstruction Error"
        )
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        annotation_text="Anomaly Threshold"
    )

    fig.update_layout(
        height=430,
        xaxis_title="Timestamp",
        yaxis_title="Reconstruction Error",
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # EVENT PERIOD
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🚨 Documented Event Period</div>',
        unsafe_allow_html=True
    )

    event_plot = ae_df[
        (ae_df["ts"] >= event_start) &
        (ae_df["ts"] <= event_end)
    ].copy()

    fig_event = go.Figure()

    fig_event.add_trace(
        go.Scatter(
            x=event_plot["ts"],
            y=event_plot["reconstruction_error"],
            mode="lines",
            name="Reconstruction Error"
        )
    )

    fig_event.add_hline(
        y=threshold,
        line_dash="dash",
        annotation_text="Threshold"
    )

    fig_event.update_layout(
        height=400,
        xaxis_title="Timestamp",
        yaxis_title="Reconstruction Error",
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(
        fig_event,
        use_container_width=True
    )

    st.info(
        f"Event-period anomalies: {event_anomalies} "
        f"out of {len(event_df)} samples "
        f"({event_anomalies / len(event_df) * 100:.2f}%). "
        f"Maximum reconstruction error occurred at "
        f"{max_error_timestamp}."
    )


# ============================================================
# SENSOR ANALYSIS
# ============================================================

elif page == "📊 Sensor Analysis":

    st.markdown(
        '<div class="main-title">📊 Sensor Analysis</div>',
        unsafe_allow_html=True
    )

    sensor_columns = [
        c for c in sensor_df.columns
        if c != "ts"
    ]

    selected_sensor = st.selectbox(
        "Select Sensor Channel",
        sensor_columns
    )

    sensor_data = sensor_df[
        ["ts", selected_sensor]
    ].copy()

    fig = px.line(
        sensor_data,
        x="ts",
        y=selected_sensor,
        title=f"{selected_sensor} Sensor Signal"
    )

    fig.update_layout(
        height=500,
        template="plotly_white",
        xaxis_title="Timestamp",
        yaxis_title=selected_sensor
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Sensor Statistics</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Minimum",
            f"{sensor_df[selected_sensor].min():.4f}"
        )

    with c2:
        st.metric(
            "Maximum",
            f"{sensor_df[selected_sensor].max():.4f}"
        )

    with c3:
        st.metric(
            "Mean",
            f"{sensor_df[selected_sensor].mean():.4f}"
        )

    with c4:
        st.metric(
            "Std. Deviation",
            f"{sensor_df[selected_sensor].std():.4f}"
        )


# ============================================================
# PCA ANALYSIS
# ============================================================

elif page == "🧩 PCA Analysis":

    st.markdown(
        '<div class="main-title">🧩 PCA Analysis</div>',
        unsafe_allow_html=True
    )

    variance_ratio = (
        pca_model["explained_variance_ratio"]
    )

    components = np.arange(
        1,
        len(variance_ratio) + 1
    )

    cumulative = np.cumsum(
        variance_ratio
    )

    pca_chart = go.Figure()

    pca_chart.add_trace(
        go.Bar(
            x=components,
            y=variance_ratio * 100,
            name="Individual Variance"
        )
    )

    pca_chart.add_trace(
        go.Scatter(
            x=components,
            y=cumulative * 100,
            mode="lines+markers",
            name="Cumulative Variance",
            yaxis="y2"
        )
    )

    pca_chart.update_layout(
        title="PCA Explained Variance",
        xaxis_title="Principal Component",
        yaxis_title="Individual Variance (%)",
        yaxis2=dict(
            title="Cumulative Variance (%)",
            overlaying="y",
            side="right"
        ),
        height=500,
        template="plotly_white"
    )

    st.plotly_chart(
        pca_chart,
        use_container_width=True
    )

    retained = cumulative[
        pca_model["n_components"] - 1
    ] * 100

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Original Features",
            len(pca_model["sensor_columns"])
        )

    with c2:

        st.metric(
            "Variance Retained",
            f"{retained:.2f}%"
        )

    st.markdown(
        '<div class="section-title">Principal Component Data</div>',
        unsafe_allow_html=True
    )

    pca_display = pca_df.head(100)

    st.dataframe(
        pca_display,
        use_container_width=True
    )


# ============================================================
# AUTOENCODER
# ============================================================

elif page == "🤖 Autoencoder":

    st.markdown(
        '<div class="main-title">🤖 Deep Autoencoder</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        <b>Architecture</b><br><br>
        Input: 13 PCA features<br>
        ↓<br>
        Dense: 8 neurons<br>
        ↓<br>
        Dense: 4 neurons<br>
        ↓<br>
        Bottleneck: 2 neurons<br>
        ↓<br>
        Dense: 4 neurons<br>
        ↓<br>
        Dense: 8 neurons<br>
        ↓<br>
        Output: 13 features
        </div>
        """,
        unsafe_allow_html=True
    )

    if not history_df.empty:

        fig = go.Figure()

        if "loss" in history_df.columns:

            fig.add_trace(
                go.Scatter(
                    y=history_df["loss"],
                    mode="lines",
                    name="Training Loss"
                )
            )

        if "val_loss" in history_df.columns:

            fig.add_trace(
                go.Scatter(
                    y=history_df["val_loss"],
                    mode="lines",
                    name="Validation Loss"
                )
            )

        fig.update_layout(
            title="Autoencoder Training History",
            xaxis_title="Epoch",
            yaxis_title="MSE Loss",
            template="plotly_white",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.metric(
        "Reconstruction Error Threshold",
        f"{threshold:.6f}"
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "🚨 Anomaly Detection":

    st.markdown(
        '<div class="main-title">🚨 Anomaly Detection</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Samples with reconstruction error above the "
        "95th-percentile threshold are classified as anomalies."
    )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    min_error = st.slider(
        "Minimum Reconstruction Error",
        min_value=0.0,
        max_value=float(
            min(max_error, 100)
        ),
        value=float(
            min(threshold, 10)
        )
    )

    filtered = ae_df[
        ae_df["reconstruction_error"] >= min_error
    ].sort_values(
        "reconstruction_error",
        ascending=False
    )

    st.metric(
        "Matching Samples",
        f"{len(filtered):,}"
    )

    st.dataframe(
        filtered.head(500),
        use_container_width=True
    )

    csv_data = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Anomaly Results",
        data=csv_data,
        file_name="bridge_anomaly_results.csv",
        mime="text/csv"
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.markdown(
        '<div class="main-title">📈 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        <b>Important:</b> The available dataset does not provide a
        direct ground-truth damage label for every observation.
        Therefore, conventional accuracy, precision, recall and F1
        are not reported as verified classification metrics.
        The system is evaluated using reconstruction error,
        threshold-based anomaly detection and event-period response.
        </div>
        """,
        unsafe_allow_html=True
    )

    performance_data = pd.DataFrame({
        "Metric": [
            "Total Samples",
            "Normal Samples",
            "Detected Anomalies",
            "Overall Anomaly Rate",
            "Threshold",
            "Maximum Reconstruction Error"
        ],
        "Value": [
            f"{total_samples:,}",
            f"{normal_count:,}",
            f"{anomaly_count:,}",
            f"{anomaly_percentage:.2f}%",
            f"{threshold:.6f}",
            f"{max_error:.6f}"
        ]
    })

    st.dataframe(
        performance_data,
        use_container_width=True,
        hide_index=True
    )

    # Model comparison
    st.markdown(
        '<div class="section-title">Model Comparison</div>',
        unsafe_allow_html=True
    )

    comparison = pd.DataFrame({
        "Capability": [
            "Dimensionality Reduction",
            "Linear Pattern Detection",
            "Non-linear Pattern Detection",
            "Reconstruction Error",
            "Anomaly Detection"
        ],
        "PCA": [
            "✓",
            "✓",
            "—",
            "Derived",
            "✓"
        ],
        "Deep Autoencoder": [
            "✓",
            "—",
            "✓",
            "✓",
            "✓"
        ],
        "PCA + Autoencoder": [
            "✓",
            "✓",
            "✓",
            "✓",
            "✓"
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="main-title">ℹ️ About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### A Deep Autoencoder and PCA-based Diagnostic System
        ### for Bridge Structural Integrity Monitoring

        **Objective**

        To develop an AI-based diagnostic system capable of
        identifying unusual patterns in bridge sensor measurements
        using Principal Component Analysis and a Deep Autoencoder.

        ### Methodology

        **1. Data Collection**

        Real bridge structural monitoring sensor data is used.

        **2. Data Preprocessing**

        Invalid, constant and problematic sensor channels are removed
        or handled through interpolation.

        **3. PCA**

        Principal Component Analysis reduces the sensor feature space
        while retaining the dominant variance.

        **4. Deep Autoencoder**

        The reduced PCA representation is reconstructed by a neural
        network.

        **5. Reconstruction Error**

        The difference between the original PCA representation and
        reconstructed representation is calculated.

        **6. Anomaly Detection**

        Samples exceeding the selected reconstruction-error threshold
        are flagged as anomalies.

        ### System Architecture

        Raw Sensor Data

        ↓

        Data Preprocessing

        ↓

        Standardization

        ↓

        PCA

        ↓

        Feature Extraction

        ↓

        Deep Autoencoder

        ↓

        Reconstruction Error

        ↓

        Threshold

        ↓

        Normal / Anomaly

        ### Project Technologies

        - Python
        - Pandas
        - NumPy
        - TensorFlow / Keras
        - Scikit-learn ecosystem
        - Plotly
        - Streamlit
        - Joblib

        ### Current Dataset Results

        - 47,113 processed samples
        - 25 sensor channels
        - 13 PCA components
        - 99.13% variance retained
        - 2,356 detected anomalies
        - 5.00% overall anomaly rate
        - 156 anomalies during the selected event period
        - 14.13% event-period anomaly rate
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    Bridge Structural Integrity Monitoring System |
    PCA + Deep Autoencoder |
    MLDS Project
    </div>
    """,
    unsafe_allow_html=True
)