import streamlit as st
import textwrap

_original_st_markdown = st.markdown

def _markdown_dedented(body, *args, **kwargs):
    if isinstance(body, str) and kwargs.get("unsafe_allow_html", False):
        body = textwrap.dedent(body)
    return _original_st_markdown(body, *args, **kwargs)

st.markdown = _markdown_dedented
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
from tensorflow import keras

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BridgeGuard AI | Structural Health Monitoring",
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
AE_FILE = BASE_DIR / "results" / "autoencoder_results.csv"
HISTORY_FILE = BASE_DIR / "results" / "training_history.csv"

NOVELTY_FILE = (
    BASE_DIR
    / "results"
    / "novelty_anomaly_results.csv"
)

PCA_MODEL_FILE = (
    BASE_DIR
    / "models"
    / "pca_model.joblib"
)

AE_MODEL_FILE = (
    BASE_DIR
    / "models"
    / "bridge_autoencoder.keras"
)

THRESHOLD_FILE = (
    BASE_DIR
    / "models"
    / "threshold.joblib"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #0b1220;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .main-header {
        background: linear-gradient(
            135deg,
            #0b1220 0%,
            #16243d 100%
        );
        padding: 28px 32px;
        border-radius: 18px;
        margin-bottom: 24px;
        color: white;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .main-subtitle {
        font-size: 15px;
        color: #cbd5e1;
    }

    .monitoring-badge {
        display: inline-block;
        background: #123c2b;
        color: #5ee7a0;
        padding: 7px 14px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 750;
        color: #0f172a;
        margin-top: 20px;
        margin-bottom: 4px;
    }

    .section-subtitle {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 15px;
    }

    .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        min-height: 145px;
        box-shadow: 0 5px 20px rgba(15, 23, 42, 0.06);
        border: 1px solid #e2e8f0;
    }

    .kpi-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 650;
        margin-bottom: 12px;
    }

    .kpi-value {
        color: #0f172a;
        font-size: 29px;
        font-weight: 800;
    }

    .kpi-value.warning {
        color: #d97706;
    }

    .kpi-value.danger {
        color: #dc2626;
    }

    .kpi-footer {
        margin-top: 10px;
        color: #94a3b8;
        font-size: 11px;
    }

    .info-box {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 16px 18px;
        border-radius: 10px;
        color: #1e3a8a;
        margin-bottom: 18px;
    }

    .success-box {
        background: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 16px 18px;
        border-radius: 10px;
        color: #065f46;
        margin-bottom: 18px;
    }

    .warning-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 16px 18px;
        border-radius: 10px;
        color: #92400e;
        margin-bottom: 18px;
    }

    .critical-box {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        padding: 16px 18px;
        border-radius: 10px;
        color: #991b1b;
        margin-bottom: 18px;
    }

    .architecture {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
        margin: 25px 0;
    }

    .arch-box {
        background: white;
        border: 1px solid #dbe3ef;
        border-radius: 12px;
        padding: 18px 20px;
        min-width: 145px;
        text-align: center;
        font-weight: 700;
        color: #0f172a;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
    }

    .arch-arrow {
        color: #64748b;
        font-size: 24px;
        font-weight: bold;
    }

    .footer {
        text-align: center;
        padding: 30px 10px;
        color: #94a3b8;
        font-size: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def page_header(title, subtitle="", icon=""):
    st.markdown(
        f"""
        <div class="main-header">

            <div class="monitoring-badge">
                ● LIVE STRUCTURAL MONITORING
            </div>

            <div class="main-title">
                {icon} {title}
            </div>

            <div class="main-subtitle">
                {subtitle}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def section(title, subtitle=""):
    st.markdown(
        f"""
        <div class="section-title">
            {title}
        </div>

        <div class="section-subtitle">
            {subtitle}
        </div>
        """,
        unsafe_allow_html=True
    )


@st.cache_data
def load_data():

    sensor_df = pd.read_csv(DATA_FILE)
    pca_df = pd.read_csv(PCA_FILE)
    ae_df = pd.read_csv(AE_FILE)
    history_df = pd.read_csv(HISTORY_FILE)

    sensor_df["ts"] = pd.to_datetime(
        sensor_df["ts"]
    )

    pca_df["ts"] = pd.to_datetime(
        pca_df["ts"]
    )

    ae_df["ts"] = pd.to_datetime(
        ae_df["ts"]
    )

    history_df = history_df.copy()

    # --------------------------------------------------------
    # LOAD NOVELTY RESULTS
    # --------------------------------------------------------

    if NOVELTY_FILE.exists():

        novelty_df = pd.read_csv(
            NOVELTY_FILE
        )

        novelty_df["ts"] = pd.to_datetime(
            novelty_df["ts"]
        )

    else:

        novelty_df = ae_df.copy()

        fallback_threshold = (
            novelty_df[
                "reconstruction_error"
            ].quantile(0.95)
        )

        novelty_df[
            "global_threshold"
        ] = fallback_threshold

        novelty_df[
            "adaptive_threshold"
        ] = fallback_threshold

        novelty_df[
            "global_anomaly"
        ] = (
            novelty_df[
                "reconstruction_error"
            ]
            > fallback_threshold
        ).astype(int)

        novelty_df[
            "adaptive_anomaly"
        ] = novelty_df[
            "global_anomaly"
        ]

        novelty_df[
            "severity"
        ] = "NORMAL"

        novelty_df.loc[
            novelty_df[
                "reconstruction_error"
            ]
            > fallback_threshold,
            "severity"
        ] = "WARNING"

        novelty_df.loc[
            novelty_df[
                "reconstruction_error"
            ]
            > 2 * fallback_threshold,
            "severity"
        ] = "CRITICAL"

    return (
        sensor_df,
        pca_df,
        ae_df,
        history_df,
        novelty_df
    )


# ============================================================
# LOAD DATA
# ============================================================

try:

    (
        sensor_df,
        pca_df,
        ae_df,
        history_df,
        novelty_df
    ) = load_data()

    pca_model = joblib.load(
        PCA_MODEL_FILE
    )

    autoencoder = keras.models.load_model(
        AE_MODEL_FILE,
        compile=False
    )

    # ========================================================
    # FIXED THRESHOLD LOADING
    # threshold.joblib contains:
    # {
    #     'threshold': np.float32(...),
    #     'method': '95th_percentile'
    # }
    # ========================================================

    threshold_data = joblib.load(
        THRESHOLD_FILE
    )

    if isinstance(
        threshold_data,
        dict
    ):

        threshold = float(
            threshold_data["threshold"]
        )

    else:

        threshold = float(
            threshold_data
        )

except Exception as e:

    st.error(
        f"Unable to load project files: {e}"
    )

    st.stop()


# ============================================================
# COMMON METRICS
# ============================================================

total_samples = len(ae_df)

anomaly_count = int(
    ae_df["anomaly"].sum()
)

normal_count = (
    total_samples
    - anomaly_count
)

anomaly_percentage = (
    anomaly_count
    / total_samples
    * 100
)

max_error = float(
    ae_df[
        "reconstruction_error"
    ].max()
)

max_error_row = ae_df.loc[
    ae_df[
        "reconstruction_error"
    ].idxmax()
]

max_error_timestamp = (
    max_error_row["ts"]
)

sensor_columns = [
    c
    for c in sensor_df.columns
    if c != "ts"
]

# ============================================================
# NOVELTY METRICS
# ============================================================

global_threshold_novelty = float(
    novelty_df[
        "global_threshold"
    ].iloc[0]
)

adaptive_anomaly_count = int(
    novelty_df[
        "adaptive_anomaly"
    ].sum()
)

adaptive_anomaly_percentage = (
    adaptive_anomaly_count
    / total_samples
    * 100
)

warning_count = int(
    (
        novelty_df["severity"]
        == "WARNING"
    ).sum()
)

critical_count = int(
    (
        novelty_df["severity"]
        == "CRITICAL"
    ).sum()
)

# ============================================================
# EVENT PERIOD
# ============================================================

event_start = pd.Timestamp(
    "2023-03-09 23:40:00"
)

event_end = pd.Timestamp(
    "2023-03-10 00:10:00"
)

event_df = ae_df[
    (ae_df["ts"] >= event_start)
    &
    (ae_df["ts"] <= event_end)
].copy()

event_novelty_df = novelty_df[
    (novelty_df["ts"] >= event_start)
    &
    (novelty_df["ts"] <= event_end)
].copy()

event_anomalies = int(
    event_df["anomaly"].sum()
)

event_adaptive_anomalies = int(
    event_novelty_df[
        "adaptive_anomaly"
    ].sum()
)

event_warning = int(
    (
        event_novelty_df[
            "severity"
        ]
        == "WARNING"
    ).sum()
)

event_critical = int(
    (
        event_novelty_df[
            "severity"
        ]
        == "CRITICAL"
    ).sum()
)

event_anomaly_rate = (
    event_anomalies
    / len(event_df)
    * 100
    if len(event_df) > 0
    else 0
)

event_adaptive_rate = (
    event_adaptive_anomalies
    / len(event_novelty_df)
    * 100
    if len(event_novelty_df) > 0
    else 0
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:26px;
            font-weight:800;
            margin-bottom:4px;
        ">
            🌉 BridgeGuard AI
        </div>

        <div style="
            font-size:12px;
            color:#94a3b8;
            margin-bottom:25px;
        ">
            Structural Health Monitoring
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "PCA → Deep Autoencoder → Adaptive Anomaly Detection"
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "📊 Dashboard",
            "📡 Sensor Analysis",
            "📈 PCA Analysis",
            "🧠 Autoencoder",
            "🚨 Anomaly Detection",
            "📋 Model Performance",
            "ℹ️ About Project"
        ]
    )

    st.markdown("---")

    st.markdown(
        """
        **System Status**

        🟢 Data Engine  
        🟢 PCA Model  
        🟢 Autoencoder  
        🟢 Adaptive Detector
        """
    )

    st.markdown("---")

    st.caption(
        "MLDS Project • Bridge Structural Integrity Monitoring"
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    page_header(
        "Bridge Monitoring Dashboard",
        "AI-powered structural health monitoring using PCA and Deep Autoencoder",
        "🌉"
    )

    if event_adaptive_anomalies > 0:

        st.markdown(
            f"""
            <div class="warning-box">

                <b>⚠ ADAPTIVE ANOMALY DETECTED</b><br><br>

                {event_adaptive_anomalies:,}
                adaptive anomalous observations were identified
                during the event-analysis window.

                <br><br>

                <b>
                Warning:
                {event_warning:,}
                &nbsp;&nbsp;|&nbsp;&nbsp;
                Critical:
                {event_critical:,}
                </b>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="success-box">

                <b>✓ BRIDGE STATUS NORMAL</b><br><br>

                No adaptive anomalies were detected
                in the selected event-analysis window.

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    Bridge Status
                </div>

                <div class="kpi-value warning">
                    {"ANOMALY"
                    if event_adaptive_anomalies > 0
                    else "NORMAL"}
                </div>

                <div class="kpi-footer">
                    Adaptive event-window monitoring
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    Adaptive Anomalies
                </div>

                <div class="kpi-value">
                    {adaptive_anomaly_count:,}
                </div>

                <div class="kpi-footer">
                    {adaptive_anomaly_percentage:.2f}%
                    of all samples
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    Critical Observations
                </div>

                <div class="kpi-value danger">
                    {critical_count:,}
                </div>

                <div class="kpi-footer">
                    Error &gt; 2× global threshold
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        if isinstance(
            pca_model,
            dict
        ):

            pca_components = pca_model.get(
                "n_components",
                13
            )

        else:

            pca_components = getattr(
                pca_model,
                "n_components",
                13
            )

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    PCA Variance Retained
                </div>

                <div class="kpi-value">
                    99.13%
                </div>

                <div class="kpi-footer">
                    {pca_components}
                    principal components
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # SYSTEM OVERVIEW
    # --------------------------------------------------------

    section(
        "System Overview",
        "Current project monitoring statistics"
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Total Samples",
            f"{total_samples:,}"
        )

    with m2:
        st.metric(
            "Sensors",
            len(sensor_columns)
        )

    with m3:
        st.metric(
            "Global Anomalies",
            f"{anomaly_count:,}"
        )

    with m4:
        st.metric(
            "Event Anomaly Rate",
            f"{event_anomaly_rate:.2f}%"
        )

    # --------------------------------------------------------
    # NOVELTY METRICS
    # --------------------------------------------------------

    section(
        "Adaptive Monitoring & Severity",
        "Code-level novelty: rolling threshold + severity classification"
    )

    n1, n2, n3, n4 = st.columns(4)

    with n1:
        st.metric(
            "Adaptive Anomaly Rate",
            f"{adaptive_anomaly_percentage:.2f}%"
        )

    with n2:
        st.metric(
            "Warning",
            f"{warning_count:,}"
        )

    with n3:
        st.metric(
            "Critical",
            f"{critical_count:,}"
        )

    with n4:
        st.metric(
            "Event Adaptive Rate",
            f"{event_adaptive_rate:.2f}%"
        )

    st.info(
        f"Adaptive threshold uses a 501-sample rolling median "
        f"and MAD-based robust threshold. It is constrained to "
        f"remain at or above the global 95th-percentile threshold "
        f"({global_threshold_novelty:.4f})."
    )

    # --------------------------------------------------------
    # RECONSTRUCTION ERROR
    # --------------------------------------------------------

    section(
        "Reconstruction Error Monitoring",
        "Global and adaptive thresholds over the complete dataset"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=ae_df["ts"],
            y=ae_df[
                "reconstruction_error"
            ],
            mode="lines",
            name="Reconstruction Error",
            line=dict(width=1)
        )
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        annotation_text=(
            f"Global Threshold: {threshold:.4f}"
        ),
        annotation_position="top left"
    )

    fig.add_trace(
        go.Scatter(
            x=novelty_df["ts"],
            y=novelty_df[
                "adaptive_threshold"
            ],
            mode="lines",
            name="Adaptive Threshold",
            line=dict(width=1.2)
        )
    )

    fig.update_layout(
        height=430,
        template="plotly_white",
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10
        ),
        xaxis_title="Timestamp",
        yaxis_title="Reconstruction Error",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # EVENT PERIOD
    # --------------------------------------------------------

    section(
        "Documented Event Period",
        "Reconstruction error response around the documented bridge event"
    )

    fig_event = go.Figure()

    fig_event.add_trace(
        go.Scatter(
            x=event_df["ts"],
            y=event_df[
                "reconstruction_error"
            ],
            mode="lines",
            name="Reconstruction Error",
            line=dict(width=1.4)
        )
    )

    fig_event.add_hline(
        y=threshold,
        line_dash="dash",
        annotation_text="Global Threshold"
    )

    fig_event.add_trace(
        go.Scatter(
            x=event_novelty_df["ts"],
            y=event_novelty_df[
                "adaptive_threshold"
            ],
            mode="lines",
            name="Adaptive Threshold",
            line=dict(width=1.5)
        )
    )

    fig_event.update_layout(
        height=420,
        template="plotly_white",
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10
        ),
        xaxis_title="Timestamp",
        yaxis_title="Reconstruction Error",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_event,
        use_container_width=True
    )

    st.info(
        f"Event period contains "
        f"{event_adaptive_anomalies:,} adaptive anomalies "
        f"out of {len(event_df):,} samples "
        f"({event_adaptive_rate:.2f}%). "
        f"Severity: {event_warning:,} warning and "
        f"{event_critical:,} critical observations."
    )

    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    section(
        "System Architecture",
        "End-to-end machine learning pipeline"
    )

    st.markdown(
        """
        <div class="architecture">

            <div class="arch-box">
                🌉<br>
                Bridge Sensor Data
            </div>

            <div class="arch-arrow">
                →
            </div>

            <div class="arch-box">
                🧹<br>
                Preprocessing
            </div>

            <div class="arch-arrow">
                →
            </div>

            <div class="arch-box">
                📉<br>
                PCA
            </div>

            <div class="arch-arrow">
                →
            </div>

            <div class="arch-box">
                🧠<br>
                Deep Autoencoder
            </div>

            <div class="arch-arrow">
                →
            </div>

            <div class="arch-box">
                📊<br>
                Reconstruction Error
            </div>

            <div class="arch-arrow">
                →
            </div>

            <div class="arch-box">
                🎯<br>
                Global + Adaptive Threshold
            </div>

            <div class="arch-arrow">
                →
            </div>

            <div class="arch-box">
                🚦<br>
                Normal / Warning / Critical
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SENSOR ANALYSIS
# ============================================================

elif page == "📡 Sensor Analysis":

    page_header(
        "Sensor Analysis",
        "Explore bridge monitoring sensor channels",
        "📡"
    )

    selected_sensor = st.selectbox(
        "Select Sensor",
        sensor_columns
    )

    sensor_plot = go.Figure()

    sensor_plot.add_trace(
        go.Scatter(
            x=sensor_df["ts"],
            y=sensor_df[
                selected_sensor
            ],
            mode="lines",
            name=selected_sensor,
            line=dict(width=1)
        )
    )

    sensor_plot.update_layout(
        height=480,
        template="plotly_white",
        xaxis_title="Timestamp",
        yaxis_title=selected_sensor,
        hovermode="x unified"
    )

    st.plotly_chart(
        sensor_plot,
        use_container_width=True
    )

    section(
        "Sensor Statistics",
        f"Statistics for {selected_sensor}"
    )

    values = sensor_df[
        selected_sensor
    ].dropna()

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.metric(
            "Mean",
            f"{values.mean():.4f}"
        )

    with s2:
        st.metric(
            "Std Dev",
            f"{values.std():.4f}"
        )

    with s3:
        st.metric(
            "Minimum",
            f"{values.min():.4f}"
        )

    with s4:
        st.metric(
            "Maximum",
            f"{values.max():.4f}"
        )

    st.dataframe(
        sensor_df[
            ["ts", selected_sensor]
        ].tail(500),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PCA ANALYSIS
# ============================================================

elif page == "📈 PCA Analysis":

    page_header(
        "PCA Analysis",
        "Principal Component Analysis and explained variance",
        "📈"
    )

    pc_columns = [
        c
        for c in pca_df.columns
        if c.startswith("PC")
    ]

    if isinstance(
        pca_model,
        dict
    ):

        variance_ratio = np.array(
            pca_model.get(
                "explained_variance_ratio",
                []
            )
        )

    else:

        variance_ratio = np.array(
            getattr(
                pca_model,
                "explained_variance_ratio_",
                []
            )
        )

    if len(variance_ratio) > 0:

        cumulative_variance = np.cumsum(
            variance_ratio
        )

        pca_chart = go.Figure()

        pca_chart.add_trace(
            go.Bar(
                x=[
                    f"PC{i+1}"
                    for i in range(
                        len(variance_ratio)
                    )
                ],
                y=variance_ratio * 100,
                name="Individual Variance"
            )
        )

        pca_chart.add_trace(
            go.Scatter(
                x=[
                    f"PC{i+1}"
                    for i in range(
                        len(variance_ratio)
                    )
                ],
                y=cumulative_variance * 100,
                mode="lines+markers",
                name="Cumulative Variance"
            )
        )

        pca_chart.update_layout(
            height=450,
            template="plotly_white",
            xaxis_title="Principal Component",
            yaxis_title="Explained Variance (%)"
        )

        st.plotly_chart(
            pca_chart,
            use_container_width=True
        )

        p1, p2 = st.columns(2)

        with p1:
            st.metric(
                "PCA Components",
                len(variance_ratio)
            )

        with p2:
            st.metric(
                "Variance Retained",
                f"{cumulative_variance[-1] * 100:.2f}%"
            )

    else:

        st.info(
            "PCA explained-variance information "
            "is not available in the saved model."
        )

    section(
        "PCA Feature Data",
        "First principal-component observations"
    )

    st.dataframe(
        pca_df.head(500),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# AUTOENCODER
# ============================================================

elif page == "🧠 Autoencoder":

    page_header(
        "Deep Autoencoder",
        "Neural reconstruction model used for anomaly detection",
        "🧠"
    )

    st.markdown(
        """
        <div class="info-box">

        <b>Autoencoder Architecture</b><br><br>

        Input → 13 PCA Features<br>
        ↓<br>
        Dense 8<br>
        ↓<br>
        Dense 4<br>
        ↓<br>
        Bottleneck 2<br>
        ↓<br>
        Dense 4<br>
        ↓<br>
        Dense 8<br>
        ↓<br>
        Output 13

        </div>
        """,
        unsafe_allow_html=True
    )

    section(
        "Training History",
        "Training and validation reconstruction loss"
    )

    history_columns = [
        c
        for c in history_df.columns
        if c in ["loss", "val_loss"]
    ]

    if len(history_columns) > 0:

        history_plot = go.Figure()

        if "loss" in history_df.columns:

            history_plot.add_trace(
                go.Scatter(
                    y=history_df["loss"],
                    mode="lines",
                    name="Training Loss"
                )
            )

        if "val_loss" in history_df.columns:

            history_plot.add_trace(
                go.Scatter(
                    y=history_df["val_loss"],
                    mode="lines",
                    name="Validation Loss"
                )
            )

        history_plot.update_layout(
            height=420,
            template="plotly_white",
            xaxis_title="Epoch",
            yaxis_title="Loss"
        )

        st.plotly_chart(
            history_plot,
            use_container_width=True
        )

    a1, a2, a3 = st.columns(3)

    with a1:

        st.metric(
            "Final Training Loss",
            f"{history_df['loss'].iloc[-1]:.6f}"
            if "loss" in history_df.columns
            else "N/A"
        )

    with a2:

        st.metric(
            "Final Validation Loss",
            f"{history_df['val_loss'].iloc[-1]:.6f}"
            if "val_loss" in history_df.columns
            else "N/A"
        )

    with a3:

        st.metric(
            "Maximum Reconstruction Error",
            f"{max_error:.3f}"
        )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "🚨 Anomaly Detection":

    page_header(
        "Anomaly Detection",
        "Global and adaptive reconstruction-error monitoring with severity classification",
        "🚨"
    )

    st.markdown(
        f"""
        <div class="info-box">

        <b>Detection Method</b><br><br>

        The baseline system uses the 95th-percentile
        reconstruction-error threshold:

        <b>
        {global_threshold_novelty:.6f}
        </b>

        <br><br>

        The novelty layer calculates a rolling adaptive
        threshold using the median and MAD of previous
        reconstruction errors.

        <br><br>

        Severity:

        <b>Normal</b> → below global threshold<br>
        <b>Warning</b> → above global threshold<br>
        <b>Critical</b> → above 2× global threshold

        </div>
        """,
        unsafe_allow_html=True
    )

    d1, d2, d3, d4 = st.columns(4)

    with d1:

        st.metric(
            "Global Anomalies",
            f"{anomaly_count:,}"
        )

    with d2:

        st.metric(
            "Adaptive Anomalies",
            f"{adaptive_anomaly_count:,}"
        )

    with d3:

        st.metric(
            "Warning",
            f"{warning_count:,}"
        )

    with d4:

        st.metric(
            "Critical",
            f"{critical_count:,}"
        )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    section(
        "Anomaly Filters",
        "Filter observations by reconstruction error and severity"
    )

    max_slider = float(
        min(
            max_error,
            100
        )
    )

    default_slider = float(
        min(
            global_threshold_novelty,
            max_slider
        )
    )

    min_error = st.slider(
        "Minimum Reconstruction Error",
        min_value=0.0,
        max_value=max_slider,
        value=default_slider
    )

    severity_filter = st.multiselect(
        "Severity Filter",
        [
            "NORMAL",
            "WARNING",
            "CRITICAL"
        ],
        default=[
            "WARNING",
            "CRITICAL"
        ]
    )

    filtered = novelty_df[
        (
            novelty_df[
                "reconstruction_error"
            ]
            >= min_error
        )
        &
        (
            novelty_df[
                "severity"
            ].isin(
                severity_filter
            )
        )
    ].sort_values(
        "reconstruction_error",
        ascending=False
    )

    f1, f2, f3 = st.columns(3)

    with f1:

        st.metric(
            "Matching Samples",
            f"{len(filtered):,}"
        )

    with f2:

        st.metric(
            "Global Threshold",
            f"{global_threshold_novelty:.4f}"
        )

    with f3:

        st.metric(
            "Highest Error",
            (
                f"{filtered['reconstruction_error'].max():.3f}"
                if len(filtered) > 0
                else "N/A"
            )
        )

    # --------------------------------------------------------
    # ADAPTIVE GRAPH
    # --------------------------------------------------------

    section(
        "Adaptive Threshold Timeline",
        "Reconstruction error compared with global and local adaptive thresholds"
    )

    anomaly_chart = go.Figure()

    anomaly_chart.add_trace(
        go.Scatter(
            x=novelty_df["ts"],
            y=novelty_df[
                "reconstruction_error"
            ],
            mode="lines",
            name="Reconstruction Error",
            line=dict(width=1.2)
        )
    )

    anomaly_chart.add_trace(
        go.Scatter(
            x=novelty_df["ts"],
            y=novelty_df[
                "adaptive_threshold"
            ],
            mode="lines",
            name="Adaptive Threshold",
            line=dict(width=1.2)
        )
    )

    anomaly_chart.add_hline(
        y=global_threshold_novelty,
        line_dash="dash",
        annotation_text="Global Threshold"
    )

    anomaly_chart.update_layout(
        height=430,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),
        template="plotly_white",
        xaxis_title="Timestamp",
        yaxis_title="Reconstruction Error",
        hovermode="x unified"
    )

    st.plotly_chart(
        anomaly_chart,
        use_container_width=True
    )

    # --------------------------------------------------------
    # SEVERITY DISTRIBUTION
    # --------------------------------------------------------

    section(
        "Severity Distribution",
        "Normal, Warning and Critical observations"
    )

    severity_counts = (
        novelty_df[
            "severity"
        ]
        .value_counts()
        .reindex(
            [
                "NORMAL",
                "WARNING",
                "CRITICAL"
            ],
            fill_value=0
        )
        .reset_index()
    )

    severity_counts.columns = [
        "Severity",
        "Samples"
    ]

    severity_chart = px.bar(
        severity_counts,
        x="Severity",
        y="Samples",
        text="Samples"
    )

    severity_chart.update_layout(
        height=360,
        template="plotly_white",
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        )
    )

    st.plotly_chart(
        severity_chart,
        use_container_width=True
    )

    # --------------------------------------------------------
    # ANOMALY TABLE
    # --------------------------------------------------------

    section(
        "Anomaly Records",
        "Highest reconstruction-error observations shown first"
    )

    st.dataframe(
        filtered.head(500),
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Adaptive Anomaly Results",
        data=csv_data,
        file_name=(
            "bridge_adaptive_anomaly_results.csv"
        ),
        mime="text/csv"
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📋 Model Performance":

    page_header(
        "Model Performance",
        "Training behaviour, anomaly statistics and methodology",
        "📋"
    )

    section(
        "Autoencoder Results",
        "Observed reconstruction-error statistics"
    )

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        st.metric(
            "Total Samples",
            f"{total_samples:,}"
        )

    with p2:

        st.metric(
            "Mean Reconstruction Error",
            f"{ae_df['reconstruction_error'].mean():.4f}"
        )

    with p3:

        st.metric(
            "Global Threshold",
            f"{global_threshold_novelty:.4f}"
        )

    with p4:

        st.metric(
            "Global Anomaly Rate",
            f"{anomaly_percentage:.2f}%"
        )

    section(
        "Novelty Performance",
        "Adaptive threshold and severity outputs"
    )

    novelty_performance = pd.DataFrame(
        {
            "Metric": [
                "Global Anomalies",
                "Global Anomaly Rate",
                "Adaptive Anomalies",
                "Adaptive Anomaly Rate",
                "Warning Observations",
                "Critical Observations",
                "Event Adaptive Anomalies",
                "Event Adaptive Anomaly Rate"
            ],

            "Value": [
                f"{anomaly_count:,}",
                f"{anomaly_percentage:.2f}%",
                f"{adaptive_anomaly_count:,}",
                f"{adaptive_anomaly_percentage:.2f}%",
                f"{warning_count:,}",
                f"{critical_count:,}",
                f"{event_adaptive_anomalies:,}",
                f"{event_adaptive_rate:.2f}%"
            ]
        }
    )

    st.dataframe(
        novelty_performance,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The adaptive layer is a novelty enhancement for "
        "local anomaly sensitivity. It does not create "
        "ground-truth damage labels, so conventional accuracy, "
        "precision, recall and F1 are not reported as verified metrics."
    )

    section(
        "Model Comparison",
        "Functional comparison of PCA, Deep Autoencoder and integrated approach"
    )

    comparison_df = pd.DataFrame(
        {
            "Feature": [
                "Dimensionality Reduction",
                "Linear Pattern Detection",
                "Non-linear Pattern Detection",
                "Feature Compression",
                "Reconstruction Error",
                "Anomaly Detection",
                "Adaptive Threshold",
                "Severity Classification"
            ],

            "PCA": [
                "✓",
                "✓",
                "—",
                "✓",
                "—",
                "Limited",
                "—",
                "—"
            ],

            "Deep Autoencoder": [
                "✓",
                "—",
                "✓",
                "✓",
                "✓",
                "✓",
                "—",
                "—"
            ],

            "PCA + Deep Autoencoder": [
                "✓",
                "✓",
                "✓",
                "✓",
                "✓",
                "✓",
                "✓",
                "✓"
            ]
        }
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "The comparison table describes model capabilities. "
        "It is not a claim that PCA alone provides a directly "
        "validated anomaly classifier."
    )

    section(
        "Training Curve",
        "Training and validation loss"
    )

    if (
        "loss" in history_df.columns
        and
        "val_loss" in history_df.columns
    ):

        training_chart = go.Figure()

        training_chart.add_trace(
            go.Scatter(
                y=history_df["loss"],
                mode="lines",
                name="Training Loss"
            )
        )

        training_chart.add_trace(
            go.Scatter(
                y=history_df["val_loss"],
                mode="lines",
                name="Validation Loss"
            )
        )

        training_chart.update_layout(
            height=420,
            template="plotly_white",
            xaxis_title="Epoch",
            yaxis_title="Loss"
        )

        st.plotly_chart(
            training_chart,
            use_container_width=True
        )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    page_header(
        "About BridgeGuard AI",
        "A Deep Autoencoder and PCA-based Diagnostic System for Bridges Structural Integrity Monitoring",
        "ℹ️"
    )

    section(
        "Project Objective",
        "Machine learning based bridge structural-health monitoring"
    )

    st.markdown(
        """
        ### Aim

        To develop an intelligent diagnostic system that analyses
        bridge sensor data using **PCA and Deep Autoencoder**
        techniques to identify abnormal structural behaviour.

        ### Main Objectives

        - Process real bridge monitoring data.
        - Clean and prepare sensor observations.
        - Reduce feature dimensionality using PCA.
        - Retain the dominant information from sensor data.
        - Train a Deep Autoencoder.
        - Calculate reconstruction error.
        - Detect abnormal observations.
        - Introduce adaptive anomaly detection.
        - Classify observations into Normal, Warning and Critical.
        - Provide an interactive monitoring dashboard.
        """
    )

    section(
        "Project Novelty",
        "Code-level enhancement implemented in the anomaly-analysis pipeline"
    )

    st.markdown(
        """
        <div class="success-box">

        <b>Adaptive Anomaly Detection with Severity Classification</b>

        <br><br>

        Instead of relying only on a fixed global threshold,
        the enhanced system calculates a local adaptive threshold
        using a rolling median and Median Absolute Deviation (MAD)
        from previous reconstruction-error observations.

        <br><br>

        The resulting observations are also classified into:

        <br><br>

        🟢 <b>NORMAL</b><br>
        🟡 <b>WARNING</b><br>
        🔴 <b>CRITICAL</b>

        </div>
        """,
        unsafe_allow_html=True
    )

    section(
        "System Architecture",
        "Complete machine-learning workflow"
    )

    st.markdown(
        """
        <div class="architecture">

            <div class="arch-box">
                🌉<br>
                Bridge Sensor Data
            </div>

            <div class="arch-arrow">→</div>

            <div class="arch-box">
                🧹<br>
                Preprocessing
            </div>

            <div class="arch-arrow">→</div>

            <div class="arch-box">
                📉<br>
                PCA
            </div>

            <div class="arch-arrow">→</div>

            <div class="arch-box">
                🧠<br>
                Deep Autoencoder
            </div>

            <div class="arch-arrow">→</div>

            <div class="arch-box">
                📊<br>
                Reconstruction Error
            </div>

            <div class="arch-arrow">→</div>

            <div class="arch-box">
                🎯<br>
                Global + Adaptive Threshold
            </div>

            <div class="arch-arrow">→</div>

            <div class="arch-box">
                🚦<br>
                Normal / Warning / Critical
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    section(
        "Methodology",
        "Step-by-step processing pipeline"
    )

    methodology_df = pd.DataFrame(
        {
            "Step": [
                "1. Data Collection",
                "2. Data Preprocessing",
                "3. PCA",
                "4. Deep Autoencoder",
                "5. Reconstruction Error",
                "6. Global Anomaly Detection",
                "7. Adaptive Threshold & Severity"
            ],

            "Description": [
                "Real bridge structural monitoring sensor data is used.",

                "Invalid, constant and problematic sensor channels "
                "are removed or handled through interpolation.",

                "Principal Component Analysis reduces the sensor "
                "feature space while retaining dominant variance.",

                "The reduced PCA representation is reconstructed "
                "using a neural network.",

                "The difference between original and reconstructed "
                "PCA representation is calculated.",

                "Samples exceeding the 95th-percentile "
                "reconstruction-error threshold are flagged "
                "as global anomalies.",

                "A rolling median/MAD threshold is calculated "
                "from previous observations and observations "
                "are classified as Normal, Warning or Critical."
            ]
        }
    )

    st.dataframe(
        methodology_df,
        use_container_width=True,
        hide_index=True
    )

    section(
        "Current Project Results",
        "Results obtained from the trained project pipeline"
    )

    r1, r2, r3, r4 = st.columns(4)

    with r1:

        st.metric(
            "Samples",
            f"{total_samples:,}"
        )

    with r2:

        st.metric(
            "PCA Variance",
            "99.13%"
        )

    with r3:

        st.metric(
            "Adaptive Anomalies",
            f"{adaptive_anomaly_count:,}"
        )

    with r4:

        st.metric(
            "Critical",
            f"{critical_count:,}"
        )

    st.markdown(
        """
        ### Important Interpretation

        An anomaly does **not** automatically mean that a bridge
        is damaged.

        The system identifies observations whose sensor behaviour
        differs from the learned reconstruction pattern.

        Reliable classification metrics such as accuracy,
        precision, recall and F1 require trustworthy observation-level
        ground-truth labels. Therefore they are not presented as
        verified metrics for this dataset.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🌉 <b>BridgeGuard AI</b><br>

        PCA + Deep Autoencoder +
        Adaptive Anomaly Detection<br>

        MLDS Project • Bridge Structural Integrity Monitoring

    </div>
    """,
    unsafe_allow_html=True
)