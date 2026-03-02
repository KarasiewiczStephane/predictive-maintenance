"""Streamlit dashboard for predictive maintenance visualization.

Displays model performance, sensor data trends, failure prediction
distributions, and threshold analysis using synthetic demo data.

Run with: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SENSOR_NAMES = ["temperature", "vibration", "pressure", "rpm"]


def generate_model_comparison(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic model comparison metrics."""
    rng = np.random.default_rng(seed)
    models = ["Random Forest", "XGBoost"]
    rows = []
    for model in models:
        base = 0.88 if model == "XGBoost" else 0.85
        rows.append(
            {
                "model": model,
                "precision": round(base + rng.uniform(-0.03, 0.03), 4),
                "recall": round(base - rng.uniform(0.02, 0.06), 4),
                "f1": round(base + rng.uniform(-0.02, 0.02), 4),
                "roc_auc": round(base + rng.uniform(0.02, 0.06), 4),
            }
        )
    return pd.DataFrame(rows)


def generate_sensor_data(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic sensor readings over time."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-11-01", periods=200, freq="h")
    data = {"timestamp": timestamps}
    for sensor in SENSOR_NAMES:
        base = {"temperature": 65, "vibration": 30, "pressure": 100, "rpm": 1500}[sensor]
        noise = rng.normal(0, base * 0.05, size=len(timestamps))
        trend = np.linspace(0, base * 0.1, len(timestamps))
        values = base + noise + trend
        data[sensor] = np.round(values, 2)
    data["failure"] = [bool(rng.random() > 0.95) for _ in range(len(timestamps))]
    return pd.DataFrame(data)


def generate_prediction_distribution(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic prediction probability distribution."""
    rng = np.random.default_rng(seed)
    healthy_probs = rng.beta(2, 8, size=300)
    failure_probs = rng.beta(6, 3, size=100)
    rows = []
    for p in healthy_probs:
        rows.append({"probability": round(p, 4), "actual": "Healthy"})
    for p in failure_probs:
        rows.append({"probability": round(p, 4), "actual": "Failure"})
    return pd.DataFrame(rows)


def generate_threshold_analysis(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic threshold vs metric analysis."""
    rng = np.random.default_rng(seed)
    thresholds = np.arange(0.1, 0.91, 0.05)
    rows = []
    for t in thresholds:
        precision = min(0.5 + t * 0.5 + rng.uniform(-0.03, 0.03), 1.0)
        recall = max(1.0 - t * 0.7 + rng.uniform(-0.03, 0.03), 0.1)
        rows.append(
            {
                "threshold": round(t, 2),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(2 * precision * recall / (precision + recall), 4),
            }
        )
    return pd.DataFrame(rows)


def render_header() -> None:
    """Render the dashboard header."""
    st.title("Predictive Maintenance Dashboard")
    st.caption(
        "Sensor-based failure prediction using Random Forest and XGBoost "
        "with real-time monitoring and threshold optimization"
    )


def render_summary_metrics(models_df: pd.DataFrame, sensor_df: pd.DataFrame) -> None:
    """Render top-level summary metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    best = models_df.loc[models_df["roc_auc"].idxmax()]
    col1.metric("Best Model", best["model"])
    col2.metric("ROC-AUC", f"{best['roc_auc']:.4f}")
    failure_rate = sensor_df["failure"].mean()
    col3.metric("Failure Rate", f"{failure_rate:.1%}")
    col4.metric("Readings", f"{len(sensor_df):,}")


def render_model_comparison(models_df: pd.DataFrame) -> None:
    """Render model comparison grouped bar chart."""
    st.subheader("Model Performance Comparison")
    fig = go.Figure()
    for metric in ["precision", "recall", "f1", "roc_auc"]:
        fig.add_trace(
            go.Bar(
                name=metric.upper().replace("_", " "),
                x=models_df["model"],
                y=models_df[metric],
                text=models_df[metric].apply(lambda x: f"{x:.3f}"),
                textposition="auto",
            )
        )
    fig.update_layout(
        barmode="group",
        yaxis={"range": [0.7, 1.0]},
        height=400,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_sensor_trends(sensor_df: pd.DataFrame) -> None:
    """Render sensor data time series."""
    st.subheader("Sensor Readings Over Time")
    sensor = st.selectbox("Select sensor:", SENSOR_NAMES)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sensor_df["timestamp"],
            y=sensor_df[sensor],
            mode="lines",
            name=sensor.capitalize(),
            line={"color": "#2196F3", "width": 1.5},
        )
    )
    failure_points = sensor_df[sensor_df["failure"]]
    if len(failure_points) > 0:
        fig.add_trace(
            go.Scatter(
                x=failure_points["timestamp"],
                y=failure_points[sensor],
                mode="markers",
                name="Failure Events",
                marker={"color": "red", "size": 8, "symbol": "x"},
            )
        )
    fig.update_layout(
        yaxis_title=sensor.capitalize(),
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_prediction_distribution(pred_df: pd.DataFrame) -> None:
    """Render prediction probability distributions."""
    st.subheader("Prediction Probability Distribution")
    fig = px.histogram(
        pred_df,
        x="probability",
        color="actual",
        nbins=40,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map={"Healthy": "#4CAF50", "Failure": "#F44336"},
    )
    fig.update_layout(
        xaxis_title="Failure Probability",
        yaxis_title="Count",
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_threshold_analysis(thresh_df: pd.DataFrame) -> None:
    """Render threshold analysis curves."""
    st.subheader("Threshold Optimization")
    fig = go.Figure()
    for metric in ["precision", "recall", "f1"]:
        fig.add_trace(
            go.Scatter(
                x=thresh_df["threshold"],
                y=thresh_df[metric],
                mode="lines+markers",
                name=metric.capitalize(),
            )
        )
    optimal = thresh_df.loc[thresh_df["f1"].idxmax()]
    fig.add_vline(
        x=optimal["threshold"],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Optimal: {optimal['threshold']:.2f}",
    )
    fig.update_layout(
        xaxis_title="Decision Threshold",
        yaxis_title="Score",
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Main dashboard entry point."""
    render_header()

    models_df = generate_model_comparison()
    sensor_df = generate_sensor_data()
    pred_df = generate_prediction_distribution()
    thresh_df = generate_threshold_analysis()

    render_summary_metrics(models_df, sensor_df)
    st.markdown("---")

    render_model_comparison(models_df)
    render_sensor_trends(sensor_df)

    col_left, col_right = st.columns(2)
    with col_left:
        render_prediction_distribution(pred_df)
    with col_right:
        render_threshold_analysis(thresh_df)


if __name__ == "__main__":
    main()
