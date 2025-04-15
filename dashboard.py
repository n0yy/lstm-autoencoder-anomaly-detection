import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import time
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from typing import Dict, List, Optional

from config.config import DATA_PATH, CATEGORICAL_COLS, SEQUENCE_LENGTH
from models.autoencoder import create_lstm_autoencoder
from utils.preprocessing import create_sequences

# Inisialisasi dashboard
app = dash.Dash(__name__, title="Anomaly Detection Dashboard")

# Tentukan layout dashboard
app.layout = html.Div(
    [
        html.H1("Monitoring Anomali Mesin", style={"textAlign": "center"}),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Status Sistem"),
                        html.Div(
                            id="status-container",
                            children=[
                                html.Div(
                                    [
                                        html.H4("Total Data Points"),
                                        html.Div(id="total-data-points"),
                                    ],
                                    className="status-box",
                                ),
                                html.Div(
                                    [
                                        html.H4("Anomali Terdeteksi"),
                                        html.Div(id="anomaly-count"),
                                    ],
                                    className="status-box",
                                ),
                                html.Div(
                                    [
                                        html.H4("Anomaly Rate"),
                                        html.Div(id="anomaly-rate"),
                                    ],
                                    className="status-box",
                                ),
                                html.Div(
                                    [
                                        html.H4("Last Update"),
                                        html.Div(id="last-update"),
                                    ],
                                    className="status-box",
                                ),
                            ],
                        ),
                    ],
                    className="status-container",
                ),
                html.Div(
                    [
                        html.H3("Time Series & Reconstruction Error"),
                        dcc.Graph(id="time-series-graph"),
                        html.Div(
                            [
                                html.Label("Pilih Fitur:"),
                                dcc.Dropdown(
                                    id="feature-dropdown",
                                    value=None,
                                    clearable=False,
                                ),
                            ]
                        ),
                    ],
                    className="graph-container",
                ),
            ],
            className="row",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Distribusi Anomali per Line"),
                        dcc.Graph(id="line-distribution"),
                    ],
                    className="graph-container",
                ),
                html.Div(
                    [html.H3("Feature Importance"), dcc.Graph(id="feature-importance")],
                    className="graph-container",
                ),
            ],
            className="row",
        ),
        html.Div(
            [html.H3("Anomaly Heatmap"), dcc.Graph(id="anomaly-heatmap")],
            className="full-width",
        ),
        dcc.Interval(
            id="interval-component",
            interval=5 * 1000,  # dalam milidetik (5 detik)
            n_intervals=0,
        ),
    ],
    className="dashboard-container",
)


def load_model():
    """Load model dan hasil training."""
    try:
        model_path = "models/saved/model.h5"
        scaler_path = "models/saved/scaler.joblib"
        data_columns_path = "models/saved/data_columns.joblib"
        results_path = "results/training_results.joblib"

        if not all(
            os.path.exists(path)
            for path in [model_path, scaler_path, data_columns_path, results_path]
        ):
            print("Model not found, running in data visualization mode only")
            return None, None, None, None

        from tensorflow.keras.models import load_model

        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        data_columns = joblib.load(data_columns_path)
        results = joblib.load(results_path)

        return model, scaler, data_columns, results

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None


def get_reconstruction_error_threshold() -> Optional[float]:
    """Get reconstruction error threshold from saved results."""
    try:
        results_path = "results/training_results.joblib"
        if os.path.exists(results_path):
            results = joblib.load(results_path)
            return results.get("threshold")
        return None
    except Exception as e:
        print(f"Error loading threshold: {e}")
        return None


@app.callback(
    [
        Output("total-data-points", "children"),
        Output("anomaly-count", "children"),
        Output("anomaly-rate", "children"),
        Output("last-update", "children"),
        Output("time-series-graph", "figure"),
        Output("feature-dropdown", "options"),
        Output("line-distribution", "figure"),
        Output("feature-importance", "figure"),
        Output("anomaly-heatmap", "figure"),
    ],
    [Input("interval-component", "n_intervals"), Input("feature-dropdown", "value")],
)
def update_dashboard(n_intervals, selected_feature):
    """Update dashboard dengan data terbaru."""
    try:
        # Load data
        data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        total_points = len(data)
        last_update = data["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S")

        # Load model dan hasil
        model, scaler, data_columns, results = load_model()

        if model is None:
            # Mode visualisasi data saja
            feature_options = [
                {"label": col, "value": col}
                for col in data.columns
                if col not in ["timestamp"] + CATEGORICAL_COLS
            ]

            # Buat figure kosong untuk feature importance
            feature_importance_fig = go.Figure()
            feature_importance_fig.update_layout(
                title="Feature Importance",
                xaxis_title="Features",
                yaxis_title="Importance Score",
                height=400,
            )

            return (
                total_points,
                0,
                "0%",
                last_update,
                go.Figure(),  # Time series figure
                feature_options,
                go.Figure(),  # Line distribution
                feature_importance_fig,
                go.Figure(),  # Anomaly heatmap
            )

        # Hitung anomali
        threshold = get_reconstruction_error_threshold()
        if threshold is None:
            threshold = results["threshold"]

        # Buat sequences
        scaled_data = scaler.transform(
            data.drop(columns=["timestamp"] + CATEGORICAL_COLS)
        )
        sequences = create_sequences(scaled_data, SEQUENCE_LENGTH)

        # Prediksi
        reconstructions = model.predict(sequences)
        mse = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
        anomalies = mse > threshold

        anomaly_count = np.sum(anomalies)
        anomaly_rate = f"{(anomaly_count / total_points * 100):.2f}%"

        # Update feature dropdown
        feature_options = [{"label": col, "value": col} for col in data_columns]

        # Buat visualisasi
        time_series_fig = create_time_series_plot(data, selected_feature, anomalies)
        line_distribution_fig = create_line_distribution_plot(data, anomalies)
        feature_importance_fig = create_feature_importance_plot(
            results["feature_importance"]
        )

        # Anomaly heatmap
        # Create a pivot table for the heatmap
        heatmap_data = None
        if "line" in data.columns and len(data) > 0:
            # Get the most recent 30 minutes of data or all if less
            recent_data = data.tail(min(len(data), 30))

            # Check if there are any anomalies
            if np.sum(anomalies[-len(recent_data) :]) > 0:
                # Create a pivot table with lines as rows and timestamps as columns
                recent_data["anomaly"] = anomalies[-len(recent_data) :]
                pivot_data = pd.pivot_table(
                    recent_data,
                    values="anomaly",
                    index="line",
                    columns="timestamp",
                    aggfunc=np.max,
                    fill_value=0,
                )

                heatmap_data = pivot_data.values
                x_labels = [ts.strftime("%H:%M:%S") for ts in pivot_data.columns]
                y_labels = pivot_data.index.tolist()
            else:
                # Create an empty heatmap if no anomalies
                unique_lines = sorted(recent_data["line"].unique())
                heatmap_data = np.zeros((len(unique_lines), 5))
                x_labels = [""] * 5
                y_labels = unique_lines

        else:
            heatmap_data = np.zeros((1, 5))
            x_labels = [""] * 5
            y_labels = ["No Data"]

        fig_heatmap = px.imshow(
            heatmap_data,
            labels=dict(x="Timestamp", y="Line", color="Anomaly"),
            x=x_labels,
            y=y_labels,
            color_continuous_scale=[[0, "green"], [1, "red"]],
            aspect="auto",
        )
        fig_heatmap.update_layout(coloraxis_showscale=False)

        return (
            total_points,
            anomaly_count,
            anomaly_rate,
            last_update,
            time_series_fig,
            feature_options,
            line_distribution_fig,
            feature_importance_fig,
            fig_heatmap,
        )

    except Exception as e:
        print(f"Error updating dashboard: {e}")
        return (
            "Error",
            "Error",
            "Error",
            "Error",
            go.Figure(),
            [],
            go.Figure(),
            go.Figure(),
            go.Figure(),
        )


def create_time_series_plot(
    data: pd.DataFrame, selected_feature: str, anomalies: np.ndarray
) -> go.Figure:
    """Buat plot time series untuk fitur yang dipilih."""
    fig = go.Figure()

    if selected_feature and selected_feature in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data["timestamp"],
                y=data[selected_feature],
                name=selected_feature,
                mode="lines",
            )
        )

        # Highlight anomali
        anomaly_indices = np.where(anomalies)[0]
        for idx in anomaly_indices:
            fig.add_vrect(
                x0=data["timestamp"].iloc[idx],
                x1=data["timestamp"].iloc[idx + 1],
                fillcolor="red",
                opacity=0.2,
                line_width=0,
            )

    fig.update_layout(
        title=f"Time Series - {selected_feature}",
        xaxis_title="Timestamp",
        yaxis_title="Value",
        height=400,
    )

    return fig


def create_line_distribution_plot(
    data: pd.DataFrame, anomalies: np.ndarray
) -> go.Figure:
    """Buat plot distribusi anomali per line."""
    if "line" not in data.columns:
        return go.Figure()

    anomaly_data = data[anomalies]
    line_counts = anomaly_data["line"].value_counts()

    fig = go.Figure(
        go.Bar(
            x=line_counts.index,
            y=line_counts.values,
            text=line_counts.values,
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Distribusi Anomali per Line",
        xaxis_title="Line",
        yaxis_title="Jumlah Anomali",
        height=400,
    )

    return fig


def create_feature_importance_plot(feature_importance: Dict[str, float]) -> go.Figure:
    """Buat plot feature importance."""
    if not feature_importance:
        return go.Figure()

    features = list(feature_importance.keys())
    importance = list(feature_importance.values())

    fig = go.Figure(
        go.Bar(
            x=features,
            y=importance,
            text=np.round(importance, 3),
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Features",
        yaxis_title="Importance Score",
        height=400,
    )

    return fig


# CSS style
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .dashboard-container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                font-family: Arial, sans-serif;
            }
            
            .row {
                display: flex;
                margin-bottom: 20px;
            }
            
            .graph-container {
                flex: 1;
                background: white;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                padding: 15px;
                margin: 10px;
            }
            
            .status-container {
                flex: 1;
                background: white;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                padding: 15px;
                margin: 10px;
            }
            
            .status-box {
                display: inline-block;
                width: 45%;
                margin: 5px;
                padding: 10px;
                background: #f9f9f9;
                border-radius: 5px;
                text-align: center;
            }
            
            .full-width {
                width: 100%;
                background: white;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                padding: 15px;
                margin: 10px;
            }
            
            h1, h3, h4 {
                margin-top: 0;
                color: #333;
            }
            
            #total-data-points, #anomaly-count, #anomaly-rate {
                font-size: 24px;
                font-weight: bold;
                color: #0066cc;
            }
            
            #anomaly-count {
                color: #cc0000;
            }
            
            #last-update {
                font-size: 18px;
                color: #666;
            }
            
            @media (max-width: 900px) {
                .row {
                    flex-direction: column;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
