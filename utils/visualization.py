import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import N_EXAMPLES, N_FEATURES_TO_SHOW


def plot_training_history(history) -> None:
    """
    Plot history training dengan metrik tambahan.

    Args:
        history: History object dari model.fit()
    """
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Loss", "MAE"))

    # Plot loss
    fig.add_trace(
        go.Scatter(
            y=history.history["loss"],
            name="Training Loss",
            mode="lines",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            y=history.history["val_loss"],
            name="Validation Loss",
            mode="lines",
        ),
        row=1,
        col=1,
    )

    # Plot MAE
    fig.add_trace(
        go.Scatter(
            y=history.history["mae"],
            name="Training MAE",
            mode="lines",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            y=history.history["val_mae"],
            name="Validation MAE",
            mode="lines",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Training History",
        height=800,
        showlegend=True,
    )

    fig.show()


def plot_reconstruction_error_distribution(
    train_mse: np.ndarray,
    test_mse: np.ndarray,
    threshold: float,
) -> None:
    """
    Plot distribusi error rekonstruksi.

    Args:
        train_mse: MSE training
        test_mse: MSE test
        threshold: Threshold anomali
    """
    fig = go.Figure()

    # Plot training error
    fig.add_trace(
        go.Histogram(
            x=train_mse,
            name="Training Error",
            opacity=0.75,
            nbinsx=50,
        )
    )

    # Plot test error
    fig.add_trace(
        go.Histogram(
            x=test_mse,
            name="Test Error",
            opacity=0.75,
            nbinsx=50,
        )
    )

    # Tambahkan threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Distribution of Reconstruction Errors",
        xaxis_title="Reconstruction Error",
        yaxis_title="Count",
        barmode="overlay",
        height=600,
    )

    fig.show()


def plot_feature_importance(feature_importance: Dict[str, float]) -> None:
    """
    Plot feature importance.

    Args:
        feature_importance: Dictionary feature importance
    """
    # Ambil top N features
    top_features = dict(list(feature_importance.items())[:N_FEATURES_TO_SHOW])

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=list(top_features.keys()),
            y=list(top_features.values()),
            text=np.round(list(top_features.values()), 3),
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Top Feature Importance",
        xaxis_title="Features",
        yaxis_title="Importance Score",
        height=600,
    )

    fig.show()


def plot_anomaly_sample(sample, sample_recon, mse):
    plt.figure(figsize=(12, 6))
    plt.plot(sample, label="Original")
    plt.plot(sample_recon, label="Reconstructed")
    plt.title(f"Contoh Anomali (MSE: {mse:.4f})")
    plt.xlabel("Feature Index")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.show()


def plot_anomaly_comparison(
    X_test,
    test_recon,
    anomalies,
    data_columns,
    test_mse,
    n_examples=N_EXAMPLES,
    n_features_to_show=N_FEATURES_TO_SHOW,
    figsize=(20, 12),
):
    # Set style untuk plot yang lebih menarik
    plt.style.use("seaborn")

    # Cari indeks anomali dan normal
    anomaly_indices = np.where(anomalies)[0]
    normal_indices = np.where(~anomalies)[0]

    # Validasi keberadaan anomali
    if len(anomaly_indices) == 0:
        print("Tidak ada anomali terdeteksi")
        return

    # Batasi jumlah contoh
    n_examples = min(n_examples, len(anomaly_indices))

    # Buat figure dengan subplot yang lebih rapi
    fig, axs = plt.subplots(n_examples, 2, figsize=figsize)
    fig.suptitle("Analisis Anomali: Perbandingan Fitur", fontsize=16, fontweight="bold")

    # Jika hanya satu contoh, pastikan axs adalah array 2D
    if n_examples == 1:
        axs = np.array([axs])

    for i in range(n_examples):
        # Ambil contoh anomali
        anomaly_idx = anomaly_indices[i]
        anomaly_sample = X_test[anomaly_idx, 0, :]
        anomaly_recon = test_recon[anomaly_idx, 0, :]

        # Ambil contoh normal dengan MSE terendah
        normal_idx = normal_indices[np.argmin(test_mse[normal_indices])]
        normal_sample = X_test[normal_idx, 0, :]
        normal_recon = test_recon[normal_idx, 0, :]

        # Hitung error per fitur
        anomaly_feature_error = np.power(anomaly_sample - anomaly_recon, 2)

        # Dapatkan fitur dengan error terbesar
        top_features_idx = np.argsort(anomaly_feature_error)[-n_features_to_show:][::-1]
        feature_names = [data_columns[idx] for idx in top_features_idx]

        # Subplot perbandingan fitur
        ax_bar = axs[i, 0]
        ax_line = axs[i, 1]

        # Bar plot perbandingan error fitur
        feature_errors = [anomaly_feature_error[idx] for idx in top_features_idx]
        sns.barplot(x=feature_names, y=feature_errors, ax=ax_bar, palette="coolwarm")
        ax_bar.set_title(f"Error Fitur Anomali #{i+1}")
        ax_bar.set_xlabel("Fitur")
        ax_bar.set_ylabel("Squared Error")
        ax_bar.tick_params(axis="x", rotation=45)

        # Line plot nilai aktual vs rekonstruksi
        ax_line.plot(
            feature_names,
            anomaly_sample[top_features_idx],
            "ro-",
            label="Aktual (Anomali)",
        )
        ax_line.plot(
            feature_names,
            anomaly_recon[top_features_idx],
            "rx--",
            label="Rekonstruksi (Anomali)",
        )
        ax_line.plot(
            feature_names,
            normal_sample[top_features_idx],
            "go-",
            label="Aktual (Normal)",
        )
        ax_line.plot(
            feature_names,
            normal_recon[top_features_idx],
            "gx--",
            label="Rekonstruksi (Normal)",
        )

        ax_line.set_title(f"Perbandingan Nilai Fitur Anomali #{i+1}")
        ax_line.set_xlabel("Fitur")
        ax_line.set_ylabel("Normalized Value")
        ax_line.tick_params(axis="x", rotation=45)
        ax_line.legend()

        # Cetak detail anomali
        print(
            f"\nDetail Anomali #{i+1} (Index: {anomaly_idx}, MSE: {test_mse[anomaly_idx]:.4f})"
        )
        print("Fitur dengan error terbesar:")
        for idx in top_features_idx:
            print(
                f"{data_columns[idx]:<25} | Error: {anomaly_feature_error[idx]:.4f} | "
                f"Aktual: {anomaly_sample[idx]:.4f} | Rekonstruksi: {anomaly_recon[idx]:.4f} | "
                f"Selisih: {abs(anomaly_sample[idx] - anomaly_recon[idx]):.4f}"
            )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_anomaly_detection(
    X_test: np.ndarray,
    test_recon: np.ndarray,
    anomalies: np.ndarray,
    feature_names: List[str],
) -> None:
    """
    Plot hasil deteksi anomali untuk beberapa fitur.

    Args:
        X_test: Data test
        test_recon: Rekonstruksi data test
        anomalies: Indikator anomali
        feature_names: Nama-nama fitur
    """
    # Pilih beberapa fitur untuk diplot
    selected_features = feature_names[: min(5, len(feature_names))]

    fig = make_subplots(
        rows=len(selected_features),
        cols=1,
        subplot_titles=selected_features,
    )

    for i, feature in enumerate(selected_features, 1):
        feature_idx = feature_names.index(feature)

        # Plot data asli
        fig.add_trace(
            go.Scatter(
                y=X_test[:, :, feature_idx].flatten(),
                name=f"{feature} - Original",
                mode="lines",
                line=dict(color="blue"),
            ),
            row=i,
            col=1,
        )

        # Plot rekonstruksi
        fig.add_trace(
            go.Scatter(
                y=test_recon[:, :, feature_idx].flatten(),
                name=f"{feature} - Reconstruction",
                mode="lines",
                line=dict(color="green"),
            ),
            row=i,
            col=1,
        )

        # Highlight anomali
        anomaly_indices = np.where(anomalies)[0]
        for idx in anomaly_indices:
            fig.add_vrect(
                x0=idx,
                x1=idx + 1,
                fillcolor="red",
                opacity=0.2,
                line_width=0,
                row=i,
                col=1,
            )

    fig.update_layout(
        title="Anomaly Detection Results",
        height=300 * len(selected_features),
        showlegend=True,
    )

    fig.show()
