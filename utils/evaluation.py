import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from typing import Tuple, Dict, List

from utils.preprocessing import calculate_reconstruction_error, detect_anomalies
from config import N_FEATURES_TO_SHOW


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Menghitung berbagai metrik evaluasi untuk model deteksi anomali.

    Args:
        y_true: Label sebenarnya (0: normal, 1: anomali)
        y_pred: Prediksi model
        threshold: Threshold untuk menentukan anomali

    Returns:
        Dictionary berisi metrik-metrik evaluasi
    """
    y_pred_binary = (y_pred > threshold).astype(int)

    metrics = {
        "precision": precision_score(y_true, y_pred_binary),
        "recall": recall_score(y_true, y_pred_binary),
        "f1_score": f1_score(y_true, y_pred_binary),
        "roc_auc": roc_auc_score(y_true, y_pred),
        "average_precision": average_precision_score(y_true, y_pred),
    }

    return metrics


def analyze_feature_importance(
    X_test: np.ndarray,
    test_recon: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Menganalisis kontribusi setiap fitur terhadap error rekonstruksi.

    Args:
        X_test: Data test
        test_recon: Rekonstruksi data test
        feature_names: Nama-nama fitur

    Returns:
        Dictionary berisi kontribusi error per fitur
    """
    # Hitung error rekonstruksi per fitur
    feature_errors = np.mean(np.abs(X_test - test_recon), axis=(0, 1))

    # Normalisasi error
    feature_importance = feature_errors / np.sum(feature_errors)

    # Buat dictionary dengan nama fitur
    importance_dict = dict(zip(feature_names, feature_importance))

    # Urutkan berdasarkan importance
    sorted_importance = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )

    return sorted_importance


def evaluate_model(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    Dict[str, float],
]:
    """
    Evaluasi model dengan berbagai metrik.

    Args:
        model: Model yang akan dievaluasi
        X_train: Data training
        X_test: Data test

    Returns:
        Tuple berisi hasil evaluasi
    """
    # Prediksi
    train_recon = model.predict(X_train)
    test_recon = model.predict(X_test)

    # Hitung MSE
    train_mse = np.mean(np.square(X_train - train_recon), axis=(1, 2))
    test_mse = np.mean(np.square(X_test - test_recon), axis=(1, 2))

    # Tentukan threshold
    threshold = np.percentile(train_mse, 95)

    # Deteksi anomali
    anomalies = test_mse > threshold

    # Hitung metrik dengan penanganan kasus khusus
    try:
        metrics = calculate_metrics(
            np.zeros_like(test_mse),  # Asumsikan semua data normal
            test_mse,
            threshold,
        )
    except Exception as e:
        print(f"Peringatan: {str(e)}")
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.0,
            "average_precision": 0.0,
        }

    return (
        train_recon,
        test_recon,
        train_mse,
        test_mse,
        threshold,
        anomalies,
        metrics,
    )


def analyze_anomaly(X_test, test_recon, anomaly_idx, data_columns):
    sample = X_test[anomaly_idx, 0, :]
    sample_recon = test_recon[anomaly_idx, 0, :]

    # Hitung error per fitur
    feature_error = np.power(sample - sample_recon, 2)

    # Dapatkan fitur dengan error terbesar
    top_features_idx = np.argsort(feature_error)[-5:][::-1]

    return sample, sample_recon, feature_error, top_features_idx
