import numpy as np
from utils.visualization import plot_anomaly_comparison
from utils.preprocessing import calculate_reconstruction_error


def detect_anomalies_in_new_data(model, X_new, threshold, data_columns=None):
    """
    Deteksi anomali pada data baru.

    Args:
        model: Model yang telah dilatih
        X_new (np.ndarray): Data baru untuk dievaluasi
        threshold (float): Threshold untuk deteksi anomali
        data_columns (list, optional): Nama kolom data

    Returns:
        tuple: reconstructed_data, mse, anomaly_indices
    """
    # Rekonstruksi data
    reconstructed_data = model.predict(X_new)

    # Hitung MSE
    mse = calculate_reconstruction_error(X_new, reconstructed_data)

    # Deteksi anomali
    anomalies = mse > threshold
    anomaly_indices = np.where(anomalies)[0]

    print(f"Jumlah data yang dievaluasi: {X_new.shape[0]}")
    print(f"Jumlah anomali terdeteksi: {len(anomaly_indices)}")

    # Visualisasi jika data_columns tersedia
    if data_columns is not None and len(anomaly_indices) > 0:
        plot_anomaly_comparison(X_new, reconstructed_data, anomalies, data_columns, mse)

    return reconstructed_data, mse, anomaly_indices
