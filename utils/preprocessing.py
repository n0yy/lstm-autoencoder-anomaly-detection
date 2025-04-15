import numpy as np


def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i : (i + seq_length)]
        X.append(seq)
    return np.array(X)


def calculate_reconstruction_error(original, reconstructed, axis=(1, 2)):

    return np.mean(np.power(original - reconstructed, 2), axis=axis)


def calculate_feature_errors(original, reconstructed):
    return np.mean(np.power(original - reconstructed, 2), axis=1)


def normalize_feature_errors(feature_errors):
    return feature_errors / np.sum(feature_errors, axis=1, keepdims=True)


def detect_anomalies(train_mse, test_mse, threshold_percentile=95):
    threshold = np.percentile(train_mse, threshold_percentile)
    anomalies = test_mse > threshold

    return threshold, anomalies
