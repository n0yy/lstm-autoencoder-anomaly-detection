import numpy as np
import os
import joblib
from tensorflow.keras.models import save_model

from data.data_loader import prepare_data
from models.autoencoder import create_lstm_autoencoder
from models.model_utils import get_callbacks
from utils.evaluation import evaluate_model, analyze_feature_importance
from utils.visualization import (
    plot_training_history,
    plot_reconstruction_error_distribution,
    plot_feature_importance,
)

from config import BATCH_SIZE, EPOCHS
from config import ENCODING_DIM, VALIDATION_SPLIT

# Buat direktori untuk menyimpan model dan hasil
os.makedirs("models/saved", exist_ok=True)
os.makedirs("results", exist_ok=True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Mematikan pesan warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Mematikan custom operations


def train_model():
    X_train, X_test, scaler, data_columns = prepare_data()

    timesteps = X_train.shape[1]
    n_features = X_train.shape[2]

    model = create_lstm_autoencoder(timesteps, n_features, encoding_dim=ENCODING_DIM)
    print(model.summary())

    # Callbacks
    callbacks = get_callbacks()

    # Training
    history = model.fit(
        X_train,
        X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    # Simpan model dan hasil
    save_model(model, "models/saved/model.h5")
    joblib.dump(scaler, "models/saved/scaler.joblib")
    joblib.dump(data_columns, "models/saved/data_columns.joblib")

    # Simpan hasil training
    training_results = {
        "history": history.history,
        "metrics": metrics,
        "threshold": threshold,
        "feature_importance": feature_importance,
    }
    joblib.dump(training_results, "results/training_results.joblib")

    # Evaluasi model
    results = evaluate_model(model, X_train, X_test)
    train_recon, test_recon, train_mse, test_mse, threshold, anomalies, metrics = (
        results
    )

    feature_importance = analyze_feature_importance(X_test, test_recon, data_columns)

    # Visualisasi
    plot_training_history(history)
    plot_reconstruction_error_distribution(train_mse, test_mse, threshold)
    plot_feature_importance(feature_importance)

    print(f"Jumlah anomali terdeteksi: {np.sum(anomalies)}")
    print("\nMetrik Evaluasi:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    return (
        model,
        history,
        (
            X_train,
            X_test,
            train_recon,
            test_recon,
            train_mse,
            test_mse,
            threshold,
            anomalies,
            data_columns,
        ),
    )


if __name__ == "__main__":
    train_model()
