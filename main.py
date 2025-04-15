import argparse

from train import train_model
from predict import detect_anomalies_in_new_data
from utils.visualization import plot_anomaly_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Anomaly Detection using LSTM Autoencoder"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "both"],
        help="Mode to run the application in",
    )

    args = parser.parse_args()

    if args.mode == "train" or args.mode == "both":
        # Train model
        print("Training model...")
        model, history, results = train_model()

        if args.mode == "both":
            # Evaluate dan visualisasi
            print("\nEvaluasi hasil dan visualisasi anomali...")
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
            ) = results

            plot_anomaly_comparison(
                X_test, test_recon, anomalies, data_columns, test_mse
            )

    elif args.mode == "evaluate":
        # Mode ini akan membutuhkan model yang telah disimpan sebelumnya
        print("Mode evaluasi membutuhkan model yang telah disimpan sebelumnya.")
        print(
            "Silakan jalankan mode 'train' terlebih dahulu atau 'both' untuk evaluasi lengkap."
        )


if __name__ == "__main__":
    main()
