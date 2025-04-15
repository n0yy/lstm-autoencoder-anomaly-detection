import os
import argparse
import threading
import subprocess
import time
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_simulation(duration, frequency, anomaly_prob):
    """Menjalankan data_stream_simulation sebagai proses terpisah"""
    try:
        cmd = [
            "python",
            "-m",
            "utils.data_stream_simulation",
            "--duration",
            str(duration),
            "--frequency",
            str(frequency),
            "--anomaly-prob",
            str(anomaly_prob),
        ]
        proc = subprocess.Popen(cmd)
        return proc
    except Exception as e:
        logging.error(f"Gagal menjalankan simulasi: {e}")
        return None


def run_dashboard():
    """Menjalankan dashboard"""
    try:
        import dashboard

        dashboard.app.run(debug=True, host="0.0.0.0", port=8050)
    except Exception as e:
        logging.error(f"Gagal menjalankan dashboard: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run Anomaly Detection Dashboard")
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Jalankan simulasi data stream bersamaan dengan dashboard",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=3600,
        help="Durasi simulasi dalam detik (default: 3600)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=1.0,
        help="Frekuensi data baru dalam detik (default: 1.0)",
    )
    parser.add_argument(
        "--anomaly-prob",
        type=float,
        default=0.05,
        help="Probabilitas anomali (default: 0.05)",
    )

    args = parser.parse_args()

    # Pastikan data dir ada
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Jalankan simulasi jika diminta
    sim_process = None
    if args.simulation:
        logging.info(
            f"Memulai simulasi data stream (durasi={args.duration}s, freq={args.frequency}s)"
        )
        sim_process = run_simulation(args.duration, args.frequency, args.anomaly_prob)

    # Jalankan dashboard
    logging.info("Memulai dashboard di http://localhost:8050")
    try:
        run_dashboard()
    except KeyboardInterrupt:
        logging.info("Dashboard dihentikan oleh pengguna")
    finally:
        # Pastikan proses simulasi dihentikan jika masih berjalan
        if sim_process is not None and sim_process.poll() is None:
            logging.info("Menghentikan proses simulasi...")
            sim_process.terminate()
            sim_process.wait()


if __name__ == "__main__":
    main()
