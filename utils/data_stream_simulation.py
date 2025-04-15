import pandas as pd
import numpy as np
import time
from datetime import datetime
import logging
import os
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Konfigurasi
CONFIG = {
    "output_file": "./data/streamed_synthetic_sig_5.csv",
    "frequency": 0.1,  # Frekuensi data baru (detik) - lebih cepat
    "duration": 100,  # Durasi simulasi (detik) - lebih singkat
    "anomaly_probability": 0.05,  # Probabilitas anomali per data point
    "columns": [
        "line",
        "enc_auger_position_angle",
        "enc_auger_rpm",
        "enc_auger_acceleration",
        "enc_auger_pulse_count",
        "enc_auger_direction",
        "enc_auger_pulse_frequency",
        "enc_stirer",
        "enc_filler_slide",
        "enc_sealing_horizontal",
        "enc_sealing_vertical",
        "enc_pisau_horizontal",
        "heater_cross_1",
        "heater_cross_2",
        "heater_cross_3",
        "heater_cross_4",
        "heater_longi_1",
        "heater_longi_2",
        "dosing",
        "conveyor_pulse",
    ],
    "line_values": [1, 2, 3, 4, 5, 6, 7, 8],
}

FEATURE_RANGES = {
    "enc_auger_position_angle": (0, 360),
    "enc_auger_rpm": (0, 100),
    "enc_auger_acceleration": (0, 20),
    "enc_auger_pulse_count": (0, 10000),
    "enc_auger_direction": [-1, 0, 1],
    "enc_auger_pulse_frequency": (0, 50),
    "enc_stirer": (0, 50),
    "enc_filler_slide": (0, 50),
    "enc_sealing_horizontal": (0, 50),
    "enc_sealing_vertical": (0, 50),
    "enc_pisau_horizontal": (0, 100),
    "heater_cross_1": (50, 150),
    "heater_cross_2": (50, 150),
    "heater_cross_3": (50, 150),
    "heater_cross_4": (50, 150),
    "heater_longi_1": (50, 150),
    "heater_longi_2": (50, 150),
    "dosing": (0, 10),
    "conveyor_pulse": (0, 1000),
}


def generate_data_point(line: int, anomaly: bool = False) -> Dict:
    data = {"line": line}

    for feature, range_vals in FEATURE_RANGES.items():
        if feature == "enc_auger_direction":
            value = np.random.choice(range_vals)
        else:
            min_val, max_val = range_vals
            value = np.random.uniform(min_val, max_val)

            if anomaly and feature != "enc_auger_direction":
                anomaly_factor = np.random.choice([0.5, 1.5])
                value *= anomaly_factor
                value = max(min_val, min(value, max_val * 1.5))

        data[feature] = value

    return data


def stream_data(config: Dict):
    """Mensimulasikan data streaming dan menyimpannya ke CSV."""
    start_time = time.time()
    end_time = start_time + config["duration"]

    # Pastikan direktori output ada
    os.makedirs(os.path.dirname(config["output_file"]), exist_ok=True)

    output_columns = ["timestamp"] + config["columns"]
    if not os.path.exists(config["output_file"]):
        pd.DataFrame(columns=output_columns).to_csv(config["output_file"], index=False)

    logging.info(
        f"Memulai simulasi streaming data selama {config['duration']} detik..."
    )

    while time.time() < end_time:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = np.random.choice(config["line_values"])
        is_anomaly = np.random.random() < config["anomaly_probability"]

        data_point = generate_data_point(line, anomaly=is_anomaly)
        data_point["timestamp"] = timestamp

        ordered_data = {col: data_point.get(col, np.nan) for col in output_columns}
        df_point = pd.DataFrame([ordered_data], columns=output_columns)

        df_point.to_csv(config["output_file"], mode="a", header=False, index=False)

        logging.info(f"Data point ditambahkan: {timestamp}, Anomali: {is_anomaly}")

        time.sleep(config["frequency"])

    logging.info(f"Simulasi selesai. Data disimpan ke {config['output_file']}")


def main():
    """Menjalankan simulasi data stream"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulasi Data Stream untuk Anomaly Detection"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=CONFIG["duration"],
        help=f"Durasi simulasi dalam detik (default: {CONFIG['duration']})",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=CONFIG["frequency"],
        help=f"Frekuensi data baru dalam detik (default: {CONFIG['frequency']})",
    )
    parser.add_argument(
        "--anomaly-prob",
        type=float,
        default=CONFIG["anomaly_probability"],
        help=f"Probabilitas anomali per data point (default: {CONFIG['anomaly_probability']})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=CONFIG["output_file"],
        help=f"Path file output (default: {CONFIG['output_file']})",
    )

    args = parser.parse_args()

    # Update konfigurasi berdasarkan argumen
    config = CONFIG.copy()
    config["duration"] = args.duration
    config["frequency"] = args.frequency
    config["anomaly_probability"] = args.anomaly_prob
    config["output_file"] = args.output

    try:
        stream_data(config)
    except KeyboardInterrupt:
        logging.info("Simulasi dihentikan oleh pengguna")
    except Exception as e:
        logging.error(f"Terjadi kesalahan: {e}")
        raise


if __name__ == "__main__":
    main()
