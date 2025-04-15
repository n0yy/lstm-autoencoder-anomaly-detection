# Sistem Deteksi Anomali dengan LSTM Autoencoder

Sistem deteksi anomali berbasis deep learning yang menggunakan LSTM Autoencoder untuk mendeteksi anomali dalam data time series.

## Fitur Utama

- Preprocessing data yang robust dengan penanganan missing values dan outliers
- Model LSTM Autoencoder dengan regularisasi dan batch normalization
- Evaluasi model yang komprehensif dengan berbagai metrik
- Visualisasi interaktif menggunakan Plotly
- Dashboard monitoring real-time
- Analisis feature importance
- Deteksi anomali multi-fitur

## Instalasi

1. Clone repository:

```bash
git clone https://github.com/yourusername/anomaly-detection.git
cd anomaly-detection
```

2. Buat virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Struktur Proyek

```
anomaly-detection/
├── config/                 # Konfigurasi model dan data
├── data/                   # Data dan data loader
├── models/                 # Implementasi model
├── utils/                  # Utility functions
├── notebooks/              # Jupyter notebooks
├── dashboard.py            # Dashboard monitoring
├── main.py                 # Script utama
├── predict.py              # Script prediksi
├── train.py                # Script training
└── requirements.txt        # Dependencies
```

## Penggunaan

### Training Model

```bash
python train.py
```

### Prediksi Anomali

```bash
python predict.py --input data/your_data.csv
```

### Menjalankan Dashboard

```bash
python run_dashboard.py
```

## Konfigurasi

Konfigurasi dapat diubah di `config/config.py`:

- `DATA_PATH`: Path ke data
- `SEQUENCE_LENGTH`: Panjang sequence untuk LSTM
- `BATCH_SIZE`: Ukuran batch training
- `EPOCHS`: Jumlah epoch training
- `ENCODING_DIM`: Dimensi encoding
- `LEARNING_RATE`: Learning rate optimizer
- Dan lainnya...

## Metrik Evaluasi

Sistem menggunakan berbagai metrik evaluasi:

- Precision
- Recall
- F1 Score
- ROC AUC
- Average Precision
- Reconstruction Error

## Visualisasi

Sistem menyediakan berbagai visualisasi:

- Training history
- Distribusi error rekonstruksi
- Feature importance
- Hasil deteksi anomali per fitur

## Kontribusi

1. Fork repository
2. Buat branch baru (`git checkout -b feature/amazing-feature`)
3. Commit perubahan (`git commit -m 'Add amazing feature'`)
4. Push ke branch (`git push origin feature/amazing-feature`)
5. Buat Pull Request
