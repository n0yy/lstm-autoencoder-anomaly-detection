import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from typing import Tuple, List

from config import DATA_PATH, DATE_COL, CATEGORICAL_COLS
from config import SEQUENCE_LENGTH
from config import TEST_SIZE, RANDOM_SEED

from utils.preprocessing import create_sequences


def validate_data(data: pd.DataFrame) -> None:
    """Validasi data untuk memastikan kualitas data."""
    # Cek missing values
    missing_values = data.isnull().sum()
    if missing_values.any():
        print("Peringatan: Ditemukan missing values:")
        print(missing_values[missing_values > 0])

    # Cek duplikat
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        print(f"Peringatan: Ditemukan {duplicates} baris duplikat")

    # Cek tipe data
    for col in data.columns:
        if data[col].dtype == "object" and col not in CATEGORICAL_COLS:
            print(f"Peringatan: Kolom {col} memiliki tipe data object")


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing data yang lebih robust."""
    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    data_imputed = pd.DataFrame(
        imputer.fit_transform(data), columns=data.columns, index=data.index
    )

    # Handle outliers menggunakan RobustScaler
    scaler = RobustScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(data_imputed),
        columns=data_imputed.columns,
        index=data_imputed.index,
    )

    return scaled_data


def load_data() -> pd.DataFrame:
    """Load dan validasi data."""
    data = pd.read_csv(
        DATA_PATH,
        index_col=DATE_COL,
        parse_dates=True,
    )

    validate_data(data)

    # One-hot encoding untuk kolom kategori
    for col in CATEGORICAL_COLS:
        data = pd.get_dummies(data, columns=[col], prefix=col)

    return data


def prepare_data() -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, List[str]]:
    """Prepare data untuk training dan testing."""
    data = load_data()
    data_columns = data.columns

    # Preprocess data
    processed_data = preprocess_data(data)

    # Final scaling untuk training
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(processed_data)

    # Create sequences
    X = create_sequences(scaled_data, SEQUENCE_LENGTH)

    # Split data
    X_train, X_test = train_test_split(
        X, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    return X_train, X_test, scaler, data_columns
