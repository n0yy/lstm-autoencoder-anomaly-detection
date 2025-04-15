from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from config import (
    ENCODING_DIM,
    ENCODER_LAYERS,
    DECODER_LAYERS,
    LEARNING_RATE,
    REGULARIZATION_RATE,
)


def create_lstm_autoencoder(
    timesteps: int,
    n_features: int,
    encoding_dim: int = ENCODING_DIM,
) -> Model:
    """
    Membuat model LSTM Autoencoder dengan fitur-fitur tambahan.

    Args:
        timesteps: Jumlah timesteps dalam sequence
        n_features: Jumlah fitur
        encoding_dim: Dimensi encoding

    Returns:
        Model LSTM Autoencoder
    """
    # Input layer
    inputs = Input(shape=(timesteps, n_features))

    # Encoder
    x = inputs
    for units in ENCODER_LAYERS:
        x = LSTM(
            units,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            kernel_regularizer=l2(REGULARIZATION_RATE),
            recurrent_regularizer=l2(REGULARIZATION_RATE),
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

    # Encoding
    encoded = LSTM(
        encoding_dim,
        activation="tanh",
        recurrent_activation="sigmoid",
        kernel_regularizer=l2(REGULARIZATION_RATE),
        recurrent_regularizer=l2(REGULARIZATION_RATE),
    )(x)

    # Decoder
    x = RepeatVector(timesteps)(encoded)
    for units in DECODER_LAYERS:
        x = LSTM(
            units,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            kernel_regularizer=l2(REGULARIZATION_RATE),
            recurrent_regularizer=l2(REGULARIZATION_RATE),
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

    # Output layer
    decoded = TimeDistributed(
        Dense(
            n_features,
            activation="linear",
            kernel_regularizer=l2(REGULARIZATION_RATE),
        )
    )(x)

    # Create model
    model = Model(inputs, decoded)

    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"],
    )

    return model
