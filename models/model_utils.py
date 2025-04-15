from tensorflow.keras.callbacks import EarlyStopping

from config import PATIENCE


def get_callbacks():
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    ]

    return callbacks
