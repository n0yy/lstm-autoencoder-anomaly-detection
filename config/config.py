DATA_PATH = "./data/streamed_synthetic_sig_5.csv"
DATE_COL = "timestamp"
CATEGORICAL_COLS = ["line", "enc_auger_direction"]
TEST_SIZE = 0.2

# Konfigurasi Sequence
SEQUENCE_LENGTH = 60

# Konfigurasi Training
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42

# Konfigurasi Model
ENCODING_DIM = 32
ENCODER_LAYERS = [64, 32]
DECODER_LAYERS = [32, 64]
LEARNING_RATE = 0.001
REGULARIZATION_RATE = 0.001
PATIENCE = 10
# Konfigurasi Visualisasi
N_EXAMPLES = 5
N_FEATURES_TO_SHOW = 10
