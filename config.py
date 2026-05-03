from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

TRAIN_CSV = RAW_DATA_DIR / "fashion-mnist_train.csv"
TEST_CSV = RAW_DATA_DIR / "fashion-mnist_test.csv"

MODEL_PATH = MODELS_DIR / "fashion_cnn.pt"
SIMILARITY_INDEX_PATH = MODELS_DIR / "similarity_index.pkl"

IMAGE_SIZE = 28
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 8
LEARNING_RATE = 0.001