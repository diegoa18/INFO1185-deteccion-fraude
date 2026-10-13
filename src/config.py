"""Configuración central y rutas relativas del proyecto."""

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.30
N_SPLITS = 5
TARGET_COLUMN = "Class"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEFAULT_DATASET_PATH = RAW_DATA_DIR / "creditcard.csv"

PREDICTOR_COLUMNS = ("Time", "Amount", *(f"V{index}" for index in range(1, 29)))
REQUIRED_COLUMNS = (*PREDICTOR_COLUMNS, TARGET_COLUMN)
