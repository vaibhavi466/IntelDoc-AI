import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

# Paths
DATA_RAW_DIR     = ROOT / "data" / "raw"
DATA_CSV         = ROOT / "data" / "processed" / "documind_dataset.csv"
MODEL_DIR        = ROOT / "models" / "documind_v1"
DB_PATH          = ROOT / "documind.db"
EVAL_METRICS     = ROOT / "models" / "eval_metrics.json"
CONFUSION_MATRIX = ROOT / "models" / "confusion_matrix.png"
TEMP_DIR         = ROOT / "data" / "temp" 

# Model config
BASE_MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LENGTH  = 512
BATCH_SIZE      = 8
LEARNING_RATE   = 2e-5
NUM_EPOCHS      = 3

# Categories (must match training data folder names)
DOCUMENT_CLASSES = ["email", "form", "invoice", "letter", "news_article", "resume"]