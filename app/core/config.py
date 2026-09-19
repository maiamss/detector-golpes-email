import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = APP_DIR / "models"

MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "modelo_logreg_optuna.pkl"))
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", str(MODELS_DIR / "tfidf.pkl"))
SCALER_PATH = os.getenv("SCALER_PATH", str(MODELS_DIR / "scaler.pkl"))  # NOVO

MODEL_VERSION = os.getenv("MODEL_VERSION", "logreg_optuna_v1")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")