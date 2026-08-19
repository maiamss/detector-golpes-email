"""
Configurações centrais do projeto.
Usa variáveis de ambiente para não deixar valores fixos no código
(útil na hora do deploy no Render/Railway/Fly.io).
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # raiz do projeto
APP_DIR = Path(__file__).resolve().parent.parent            # app/

MODELS_DIR = APP_DIR / "models"

# Caminhos dos artefatos do modelo. Sobrescrevíveis via variável de
# ambiente para facilitar deploy (ex: montar volume separado).
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "modelo_logistico.pkl"))
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", str(MODELS_DIR / "tfidf.pkl"))

MODEL_VERSION = os.getenv("MODEL_VERSION", "logreg_v1")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")