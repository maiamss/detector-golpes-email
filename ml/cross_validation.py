"""
Validação cruzada (k-fold) para Regressão Logística e Random Forest.
"""

import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "phishing_email_clean.csv")

N_SPLITS = 5
RANDOM_STATE = 42

# recall (sem sufixo) usa pos_label=1 por padrão em problemas binários,
# ou seja, mede especificamente o recall da classe phishing (1)
SCORING = {
    "accuracy": "accuracy",
    "precision_weighted": "precision_weighted",
    "recall_weighted": "recall_weighted",
    "f1_weighted": "f1_weighted",
    "recall_phishing": "recall",
}


def carregar_dados_limpos(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text_clean", "label"])
    return df


def construir_pipeline(modelo) -> Pipeline:
    # TF-IDF entra DENTRO do pipeline: assim, em cada fold, o vectorizer é
    # ajustado apenas com os dados de treino daquele fold, sem vazar
    # informação do fold de validação (mesmo princípio do train_test_split).
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", modelo),
    ])


def avaliar_cv(nome: str, modelo, X, y, cv):
    pipeline = construir_pipeline(modelo)
    resultados = cross_validate(
        pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1, return_train_score=False
    )

    print(f"\n=== {nome} — Validação Cruzada ({N_SPLITS}-fold) ===")
    for metrica in SCORING:
        valores = resultados[f"test_{metrica}"]
        print(f"{metrica:>18s}: {valores.mean():.4f} ± {valores.std():.4f}  "
              f"(folds: {[round(float(v), 4) for v in valores]})")

    return resultados


def main():
    df = carregar_dados_limpos()
    X = df["text_clean"]
    y = df["label"]

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    avaliar_cv(
        "Regressão Logística",
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        X, y, cv,
    )

    avaliar_cv(
        "Random Forest",
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,  # evita paralelismo aninhado com o n_jobs=-1 do cross_validate
        ),
        X, y, cv,
    )

    print(
        "\nNo texto do TCC: reporte média ± desvio padrão (não só a média). "
        "Um desvio pequeno reforça que o desempenho é estável entre diferentes "
        "divisões dos dados, não fruto de sorte numa divisão específica."
    )


if __name__ == "__main__":
    main()