import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from app.services.preprocess import preprocess_batch, build_feature_matrix
from ml.metrics import avaliar_modelo
from ml.logging_utils import configurar_logger, registrar_experimento

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "app", "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "phishing_email.csv")

os.makedirs(MODELS_DIR, exist_ok=True)

logger = configurar_logger("treinamento")


def carregar_dados(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text_combined", "label"])
    df = df.drop_duplicates(subset=["text_combined"])
    return df


def preparar_dados(df: pd.DataFrame):
    logger.info(f"Pré-processando {len(df)} e-mails... (pode demorar um pouco)")
    df = df.copy()
    df["text_clean"] = preprocess_batch(df["text_combined"].tolist())
    return df


def treinar_regressao_logistica(X_train, y_train):
    modelo = LogisticRegression(max_iter=1000, class_weight="balanced")
    modelo.fit(X_train, y_train)
    return modelo


def treinar_random_forest(X_train, y_train):
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)
    return modelo


def treinar_mlp(X_train, y_train, **kwargs):
    params_default = dict(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
    )
    params_default.update(kwargs)

    modelo = MLPClassifier(**params_default)
    modelo.fit(X_train, y_train)
    return modelo


def main():
    df = carregar_dados()
    logger.info(f"Dataset carregado: {len(df)} linhas.")

    df = preparar_dados(df)

    clean_path = os.path.join(BASE_DIR, "data", "phishing_email_clean.csv")
    df.to_csv(clean_path, index=False)
    logger.info(f"Dataset limpo salvo em {clean_path}")

    y = df["label"]

    idx_train, idx_test, y_train, y_test = train_test_split(
        df.index, y, test_size=0.2, random_state=42, stratify=y
    )

    raw_train = df.loc[idx_train, "text_combined"].tolist()
    raw_test = df.loc[idx_test, "text_combined"].tolist()
    clean_train = df.loc[idx_train, "text_clean"].tolist()
    clean_test = df.loc[idx_test, "text_clean"].tolist()

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    scaler = MinMaxScaler()

    X_train = build_feature_matrix(raw_train, clean_train, vectorizer, scaler, fit=True)
    X_test = build_feature_matrix(raw_test, clean_test, vectorizer, scaler, fit=False)

    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    logger.info("Vectorizer e scaler salvos em app/models/")

    # ---------- Regressão Logística ----------
    logger.info("=== Treinando Regressão Logística ===")
    modelo_log = treinar_regressao_logistica(X_train, y_train)
    resultados_log = avaliar_modelo(modelo_log, X_test, y_test, nome="Regressão Logística")
    joblib.dump(modelo_log, os.path.join(MODELS_DIR, "modelo_logistico.pkl"))
    registrar_experimento({
        "modelo": "LogisticRegression",
        "hiperparametros": str(modelo_log.get_params()),
        **{k: v for k, v in resultados_log.items()},
    })

    # ---------- Random Forest ----------
    logger.info("=== Treinando Random Forest ===")
    modelo_rf = treinar_random_forest(X_train, y_train)
    resultados_rf = avaliar_modelo(modelo_rf, X_test, y_test, nome="Random Forest")
    joblib.dump(modelo_rf, os.path.join(MODELS_DIR, "modelo_random_forest.pkl"))
    registrar_experimento({
        "modelo": "RandomForest",
        "hiperparametros": str(modelo_rf.get_params()),
        **{k: v for k, v in resultados_rf.items()},
    })

    # ---------- MLP ----------
    logger.info("=== Treinando MLP (rede neural) ===")
    modelo_mlp = treinar_mlp(X_train, y_train)
    resultados_mlp = avaliar_modelo(modelo_mlp, X_test, y_test, nome="MLP")
    joblib.dump(modelo_mlp, os.path.join(MODELS_DIR, "modelo_mlp.pkl"))
    registrar_experimento({
        "modelo": "MLPClassifier",
        "hiperparametros": str(modelo_mlp.get_params()),
        **{k: v for k, v in resultados_mlp.items()},
    })

    logger.info("Treinamento concluído. Modelos salvos em app/models/.")
    logger.info(f"Histórico de experimentos atualizado em logs/historico_experimentos.csv")


if __name__ == "__main__":
    main()