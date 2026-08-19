import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from app.services.preprocess import preprocess_batch
from ml.metrics import avaliar_modelo

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz do projeto
MODELS_DIR = os.path.join(BASE_DIR, "app", "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "phishing_email.csv")

os.makedirs(MODELS_DIR, exist_ok=True)

def carregar_dados(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text_combined", "label"]) 
    df = df.drop_duplicates(subset=["text_combined"])
    return df 

def preparar_dados(df: pd.DataFrame): #recebe um DataFrame
    print(f"Pré-processando {len(df)} e-mails... (pode demorar um pouco)")
    df = df.copy()
    df["text_clean"] = preprocess_batch(df["text_combined"].tolist()) 
    return df

def treinar_regressao_logistica(X_train, y_train):
    modelo = LogisticRegression(
        max_iter=1000,
        class_weight="balanced")
    modelo.fit(X_train, y_train)
    return modelo

def treinar_random_forest(X_train, y_train):
    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)
    return modelo

def main():

    df = carregar_dados()
    print(f"Dataset carregado: {len(df)} linhas.")

    df = preparar_dados(df)

    clean_path = os.path.join(BASE_DIR, "data", "phishing_email_clean.csv")
    df.to_csv(clean_path, index=False)
    print(f"Dataset limpo salvo em {clean_path}")

    X = df["text_clean"]
    y = df["label"]
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf.pkl"))
    print("Vectorizer salvo em models/tfidf.pkl")

    print("\n=== Treinando Regressão Logística ===")
    modelo_log = treinar_regressao_logistica(X_train, y_train)
    avaliar_modelo(modelo_log, X_test, y_test, nome="Regressão Logística")
    joblib.dump(modelo_log, os.path.join(MODELS_DIR, "modelo_logistico.pkl"))

    print("\n=== Treinando Random Forest ===")
    modelo_rf = treinar_random_forest(X_train, y_train)
    avaliar_modelo(modelo_rf, X_test, y_test, nome="Random Forest")
    joblib.dump(modelo_rf, os.path.join(MODELS_DIR, "modelo_random_forest.pkl"))

    print("\nTreinamento concluído. Modelos salvos em /models.")

if __name__ == "__main__":
    main()