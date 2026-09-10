import re
import spacy
import numpy as np
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
stop_words = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    doc = nlp(text)
    palavras = [
        token.lemma_ for token in doc
        if token.text not in stop_words and not token.is_space
    ]
    return " ".join(palavras)


def preprocess_batch(texts: list[str], batch_size: int = 100, max_chars: int = 20000) -> list[str]:
    textos_limpos = []
    for text in texts:
        if not isinstance(text, str):
            textos_limpos.append("")
            continue
        t = text[:max_chars].lower()
        t = re.sub(r"http\S+|www\S+", "", t)
        t = re.sub(r"\d+", "", t)
        t = re.sub(r"[^\w\s]", "", t)
        textos_limpos.append(t)

    resultados = []
    for doc in nlp.pipe(textos_limpos, batch_size=batch_size):
        palavras = [
            token.lemma_ for token in doc
            if token.text not in stop_words and not token.is_space
        ]
        resultados.append(" ".join(palavras))
    return resultados


def extract_extra_features(text: str) -> dict:
    """
    Features estruturais que o TF-IDF não captura por olhar só vocabulário.
    Ajudam a diferenciar golpes informais/agressivos (muita maiúscula,
    exclamação, link cru) de comunicação formal legítima.
    """
    if not isinstance(text, str):
        text = ""

    total_chars = len(text) if len(text) > 0 else 1

    return {
        "has_url": int(bool(re.search(r"http\S+|www\S+", text))),
        "num_exclamacoes": text.count("!"),
        "pct_maiusculas": sum(1 for c in text if c.isupper()) / total_chars,
        "tamanho_texto": len(text),
    }


# Ordem fixa das features extras. Usada tanto no treino quanto na
# inferência para garantir que o vetor gerado tenha sempre as mesmas
# colunas, na mesma ordem — se isso divergir entre treino e produção,
# o modelo aplica os pesos errados às colunas erradas silenciosamente.
FEATURE_KEYS = ["has_url", "num_exclamacoes", "pct_maiusculas", "tamanho_texto"]


def extract_extra_features_batch(texts: list[str]) -> np.ndarray:
    """Versão vetorizada de extract_extra_features para uso em lote (treino)."""
    linhas = []
    for text in texts:
        feats = extract_extra_features(text)
        linhas.append([feats[k] for k in FEATURE_KEYS])
    return np.array(linhas, dtype=float)


def build_feature_matrix(texts_raw: list[str], texts_clean: list[str],
                          vectorizer, scaler, fit: bool = False):
    extras = extract_extra_features_batch(texts_raw)

    if fit:
        X_tfidf = vectorizer.fit_transform(texts_clean)
        extras_norm = scaler.fit_transform(extras)
    else:
        X_tfidf = vectorizer.transform(texts_clean)
        extras_norm = scaler.transform(extras)

    # O MinMaxScaler mapeia para [0, 1] a faixa que viu no treino, mas não
    # limita valores fora dela: um e-mail em CAIXA ALTA tem proporção de
    # maiúsculas de ~0,85 contra o máximo de 0,118 do treino, e sai do scaler
    # como 7,2 — sete vezes além de qualquer coisa que o modelo ajustou.
    # Multiplicado pelo peso negativo dessa coluna, isso zerava o score de
    # golpes escritos em maiúsculas. O clip prende os valores na faixa
    # treinada, para que uma feature fora de escala não domine a decisão.
    extras_norm = np.clip(extras_norm, 0.0, 1.0)

    X_combined = hstack([X_tfidf, csr_matrix(extras_norm)])
    return X_combined


if __name__ == "__main__":
    exemplo = "URGENT!!! Verify your account NOW at http://fake-bank.com or it will be suspended."
    print("Antes: ", exemplo)
    print("Depois: ", preprocess_text(exemplo))
    print("Features extras: ", extract_extra_features(exemplo))