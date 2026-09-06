"""
Serviço responsável por classificar o e-mail.
"""

import time
import logging

import joblib
import re

from app.models.schemas import EmailAnaliseResponse, PadraoDetectado, NivelRisco
from app.core.config import MODEL_PATH, VECTORIZER_PATH, SCALER_PATH, MODEL_VERSION
from app.services.preprocess import preprocess_text, build_feature_matrix

logger = logging.getLogger(__name__)

PALAVRAS_URGENCIA = ["urgent", "24 hours", "immediately", "suspended", "expire"]
PALAVRAS_DADOS = ["verify your account", "confirm your details", "update your information"]
PALAVRAS_LINK = ["click here", "http://", "https://"]


def _carregar_artefatos():
    try:
        modelo = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info("Modelo, vectorizer e scaler carregados com sucesso (%s).", MODEL_VERSION)
        return modelo, vectorizer, scaler
    except FileNotFoundError as e:
        logger.error("Arquivo de modelo/vectorizer/scaler não encontrado: %s", e)
    except Exception as e:
        logger.exception("Falha ao carregar artefatos: %s", e)
    return None, None, None


_modelo, _vectorizer, _scaler = _carregar_artefatos()


def _modelo_esta_carregado() -> bool:
    return _modelo is not None and _vectorizer is not None and _scaler is not None


def _extrair_padroes(texto_lower: str) -> list[PadraoDetectado]:
    padroes: list[PadraoDetectado] = []

    for palavra in PALAVRAS_URGENCIA:
        if palavra in texto_lower:
            padroes.append(PadraoDetectado(
                tipo="urgencia_artificial",
                descricao="Uso de expressão de urgência",
                trecho=palavra,
            ))
            break

    for palavra in PALAVRAS_DADOS:
        if palavra in texto_lower:
            padroes.append(PadraoDetectado(
                tipo="solicitacao_dados",
                descricao="Solicitação de atualização de dados sensíveis",
                trecho=palavra,
            ))
            break

    for palavra in PALAVRAS_LINK:
        if palavra in texto_lower:
            padroes.append(PadraoDetectado(
                tipo="link_suspeito",
                descricao="Presença de link na mensagem",
                trecho=palavra,
            ))
            break

    return padroes


def _analisar_remetente(sender: str | None) -> float:
    if not sender:
        return 0.0

    sender_lower = sender.lower()
    ajuste = 0.0

    dominios_gratuitos = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "aol.com", "mail.com", "protonmail.com"]
    palavras_bancarias = [
        "bank", "chase", "wellsfargo", "bankofamerica", "citibank",
        "paypal", "amex", "americanexpress", "capitalone",
        "hsbc", "barclays", "lloyds",
    ]

    if any(p in sender_lower for p in palavras_bancarias) and \
       any(d in sender_lower for d in dominios_gratuitos):
        ajuste += 0.15

    dominio = sender_lower.split("@")[-1] if "@" in sender_lower else ""
    if re.search(r"\d{3,}", dominio):
        ajuste += 0.05

    return ajuste


def classificar_email(email_text: str, email_subject: str | None = None,
                       sender: str | None = None) -> EmailAnaliseResponse:
    inicio = time.perf_counter()

    if not _modelo_esta_carregado():
        raise RuntimeError("Modelo não carregado. Verifique MODEL_PATH, VECTORIZER_PATH e SCALER_PATH.")

    texto_completo = f"{email_subject or ''} {email_text}".strip()
    texto_lower = texto_completo.lower()
    texto_limpo = preprocess_text(texto_completo)

    # mesma combinação TF-IDF + features extras usada no treino
    X = build_feature_matrix(
        texts_raw=[texto_completo],
        texts_clean=[texto_limpo],
        vectorizer=_vectorizer,
        scaler=_scaler,
        fit=False,
    )

    proba = _modelo.predict_proba(X)[0]
    score_modelo = float(proba[1])

    ajuste_remetente = _analisar_remetente(sender)
    score = min(score_modelo + ajuste_remetente, 1.0)

    padroes = _extrair_padroes(texto_lower)
    if ajuste_remetente > 0:
        padroes.append(PadraoDetectado(
            tipo="remetente_suspeito",
            descricao="Remetente com características suspeitas (domínio/nome)",
            trecho=sender,
        ))

    if score >= 0.6:
        risco = NivelRisco.alto
        classificacao = "golpe"
    elif score >= 0.3:
        risco = NivelRisco.medio
        classificacao = "suspeito"
    else:
        risco = NivelRisco.baixo
        classificacao = "legitimo"

    tempo_ms = (time.perf_counter() - inicio) * 1000

    return EmailAnaliseResponse(
        risco=risco,
        score=round(score, 4),
        classificacao=classificacao,
        explicacao=padroes,
        modelo_usado=MODEL_VERSION,
        tempo_inferencia_ms=round(tempo_ms, 2),
    )