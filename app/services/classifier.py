"""
Serviço responsável por classificar o e-mail.

Usa o modelo real treinado em ml/train.py (TF-IDF + Regressão Logística
por padrão, configurável via MODEL_PATH). A extração de padrões
(`explicacao`) continua baseada em palavras-chave, pois serve como
camada de interpretabilidade complementar à predição do modelo —
o modelo dá o score, as regras explicam o "porquê" para o usuário final.
"""

import time
import logging

import joblib
import re

from app.models.schemas import EmailAnaliseResponse, PadraoDetectado, NivelRisco
from app.core.config import MODEL_PATH, VECTORIZER_PATH, MODEL_VERSION
from app.services.preprocess import preprocess_text

logger = logging.getLogger(__name__)

# Palavras-chave usadas apenas para gerar a explicação exibida ao usuário.
# Não influenciam o score, que agora vem do modelo real.
PALAVRAS_URGENCIA = ["urgente", "24 horas", "imediatamente", "bloqueada", "bloqueado"]
PALAVRAS_DADOS = ["atualizar seus dados", "confirme seus dados", "regularize"]
PALAVRAS_LINK = ["clique aqui", "http://", "https://"]


def _carregar_artefatos():
    """
    Carrega modelo e vectorizer uma única vez, na importação do módulo.
    Se algo falhar (arquivo ausente, .pkl corrompido), o serviço continua
    de pé — /health vai reportar modelo_carregado=False e /analisar-email
    retorna erro 503 em vez de derrubar a API inteira.
    """
    try:
        modelo = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("Modelo e vectorizer carregados com sucesso (%s).", MODEL_VERSION)
        return modelo, vectorizer
    except FileNotFoundError as e:
        logger.error("Arquivo de modelo/vectorizer não encontrado: %s", e)
    except Exception as e:
        logger.exception("Falha ao carregar modelo/vectorizer: %s", e)
    return None, None


_modelo, _vectorizer = _carregar_artefatos()


def _modelo_esta_carregado() -> bool:
    """Usado pelo endpoint /health para reportar o estado real do modelo."""
    return _modelo is not None and _vectorizer is not None


def _extrair_padroes(texto_lower: str) -> list[PadraoDetectado]:
    """Camada de explicabilidade baseada em regras (não afeta o score)."""
    padroes: list[PadraoDetectado] = []

    for palavra in PALAVRAS_URGENCIA:
        if palavra in texto_lower:
            padroes.append(
                PadraoDetectado(
                    tipo="urgencia_artificial",
                    descricao="Uso de expressão de urgência",
                    trecho=palavra,
                )
            )
            break

    for palavra in PALAVRAS_DADOS:
        if palavra in texto_lower:
            padroes.append(
                PadraoDetectado(
                    tipo="solicitacao_dados",
                    descricao="Solicitação de atualização de dados sensíveis",
                    trecho=palavra,
                )
            )
            break

    for palavra in PALAVRAS_LINK:
        if palavra in texto_lower:
            padroes.append(
                PadraoDetectado(
                    tipo="link_suspeito",
                    descricao="Presença de link na mensagem",
                    trecho=palavra,
                )
            )
            break

    return padroes

def _analisar_remetente(sender: str | None) -> float:
    """
    Heurística leve sobre o remetente. Retorna um ajuste (0 a ~0.15)
    a ser somado ao score do modelo. Não usa ML porque o dataset de
    treino não tem remetente confiável — isso é regra, não predição.
    """
    if not sender:
        return 0.0

    sender_lower = sender.lower()
    ajuste = 0.0

    # domínio gratuito tentando se passar por instituição
    dominios_gratuitos = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "aol.com", "mail.com", "protonmail.com"]
    palavras_bancarias = [
        "bank", "chase", "wellsfargo", "bankofamerica", "citibank",
        "paypal", "amex", "americanexpress", "capitalone",
        "hsbc", "barclays", "lloyds",
    ]

    if any(p in sender_lower for p in palavras_bancarias) and \
       any(d in sender_lower for d in dominios_gratuitos):
        ajuste += 0.15  # nome de banco em domínio gratuito = bandeira vermelha

    # muitos números/caracteres estranhos no domínio (típico de spoofing)
    dominio = sender_lower.split("@")[-1] if "@" in sender_lower else ""
    if re.search(r"\d{3,}", dominio):
        ajuste += 0.05

    return ajuste


def classificar_email(email_text: str, email_subject: str | None = None,
                       sender: str | None = None) -> EmailAnaliseResponse:
    inicio = time.perf_counter()

    if not _modelo_esta_carregado():
        raise RuntimeError("Modelo não carregado. Verifique MODEL_PATH e VECTORIZER_PATH.")

    # monta o mesmo formato usado no treino: assunto + corpo
    texto_completo = f"{email_subject or ''} {email_text}".strip()
    texto_lower = texto_completo.lower()

    texto_limpo = preprocess_text(texto_completo)
    X = _vectorizer.transform([texto_limpo])

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