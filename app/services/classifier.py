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


def classificar_email(email_text: str) -> EmailAnaliseResponse:
    """
    Recebe o texto do e-mail e retorna a análise de risco usando o
    modelo real (TF-IDF + Regressão Logística, por padrão).

    Contrato mantido: input string -> output EmailAnaliseResponse,
    igual ao mock anterior, para não quebrar app/main.py nem os testes.
    """
    inicio = time.perf_counter()

    if not _modelo_esta_carregado():
        # Sem modelo carregado, não tem como classificar de verdade.
        # Quem decide o que fazer com isso é o endpoint (main.py),
        # aqui só sinalizamos via exceção.
        raise RuntimeError(
            "Modelo não carregado. Verifique MODEL_PATH e VECTORIZER_PATH."
        )

    texto_lower = email_text.lower()

    # Mesmo pipeline de limpeza usado no treino (ml/train.py ->
    # preprocess_batch), para não haver distância entre treino e inferência.
    texto_limpo = preprocess_text(email_text)
    X = _vectorizer.transform([texto_limpo])

    # predict_proba retorna [P(classe=0), P(classe=1)]; assumimos que
    # label 1 == golpe/phishing, conforme o dataset usado no treino.
    proba = _modelo.predict_proba(X)[0]
    score = float(proba[1])

    padroes = _extrair_padroes(texto_lower)

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