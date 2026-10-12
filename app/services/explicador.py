"""
Explicador baseado no modelo (TF-IDF + Regressão Logística).

Complementa as regras fixas de classifier.py: quando nenhuma regra dispara
(ou para além delas), explica a decisão usando o que o PRÓPRIO modelo viu.

Ideia central
-------------
Em uma regressão logística, o logit do e-mail é uma soma:

    logit = intercepto + sum_j (coef_j * x_j)

Cada termo `j` presente no e-mail contribui com `coef_j * x_j`. Os termos de
maior contribuição positiva são, literalmente, os motivos pelos quais o modelo
chamou o e-mail de golpe. Isso é uma explicação fiel (não é adivinhação).

Os termos brutos do vocabulário nem sempre fazem sentido para o usuário final
(ex.: "aug", "josemonkeyorg" são artefatos do dataset). Por isso os termos
passam por um léxico semântico: só viram explicação os que pertencem a uma
categoria compreensível; o restante é ignorado na resposta (mas conta na
métrica de cobertura).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

# categoria -> (descrição mostrada ao usuário, termos do vocabulário do TF-IDF)
# Os termos são lemas em inglês, do jeito que saem de preprocess_text().
LEXICO_SEMANTICO: dict[str, tuple[str, set[str]]] = {
    "urgencia_artificial": (
        "Linguagem de urgência ou prazo para pressionar uma ação rápida",
        {"urgent", "immediately", "within hour", "within", "hour", "expire",
         "attention", "soon", "act", "important", "limited", "suspend"},
    ),
    "solicitacao_dados": (
        "Pedido de verificação, confirmação ou envio de dados pessoais/da conta",
        {"verify", "verification", "validate", "confirm", "please confirm",
         "detail", "identity", "information", "account", "secure", "card",
         "credit"},
    ),
    "assunto_financeiro": (
        "Assunto financeiro típico de golpe (transferência, fundos, pagamento)",
        {"bank", "banking", "transfer", "fund", "payment", "pay", "money",
         "loan", "mortgage", "investment", "dollar", "beneficiary", "paypal",
         "usaa"},
    ),
    "promessa_ganho": (
        "Promessa de prêmio, ganho fácil ou oferta vantajosa demais",
        {"win", "lottery", "prize", "claim", "reward", "million", "free",
         "offer", "cheap", "earn", "gain", "guarantee", "eligible", "huge",
         "low rate"},
    ),
    "chamada_acao": (
        "Chamada para clicar em link ou executar uma ação imediata",
        {"click", "http", "site"},
    ),
    "mensagem_em_massa": (
        "Estilo de mensagem enviada em massa (saudação genérica, instruções para sair da lista)",
        {"dear", "kindly", "sir", "mrs", "hello", "well regard", "remove",
         "remove subject", "removal"},
    ),
    "produto_suspeito": (
        "Oferta de produtos típicos de spam (medicamentos, réplicas etc.)",
        {"pill", "viagra", "cialis", "rx", "medication", "prescription",
         "enhancement", "replica", "herbal", "med", "medical"},
    ),
}

_TERMO_PARA_CATEGORIA: dict[str, str] = {
    termo: cat for cat, (_, termos) in LEXICO_SEMANTICO.items() for termo in termos
}

_SENTENCAS = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Evidencia:
    categoria: str
    descricao: str
    termos: list[str]
    contribuicao: float            # soma de coef * x, em unidades de logit
    trecho: str | None = None


@dataclass
class ExplicacaoModelo:
    evidencias: list[Evidencia] = field(default_factory=list)
    # fração da contribuição positiva total que foi explicada pelo léxico (0-1).
    # Baixa = o modelo decidiu por termos que não sabemos traduzir.
    cobertura: float = 0.0
    logit: float = 0.0


def contribuicoes(X_linha, modelo) -> tuple[np.ndarray, np.ndarray, float]:
    """Retorna (índices, contribuições coef*x, logit) das features não-nulas."""
    linha = X_linha.tocsr()
    idx = linha.indices
    contrib = modelo.coef_[0][idx] * linha.data
    logit = float(modelo.intercept_[0] + contrib.sum())
    return idx, contrib, logit


def _localizar_trecho(texto: str, termos_peso: dict[str, float],
                      preprocess_fn: Callable[[str], str],
                      max_sentencas: int = 60, max_chars: int = 180) -> str | None:
    """
    Acha a sentença do e-mail original que concentra mais contribuição dos termos
    da categoria. Usa o MESMO preprocess do modelo em cada sentença, então o
    alinhamento de lemas é idêntico ao que o modelo enxergou.
    """
    melhor, melhor_peso = None, 0.0
    for sentenca in _SENTENCAS.split(texto)[:max_sentencas]:
        sentenca = sentenca.strip()
        if not sentenca:
            continue
        limpa = f" {preprocess_fn(sentenca)} "
        peso = sum(w for t, w in termos_peso.items() if f" {t} " in limpa)
        if peso > melhor_peso:
            melhor, melhor_peso = sentenca, peso
    if melhor is None:
        return None
    return melhor if len(melhor) <= max_chars else melhor[:max_chars].rstrip() + "…"


def explicar_por_modelo(X_linha, modelo, nomes_features: np.ndarray, texto_original: str,
                        preprocess_fn: Callable[[str], str], *, k: int = 3,
                        contrib_minima: float = 0.05,
                        ignorar: set[str] | None = None) -> ExplicacaoModelo:
    """
    nomes_features: vectorizer.get_feature_names_out() (apenas as colunas TF-IDF;
                    as colunas extras vêm depois e são ignoradas aqui).
    ignorar:        categorias que já foram explicadas pelas regras fixas.
    """
    ignorar = ignorar or set()
    idx, contrib, logit = contribuicoes(X_linha, modelo)

    n_texto = len(nomes_features)
    por_categoria: dict[str, list[tuple[str, float]]] = {}
    total_positivo = 0.0
    explicado = 0.0

    for j, c in zip(idx, contrib):
        if j >= n_texto or c <= 0:       # só termos de texto que empurram para "golpe"
            continue
        termo = str(nomes_features[j])
        total_positivo += c
        cat = _TERMO_PARA_CATEGORIA.get(termo)
        if cat is not None:
            explicado += c
            por_categoria.setdefault(cat, []).append((termo, float(c)))

    evidencias: list[Evidencia] = []
    for cat, itens in por_categoria.items():
        if cat in ignorar:
            continue
        soma = sum(c for _, c in itens)
        if soma < contrib_minima:
            continue
        itens.sort(key=lambda t: -t[1])
        termos = [t for t, _ in itens]
        evidencias.append(Evidencia(
            categoria=cat,
            descricao=LEXICO_SEMANTICO[cat][0],
            termos=termos,
            contribuicao=round(soma, 3),
            trecho=_localizar_trecho(texto_original, dict(itens), preprocess_fn),
        ))

    evidencias.sort(key=lambda e: -e.contribuicao)
    return ExplicacaoModelo(
        evidencias=evidencias[:k],
        cobertura=round(explicado / total_positivo, 3) if total_positivo else 0.0,
        logit=logit,
    )