"""
Utilitários de logging para os treinamentos de ML.

Gera dois tipos de registro, complementares:

1. Arquivo .log (texto corrido, via logging padrão do Python) — bom para
   acompanhar o que aconteceu numa rodada específica (prints organizados,
   com timestamp, indo pro console E pro arquivo ao mesmo tempo).

2. Um CSV "historico_experimentos.csv" com UMA LINHA POR MODELO TREINADO,
   contendo hiperparâmetros + métricas. Esse é o mais útil pro TCC: dá pra
   abrir no pandas/Excel depois e comparar todas as rodadas (LogReg, RF,
   MLP, cada configuração do Optuna, etc.) numa tabela só.
"""

import csv
import logging
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

HISTORICO_CSV = os.path.join(LOGS_DIR, "historico_experimentos.csv")


def configurar_logger(nome: str = "treinamento") -> logging.Logger:
    """
    Cria (ou reaproveita) um logger que escreve simultaneamente:
    - no console (pra você acompanhar em tempo real)
    - num arquivo logs/<nome>_AAAAMMDD.log (pra consultar depois)
    """
    logger = logging.getLogger(nome)

    if logger.handlers:
        # evita duplicar handlers se a função for chamada mais de uma vez
        return logger

    logger.setLevel(logging.INFO)

    data_hoje = datetime.now().strftime("%Y%m%d")
    caminho_log = os.path.join(LOGS_DIR, f"{nome}_{data_hoje}.log")

    formato = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler_arquivo = logging.FileHandler(caminho_log, encoding="utf-8")
    handler_arquivo.setFormatter(formato)
    logger.addHandler(handler_arquivo)

    handler_console = logging.StreamHandler()
    handler_console.setFormatter(formato)
    logger.addHandler(handler_console)

    return logger


def registrar_experimento(dados: dict, caminho_csv: str = HISTORICO_CSV) -> None:
    dados = {"timestamp": datetime.now().isoformat(timespec="seconds"), **dados}

    arquivo_existe = os.path.isfile(caminho_csv)
    with open(caminho_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dados.keys()))
        if not arquivo_existe:
            writer.writeheader()
        writer.writerow(dados)