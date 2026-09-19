"""
Monta o conjunto de avaliacao externo: baixa, deduplica e DESCONTAMINA.

Por que descontaminar: o modelo foi treinado em data/phishing_email.csv, que
e a uniao exata de seis corpora publicos (CEAS_08, Enron, Ling, Nazario,
Nigerian_Fraud, SpamAssassin) - 82.486 linhas em ambos, mesma distribuicao de
rotulos. Qualquer dataset publico de phishing tende a reaproveitar essas
mesmas fontes. Medir em cima delas seria medir memorizacao.

Este script remove qualquer e-mail que compartilhe uma sequencia de 10
palavras com o corpus de treino. Na pratica isso descartou 49% das linhas
dos dois datasets usados.

Uso:
    python eval/preparar_externo.py --treino ~/Downloads/archive/phishing_email.csv

Depois:
    python eval/avaliar_api.py --csv eval/dados/externo_limpo.csv

Requer: pandas, pyarrow (pip install pandas pyarrow)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DADOS = Path(__file__).parent / "dados"

# Datasets externos usados. Ambos tem e-mails reais (~160 palavras de mediana).
FONTES = [
    {
        "arquivo": "hf_zefang.parquet",
        "url": "https://huggingface.co/api/datasets/zefang-liu/phishing-email-dataset/parquet/default/train/0.parquet",
        "col_texto": "Email Text",
        "col_rotulo": "Email Type",
        "eh_golpe": lambda v: str(v).strip() == "Phishing Email",
    },
    {
        "arquivo": "hf_cybersectony.parquet",
        "url": "https://huggingface.co/api/datasets/cybersectony/PhishingEmailDetectionv2.0/parquet/default/train/0.parquet",
        "col_texto": "content",
        "col_rotulo": "label",
        "eh_golpe": lambda v: int(v) == 1,
    },
]

K = 10        # tamanho da sequencia de palavras usada para detectar reuso
PASSO = 5     # no indice do treino, guarda 1 sequencia a cada 5 (economiza memoria)
MIN_PALAVRAS = 8


def norm(s) -> str:
    """Mesma normalizacao aplicada ao corpus de treino: minusculas, so alfanumerico."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split())


def sequencias(palavras: list[str], passo: int = 1):
    return (" ".join(palavras[i:i + K]) for i in range(0, len(palavras) - K + 1, passo))


def indexar_treino(caminho: Path) -> tuple[set, set]:
    print(f"Indexando o corpus de treino ({caminho.name})...")
    exatos, seqs = set(), set()
    for pedaco in pd.read_csv(caminho, chunksize=20000, on_bad_lines="skip", low_memory=False):
        coluna = "text_combined" if "text_combined" in pedaco.columns else pedaco.columns[0]
        for texto in pedaco[coluna].astype(str):
            n = norm(texto)
            exatos.add(n)
            seqs.update(hash(s) for s in sequencias(n.split(), passo=PASSO))
    print(f"  {len(exatos):,} e-mails, {len(seqs):,} sequencias indexadas\n")
    return exatos, seqs


def baixar(fonte: dict) -> Path:
    destino = DADOS / fonte["arquivo"]
    if destino.exists():
        print(f"  {fonte['arquivo']}: ja baixado")
        return destino
    print(f"  {fonte['arquivo']}: baixando...")
    DADOS.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(fonte["url"], destino)
    return destino


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--treino", type=Path, required=True,
                    help="CSV usado no treino (phishing_email.csv)")
    ap.add_argument("--saida", type=Path, default=DADOS / "externo_limpo.csv")
    args = ap.parse_args()

    if not args.treino.exists():
        sys.exit(f"Corpus de treino nao encontrado: {args.treino}")

    exatos, seqs = indexar_treino(args.treino)

    print("Baixando os datasets externos:")
    brutos: list[tuple[str, int]] = []
    for fonte in FONTES:
        caminho = baixar(fonte)
        df = pd.read_parquet(caminho)
        for texto, rotulo in zip(df[fonte["col_texto"]], df[fonte["col_rotulo"]]):
            texto = str(texto) if texto is not None else ""
            if len(texto.split()) < MIN_PALAVRAS:   # descarta URL solta e fragmento
                continue
            brutos.append((texto[:20000], int(fonte["eh_golpe"](rotulo))))
    print(f"\nlinhas brutas             : {len(brutos):,}")

    vistos, unicos = set(), []
    for texto, rotulo in brutos:
        chave = norm(texto)
        if chave not in vistos:
            vistos.add(chave)
            unicos.append((texto, rotulo))
    print(f"apos remover duplicatas   : {len(unicos):,}")

    def contaminado(texto: str) -> bool:
        n = norm(texto)
        if n in exatos:
            return True
        palavras = n.split()
        return len(palavras) >= K + 2 and any(hash(s) in seqs for s in sequencias(palavras))

    limpos = [(t, r) for t, r in unicos if not contaminado(t)]
    removidos = len(unicos) - len(limpos)
    print(f"apos remover contaminados : {len(limpos):,}  "
          f"(removidos {removidos:,} = {removidos/len(unicos):.0%})")

    golpes = sum(r for _, r in limpos)
    print(f"\ncomposicao final: {golpes:,} golpes | {len(limpos) - golpes:,} legitimos")

    with args.saida.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(["subject", "body", "sender", "label"])
        for texto, rotulo in limpos:
            escritor.writerow(["", texto, "", rotulo])
    print(f"salvo em {args.saida}")


if __name__ == "__main__":
    main()
