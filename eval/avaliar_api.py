"""
Avalia o sistema em massa, batendo na API que ja esta rodando.

Por que pela API e nao chamando o modelo direto: o score que o usuario recebe
nao e so o predict_proba. O classifier.py soma um ajuste de remetente
(_analisar_remetente) por cima da probabilidade. Medir pela API e a unica
forma de medir o que o sistema realmente entrega.

Uso:
    docker run -p 8000:8000 detector-golpes          # em outro terminal
    python eval/avaliar_api.py --csv eval/dados/sonda_en.csv

O CSV precisa das colunas: body, label (1=golpe, 0=legitimo).
Opcionais: subject, sender.

Saidas:
    eval/saida/predicoes_<nome>.csv   score e faixa de cada e-mail
    eval/saida/erros_<nome>.csv       so os que o sistema errou
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

SAIDA = Path(__file__).parent / "saida"
# Faixas definidas em app/services/classifier.py
LIMIAR_ALTO, LIMIAR_MEDIO = 0.6, 0.3
csv.field_size_limit(10_000_000)


def faixa(score: float) -> str:
    if score >= LIMIAR_ALTO:
        return "alto"
    return "medio" if score >= LIMIAR_MEDIO else "baixo"


def classificar(url: str, linha: dict, timeout: float) -> dict:
    payload = {"email_text": linha.get("body", "")}
    if linha.get("subject"):
        payload["email_subject"] = linha["subject"]
    if linha.get("sender"):
        payload["sender"] = linha["sender"]

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def metricas(pares: list[tuple[int, int]]) -> dict:
    """pares = (rotulo_real, previsto). Classe positiva = 1 (golpe)."""
    vp = sum(1 for r, p in pares if r == 1 and p == 1)
    fp = sum(1 for r, p in pares if r == 0 and p == 1)
    fn = sum(1 for r, p in pares if r == 1 and p == 0)
    vn = sum(1 for r, p in pares if r == 0 and p == 0)
    prec = vp / (vp + fp) if vp + fp else 0.0
    rec = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (vp + vn) / len(pares) if pares else 0.0
    return {"vp": vp, "fp": fp, "fn": fn, "vn": vn,
            "precisao": prec, "recall": rec, "f1": f1, "acuracia": acc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="CSV rotulado a avaliar")
    ap.add_argument("--url", default="http://localhost:8000/analisar-email")
    ap.add_argument("--limite", type=int, help="avaliar so os N primeiros")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(f"CSV nao encontrado: {args.csv}")

    with args.csv.open(encoding="utf-8") as fh:
        linhas = list(csv.DictReader(fh))
    if args.limite:
        linhas = linhas[: args.limite]
    if "body" not in linhas[0] or "label" not in linhas[0]:
        sys.exit("O CSV precisa ter as colunas 'body' e 'label'.")

    print(f"Avaliando {len(linhas)} e-mails em {args.url}\n")
    resultados = []
    for i, linha in enumerate(linhas, 1):
        try:
            r = classificar(args.url, linha, args.timeout)
        except urllib.error.URLError as e:
            sys.exit(f"\nFalha ao chamar a API: {e}\n"
                     "A API esta rodando? Verifique: curl localhost:8000/health")
        resultados.append({
            "label": int(linha["label"]),
            "score": r["score"],
            "faixa": faixa(r["score"]),
            "classificacao": r["classificacao"],
            "subject": linha.get("subject", ""),
            "sender": linha.get("sender", ""),
            "body": linha.get("body", "")[:400],
            "padroes": "|".join(p["tipo"] for p in r.get("explicacao", [])),
        })
        if i % 25 == 0 or i == len(linhas):
            print(f"  {i}/{len(linhas)}", end="\r", flush=True)
    print("\n")

    # --- Distribuicao por faixa: e aqui que aparece o falso alarme ---
    print("Como cada classe real foi distribuida nas faixas de risco:\n")
    print(f"{'classe real':<16}{'baixo':>10}{'medio':>10}{'alto':>10}{'total':>9}")
    for rotulo, nome in ((0, "legitimo"), (1, "golpe")):
        grupo = [r for r in resultados if r["label"] == rotulo]
        if not grupo:
            continue
        cont = {f: sum(1 for r in grupo if r["faixa"] == f) for f in ("baixo", "medio", "alto")}
        linha_txt = "".join(f"{cont[f]:>6} {cont[f]/len(grupo):>3.0%}" for f in ("baixo", "medio", "alto"))
        print(f"{nome:<16}{linha_txt}{len(grupo):>9}")

    legitimos = [r for r in resultados if r["label"] == 0]
    if legitimos:
        alarme = sum(1 for r in legitimos if r["faixa"] != "baixo")
        print(f"\n  Falso alarme: {alarme}/{len(legitimos)} ({alarme/len(legitimos):.0%}) dos "
              "e-mails legitimos receberam algum alerta.")

    # --- Metricas nos dois pontos de operacao ---
    print("\nMetricas (classe positiva = golpe):\n")
    print(f"{'ponto de corte':<28}{'acuracia':>10}{'precisao':>10}{'recall':>9}{'F1':>8}")
    for nome, corte in (("score >= 0.6  (so 'alto')", LIMIAR_ALTO),
                        ("score >= 0.3  (alerta)", LIMIAR_MEDIO)):
        m = metricas([(r["label"], int(r["score"] >= corte)) for r in resultados])
        print(f"{nome:<28}{m['acuracia']:>10.3f}{m['precisao']:>10.3f}"
              f"{m['recall']:>9.3f}{m['f1']:>8.3f}")

    m = metricas([(r["label"], int(r["score"] >= LIMIAR_ALTO)) for r in resultados])
    print(f"\nMatriz de confusao no ponto do sistema (>= {LIMIAR_ALTO}):")
    print(f"  legitimos certos (VN): {m['vn']:>4}   |  legitimos marcados como golpe (FP): {m['fp']:>4}")
    print(f"  golpes perdidos  (FN): {m['fn']:>4}   |  golpes detectados            (VP): {m['vp']:>4}")

    # --- Arquivos ---
    SAIDA.mkdir(parents=True, exist_ok=True)
    nome = args.csv.stem
    campos = ["label", "score", "faixa", "classificacao", "padroes", "sender", "subject", "body"]

    def salvar(caminho: Path, dados: list[dict]) -> None:
        with caminho.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
            w.writeheader()
            w.writerows(dados)

    salvar(SAIDA / f"predicoes_{nome}.csv", resultados)
    erros = sorted(
        [r for r in resultados
         if (r["label"] == 0 and r["score"] >= LIMIAR_MEDIO) or
            (r["label"] == 1 and r["score"] < LIMIAR_ALTO)],
        key=lambda r: -abs(r["score"] - 0.5),
    )
    salvar(SAIDA / f"erros_{nome}.csv", erros)

    print(f"\nDetalhe por e-mail: eval/saida/predicoes_{nome}.csv")
    print(f"So os erros ({len(erros)}): eval/saida/erros_{nome}.csv")

    piores = [r for r in erros if r["label"] == 0][:5]
    if piores:
        print("\nLegitimos com maior score (os falsos alarmes mais graves):")
        for r in piores:
            print(f"  {r['score']:.3f}  {r['faixa']:<6} {r['subject'][:56]}")


if __name__ == "__main__":
    main()
