# import csv, requests

# URL = "http://localhost:8000/analisar-email"
# saida = []
# with open("emails_teste_manuais.csv", newline="", encoding="utf-8") as f:
#     for r in csv.DictReader(f):
#         resp = requests.post(URL, json={"email_text": r["email_text"]}, timeout=60).json()
#         linha = {"id": r["id"], "rotulo": r["rotulo"]}
#         for k, v in resp.items():
#             linha[k] = v if not isinstance(v, (list, dict)) else str(v)
#         saida.append(linha)

# campos = []
# for l in saida:
#     for k in l:
#         if k not in campos:
#             campos.append(k)

# with open("resultado_funcional.csv", "w", newline="", encoding="utf-8") as f:
#     w = csv.DictWriter(f, fieldnames=campos)
#     w.writeheader()
#     w.writerows(saida)

# print("Campos retornados:", campos)
# print(f"{len(saida)} e-mails processados")

import requests

URL = "http://localhost:8000/analisar-email"
base = ("Hi, how are you? Shall we meet for coffee tomorrow? I was thinking about "
        "the place near the park, around three in the afternoon, if that works for you.")

for n in [1, 2, 3, 5, 10]:
    texto = " ".join([base] * n)
    r = requests.post(URL, json={"email_text": texto}).json()
    print(f"{n:>2}x | {len(texto.split()):>4} palavras | score {r['score']:.4f} | {r['risco']}")